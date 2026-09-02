import gc
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from uav_visloc.dataset import UAVSingleDataset, UAVTripletDataset
from uav_visloc.utils import (
    get_top_similar,
    haversine_distance,
    visualize_top_k,
)


class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, learnable=True):
        super().__init__()

        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps)
        gem = torch.mean(x**self.p, dim=1) ** (1.0 / self.p)
        return gem


class DINOv2_Embedder(nn.Module):
    def __init__(self, embed_dim: int = 1024):
        super().__init__()

        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.gem = GeMPooling()
        self.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone.forward_features(x)["x_norm_patchtokens"]

        x = self.gem(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


def get_embeddings(
    dataloader: DataLoader, model: nn.Module, device: torch.device
) -> Tensor:
    model.eval()
    model.to(device)

    is_triplet = None
    anc_list, pos_list, neg_list = [], [], []
    single_list = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Calculating embeddings: ", leave=False):
            if is_triplet is None:
                is_triplet = len(batch) == 3

            if is_triplet:
                anc, pos, neg = batch

                combined = torch.cat([anc, pos, neg], dim=0).to(
                    device, non_blocking=True
                )
                combined_embs = model(combined)
                anc_emb, pos_emb, neg_emb = torch.chunk(combined_embs, 3, dim=0)

                anc_list.append(anc_emb.cpu())
                pos_list.append(pos_emb.cpu())
                neg_list.append(neg_emb.cpu())
            else:
                imgs = batch[0]

                imgs = imgs.to(device, non_blocking=True)
                embs = model(imgs)
                single_list.append(embs.cpu())

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if is_triplet:
        return (
            torch.cat(anc_list, dim=0),
            torch.cat(pos_list, dim=0),
            torch.cat(neg_list, dim=0),
        )
    else:
        return torch.cat(single_list, dim=0)


def train_model(
    model: nn.Module,
    train_dataset: UAVTripletDataset,
    test_dataset: UAVTripletDataset,
    device: torch.device,
    batch_size: int = 64,
    epoch: int = 10,
    model_statedict_filename: str | None = None,
):
    model.to(device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_params = [param for param in model.parameters() if param.requires_grad]

    optimizer = torch.optim.Adam(params=model_params)
    loss = torch.nn.TripletMarginLoss()

    for i in range(epoch):
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()

            batch_anc, batch_pos, batch_neg = batch
            combined = torch.cat([batch_anc, batch_pos, batch_neg], dim=0).to(
                device, non_blocking=True
            )

            combined_embs = model(combined)
            logits_anc, logits_pos, logits_neg = torch.chunk(combined_embs, 3, dim=0)

            loss_val = loss(logits_anc, logits_pos, logits_neg)
            total_train_loss += loss_val.item()
            loss_val.backward()
            optimizer.step()
        average_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_test_loss = 0

        with torch.no_grad():
            for batch in test_dataloader:
                batch_anc, batch_pos, batch_neg = batch
                combined = torch.cat([batch_anc, batch_pos, batch_neg], dim=0).to(
                    device, non_blocking=True
                )

                combined_embs = model(combined)
                logits_anc, logits_pos, logits_neg = torch.chunk(
                    combined_embs, 3, dim=0
                )

                loss_val = loss(logits_anc, logits_pos, logits_neg)
                total_test_loss += loss_val.item()
        average_test_loss = total_test_loss / len(test_dataloader)

        print(
            f"Epoch[{i + 1}] Test loss: {average_test_loss}; Train loss: {average_train_loss}"
        )

    torch.save(
        model.state_dict(),
        f"dinov2_{model_statedict_filename if model_statedict_filename else datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
    )
    return model


def get_refined_location(
    extractor, matcher, drone_img_pil, candidate_tiles, src_raster, device
):
    drone_tensor = TF.to_tensor(drone_img_pil).unsqueeze(0).to(device)
    feats0 = extractor.extract(drone_tensor)
    w_d, h_d = drone_img_pil.size

    best_match = {
        "inliers": 0,
        "lat": None,
        "lon": None,
        "tile_id": None,
        "center_in_tile": None,
        "pts0": None,
        "pts1": None,
        "is_fallback": False,
    }

    for tile_info in candidate_tiles:
        tile_data = src_raster.read(window=tile_info["window_obj"])[:3, ...]
        tile_tensor = TF.to_tensor(tile_data.transpose(1, 2, 0)).unsqueeze(0).to(device)

        feats1 = extractor.extract(tile_tensor)

        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0_m, feats1_m = rbd(feats0), rbd(feats1)
        matches = matches01["matches"][0]
        points0, points1 = (
            feats0_m["keypoints"][matches[..., 0]],
            feats1_m["keypoints"][matches[..., 1]],
        )

        if len(points0) < 15:
            continue

        H, mask = cv2.findHomography(
            points0.cpu().numpy(), points1.cpu().numpy(), cv2.USAC_MAGSAC, 2.0
        )

        if mask is not None:
            inliers_count = mask.sum()
            if inliers_count > best_match["inliers"]:
                inliers_mask = mask.ravel() == 1
                center_drone = np.array([[w_d / 2, h_d / 2]], dtype="float32").reshape(
                    -1, 1, 2
                )

                try:
                    center_in_tile = cv2.perspectiveTransform(center_drone, H).reshape(
                        -1
                    )
                    target_x_px, target_y_px = (
                        tile_info["x_px"] + center_in_tile[0],
                        tile_info["y_px"] + center_in_tile[1],
                    )
                    lon, lat = rasterio.transform.xy(
                        src_raster.transform, target_y_px, target_x_px
                    )

                    best_match = {
                        "inliers": inliers_count,
                        "lat": lat,
                        "lon": lon,
                        "tile_id": tile_info["tile_id"],
                        "center_in_tile": center_in_tile,
                        "pts0": points0.cpu().numpy()[inliers_mask],
                        "pts1": points1.cpu().numpy()[inliers_mask],
                        "is_fallback": False,
                    }
                except:
                    continue

    if best_match["lat"] is None and len(candidate_tiles) > 0:
        top1 = candidate_tiles[0]
        best_match.update(
            {
                "inliers": 0,
                "lat": top1["center_lat"],
                "lon": top1["center_lon"],
                "tile_id": top1["tile_id"],
                "center_in_tile": np.array([top1["size"] / 2.0, top1["size"] / 2.0]),
                "is_fallback": True,
            }
        )

    return best_match


def evaluate_model(
    model: nn.Module,
    extractor: nn.Module,
    matcher: nn.Module,
    top_k: int,
    dataset_path: Path,
    test_folders: list[str],
    scales: list[int],
    device: torch.device,
) -> pd.DataFrame:
    all_predictions = []
    top1_hits = []
    top5_hits = []
    topK_hits = []

    for folder in test_folders:
        drone_dataset = UAVSingleDataset(
            dataset_path=dataset_path, folder_list=[folder], is_tile=False
        )
        drone_dataloader = DataLoader(
            dataset=drone_dataset, batch_size=32, shuffle=False
        )
        drone_embeddings = get_embeddings(
            dataloader=drone_dataloader, model=model, device=device
        )

        satelite_dataset = UAVSingleDataset(
            dataset_path=dataset_path, folder_list=[folder], is_tile=True, scales=scales
        )
        satelite_dataloader = DataLoader(
            dataset=satelite_dataset, batch_size=32, shuffle=False
        )
        satelite_embeddings = get_embeddings(
            dataloader=satelite_dataloader, model=model, device=device
        )

        _, topk_indices = get_top_similar(
            drone_embeddings=drone_embeddings,
            satelite_embeddings=satelite_embeddings,
            top_k=top_k,
        )

        sat_raster_path = satelite_dataset.items[0]["raster_path"]

        with rasterio.open(sat_raster_path) as src_raster:
            for idx in tqdm(range(len(drone_dataset)), desc="Calculating metrics: "):
                candidate_indices = topk_indices[idx].tolist()
                candidates = [
                    satelite_dataset.items[t_idx] for t_idx in candidate_indices
                ]

                gt_row = drone_dataset.items[idx]
                gt_y_px, gt_x_px = rasterio.transform.rowcol(
                    src_raster.transform, gt_row["lon"], gt_row["lat"]
                )

                def is_hit(tile_item):
                    x_start, y_start = tile_item["x_px"], tile_item["y_px"]
                    size = tile_item["size"]
                    return (x_start <= gt_x_px <= x_start + size) and (
                        y_start <= gt_y_px <= y_start + size
                    )

                top1_hits.append(
                    is_hit(satelite_dataset.items[candidate_indices[0]])
                    if candidate_indices
                    else False
                )
                top5_hits.append(
                    any(
                        is_hit(satelite_dataset.items[t_idx])
                        for t_idx in candidate_indices[: min(5, len(candidate_indices))]
                    )
                )
                topK_hits.append(
                    any(
                        is_hit(satelite_dataset.items[t_idx])
                        for t_idx in candidate_indices
                    )
                )

                with Image.open(drone_dataset.items[idx]["img_path"]) as d_img:
                    res = get_refined_location(
                        extractor, matcher, d_img, candidates, src_raster, device
                    )

                error_m = None
                if gt_row and res["lat"] is not None:
                    error_m = haversine_distance(
                        res["lat"],
                        res["lon"],
                        gt_row["lat"],
                        gt_row["lon"],
                    )

                all_predictions.append(
                    {
                        "file": drone_dataset.items[idx]["filename"],
                        "lat": res["lat"],
                        "lon": res["lon"],
                        "inliers": res["inliers"],
                        "is_fallback": res["is_fallback"],
                        "error_m": error_m,
                    }
                )

    valid_errors = np.array(
        [pred["error_m"] for pred in all_predictions if pred.get("error_m") is not None]
    )

    mean_err = float(np.mean(valid_errors)) if len(valid_errors) > 0 else 0.0
    median_err = float(np.median(valid_errors)) if len(valid_errors) > 0 else 0.0
    p90_err = float(np.percentile(valid_errors, 90)) if len(valid_errors) > 0 else 0.0

    acc_50m = float(np.mean(valid_errors <= 50) * 100) if len(valid_errors) > 0 else 0.0
    acc_100m = (
        float(np.mean(valid_errors <= 100) * 100) if len(valid_errors) > 0 else 0.0
    )
    acc_500m = (
        float(np.mean(valid_errors <= 500) * 100) if len(valid_errors) > 0 else 0.0
    )

    top1_recall = float(np.mean(top1_hits) * 100) if len(top1_hits) > 0 else 0.0
    top5_recall = float(np.mean(top5_hits) * 100) if len(top5_hits) > 0 else 0.0
    topK_recall = float(np.mean(topK_hits) * 100) if len(topK_hits) > 0 else 0.0

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Mean Error (m)",
                "Median Error (m)",
                "90th Percentile Error (m)",
                "Accuracy <= 50m (%)",
                "Accuracy <= 100m (%)",
                "Accuracy <= 500m (%)",
                "Top-1 Recall (%)",
                "Top-5 Recall (%)",
                f"Top-{top_k} Recall (%)",
            ],
            "Value": [
                round(mean_err, 2),
                round(median_err, 2),
                round(p90_err, 2),
                round(acc_50m, 2),
                round(acc_100m, 2),
                round(acc_500m, 2),
                round(top1_recall, 2),
                round(top5_recall, 2),
                round(topK_recall, 2),
            ],
        }
    )
    return metrics_df


def visualize_refined_matches(
    num_samples: int,
    extractor: nn.Module,
    matcher: nn.Module,
    drone_dataset: Dataset,
    satelite_dataset: Dataset,
    topk_indices: Tensor,
    device: torch.device,
    drone_res_height: int,
    drone_res_width: int,
):
    random_indices = np.random.choice(
        len(drone_dataset), size=num_samples, replace=False
    )

    sat_raster_path = satelite_dataset.items[0]["raster_path"]

    _, axes = plt.subplots(num_samples, 2, figsize=(15, 7 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    with rasterio.open(sat_raster_path) as src_raster:
        for i, idx in tqdm(
            enumerate(random_indices), desc="Refining & Visualizing", total=num_samples
        ):
            drone_item = drone_dataset.items[idx]
            drone_img_path = drone_item["img_path"]

            with Image.open(drone_img_path) as d_img:
                drone_img = TF.resize(
                    d_img.convert("RGB"), (drone_res_height, drone_res_width)
                )

            candidate_tile_indices = topk_indices[idx].tolist()
            candidates = [
                satelite_dataset.items[t_idx] for t_idx in candidate_tile_indices
            ]

            res = get_refined_location(
                extractor=extractor,
                matcher=matcher,
                drone_img_pil=drone_img,
                candidate_tiles=candidates,
                src_raster=src_raster,
                device=device,
            )

            axes[i, 0].imshow(drone_img)
            if res["pts0"] is not None and len(res["pts0"]) > 0:
                axes[i, 0].scatter(
                    res["pts0"][:, 0], res["pts0"][:, 1], c="lime", s=15, alpha=0.7
                )
            axes[i, 0].set_title(f"Drone: {drone_img_path.name}")
            axes[i, 0].axis("off")

            if res["tile_id"] is not None:
                matched_tile_info = next(
                    t for t in candidates if t["tile_id"] == res["tile_id"]
                )
                tile_img = src_raster.read(window=matched_tile_info["window_obj"])[
                    :3, ...
                ].transpose(1, 2, 0)

                axes[i, 1].imshow(tile_img)

                if res["pts1"] is not None and len(res["pts1"]) > 0:
                    axes[i, 1].scatter(
                        res["pts1"][:, 0],
                        res["pts1"][:, 1],
                        c="lime",
                        s=15,
                        alpha=0.7,
                        label="Inliers",
                    )

                if res["center_in_tile"] is not None:
                    axes[i, 1].scatter(
                        res["center_in_tile"][0],
                        res["center_in_tile"][1],
                        c="red",
                        marker="x",
                        s=200,
                        linewidths=3,
                        label="Predicted Center",
                    )

                axes[i, 1].set_title(
                    f"Match (Scale: {matched_tile_info['size']}, Inliers: {res['inliers']}, Lat: {res['lat']:.5f}, Lon: {res['lon']:.5f})"
                )
                axes[i, 1].legend(loc="upper right")
            else:
                axes[i, 1].set_title("No match found")

            axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(PROJECT_ROOT / "config.yaml") as config_file:
        config = yaml.safe_load(config_file)
        dataset_link = config["dataset_link"]
        data_folder = config["data_folder"]
        raw_folder = config["raw_folder"]
        zip_file_name = config["zip_file_name"]
        extract_folder = config["extract_folder"]

        train_folders = config["train_folders"]
        test_folders = config["test_folders"]
        tile_scales = config["tile_scales"]

        seed = config["seed"]

        drone_res_height = config["drone_res_height"]
        drone_res_width = config["drone_res_width"]

        top_k = int(config["top_k"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = DINOv2_Embedder()
    model.to(device)

    dataset_path = PROJECT_ROOT / data_folder / raw_folder / extract_folder
    trained_model = train_model(
        model=model,
        train_dataset=UAVTripletDataset(
            dataset_path=dataset_path, folder_list=train_folders
        ),
        test_dataset=UAVTripletDataset(
            dataset_path=dataset_path, folder_list=test_folders
        ),
        device=device,
    )

    extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    for folder in test_folders:
        drone_dataset = UAVSingleDataset(
            dataset_path=dataset_path, folder_list=[folder], is_tile=False
        )
        drone_dataloader = DataLoader(
            dataset=drone_dataset, batch_size=32, shuffle=False
        )
        drone_embeddings = get_embeddings(
            dataloader=drone_dataloader, model=trained_model, device=device
        )

        satelite_dataset = UAVSingleDataset(
            dataset_path=dataset_path,
            folder_list=[folder],
            is_tile=True,
            scales=tile_scales,
        )
        satelite_dataloader = DataLoader(
            dataset=satelite_dataset, batch_size=32, shuffle=False
        )
        satelite_embeddings = get_embeddings(
            dataloader=satelite_dataloader, model=trained_model, device=device
        )

        topk_scores, topk_indices = get_top_similar(
            drone_embeddings=drone_embeddings,
            satelite_embeddings=satelite_embeddings,
            top_k=top_k,
        )

        visualize_top_k(
            num_samples=3,
            drone_dataset=drone_dataset,
            satelite_dataset=satelite_dataset,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
        )

        visualize_refined_matches(
            num_samples=5,
            extractor=extractor,
            matcher=matcher,
            drone_dataset=drone_dataset,
            satelite_dataset=satelite_dataset,
            topk_indices=topk_indices,
            device=device,
            drone_res_height=drone_res_height,
            drone_res_width=drone_res_width,
        )
