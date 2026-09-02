from pathlib import Path

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from PIL import Image
from rasterio.windows import Window
from torch import Tensor, topk
from torch.nn import functional as F
from torch.utils.data import Dataset


def cut_into_tiles(satelite_path: Path, scales: tuple[int]):
    folder_name = satelite_path.parent.name
    tiles = []
    with rasterio.open(satelite_path) as src:
        width = src.width
        height = src.height
        transform = src.transform

        print(f"[{folder_name}] Map size: {width}x{height} pixels")

        for tile_size in scales:
            stride = tile_size // 2

            for x_start in range(0, width - tile_size + 1, stride):
                for y_start in range(0, height - tile_size + 1, stride):
                    window = Window(x_start, y_start, tile_size, tile_size)

                    x_center_px = x_start + tile_size // 2
                    y_center_px = y_start + tile_size // 2

                    lon, lat = rasterio.transform.xy(
                        transform, y_center_px, x_center_px
                    )

                    tiles.append(
                        {
                            "tile_id": f"tile_{tile_size}_{x_start}_{y_start}",
                            "x_px": x_start,
                            "y_px": y_start,
                            "size": tile_size,
                            "center_x_px": x_center_px,
                            "center_y_px": y_center_px,
                            "center_lat": lat,
                            "center_lon": lon,
                            "window_obj": window,
                        }
                    )

        print(f"[{folder_name}] Tiles: {len(tiles)}")
        return tiles


def get_top_similar(drone_embeddings: Tensor, satelite_embeddings: Tensor, top_k: int):
    drone_embs_norm = F.normalize(drone_embeddings, p=2, dim=1)
    satelite_embs_norm = F.normalize(satelite_embeddings, p=2, dim=1)

    sim_matrix = drone_embs_norm @ satelite_embs_norm.T

    k = min(top_k, satelite_embeddings.size(0))
    topk_scores, topk_indices = topk(sim_matrix, k=k, dim=1)
    return topk_scores, topk_indices


def visualize_top_k(
    num_samples: int,
    drone_dataset: Dataset,
    satelite_dataset: Dataset,
    topk_indices: Tensor,
    topk_scores: Tensor,
    topm_to_show: int = 3,
):
    random_indices = np.random.choice(
        len(drone_dataset), size=num_samples, replace=False
    )

    _, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, drone_idx in enumerate(random_indices):
        drone_img_path = drone_dataset.items[drone_idx]["img_path"]
        with Image.open(drone_img_path) as d_img:
            axes[row_idx, 0].imshow(d_img)
            axes[row_idx, 0].set_title(f"Image: {drone_img_path.name}", fontsize=10)
            axes[row_idx, 0].axis("off")

        top_m_tile_indices = topk_indices[drone_idx, :topm_to_show]

        for col_idx, tile_idx in enumerate(top_m_tile_indices):
            tile_info = satelite_dataset.items[tile_idx.item()]
            with rasterio.open(tile_info["raster_path"]) as src:
                tile_data = src.read(window=tile_info["window_obj"]).transpose(1, 2, 0)

            score = topk_scores[drone_idx, col_idx].item()
            axes[row_idx, col_idx + 1].imshow(tile_data)
            axes[row_idx, col_idx + 1].set_title(
                f"Top-{col_idx + 1} (Similarity: {score:.3f})", fontsize=10
            )
            axes[row_idx, col_idx + 1].axis("off")

    plt.tight_layout()
    plt.show()


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c
