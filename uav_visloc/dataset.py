import random
import shutil
from pathlib import Path

import pandas as pd
import rasterio
import requests
import torchvision.transforms as T
import yaml
from PIL import Image
from rasterio.windows import Window
from torch.utils.data import Dataset
from tqdm import tqdm

from uav_visloc.utils import cut_into_tiles


class UAVSingleDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        folder_list: list,
        is_tile: bool = False,
        drone_size: tuple = (1036, 1540),
        scales: list = (518, 1036, 1554),
    ):
        self.dataset_path = dataset_path
        self.folder_list = folder_list
        self.is_tile = is_tile
        self.items = []

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        if self.is_tile:
            self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
        else:
            self.transform = T.Compose(
                [T.Resize(drone_size), T.ToTensor(), T.Normalize(mean=mean, std=std)]
            )

        for folder in folder_list:
            folder_str = f"{int(folder):02d}"
            folder_dir = self.dataset_path / folder_str

            if self.is_tile:
                raster_files = list(folder_dir.glob("*.tif"))
                if not raster_files:
                    continue

                raster_path = raster_files[0]
                tiles = cut_into_tiles(raster_path, scales=scales)

                for tile in tiles:
                    tile["raster_path"] = raster_path
                    tile["folder"] = folder_str
                    self.items.append(tile)
            else:
                csv_path = folder_dir / f"{folder_str}.csv"
                if not csv_path.exists():
                    continue

                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    self.items.append(
                        {
                            "img_path": folder_dir / "drone" / row["filename"],
                            "filename": row["filename"],
                            "folder": folder_str,
                            "lat": row["lat"],
                            "lon": row["lon"],
                        }
                    )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]

        if self.is_tile:
            with rasterio.open(item["raster_path"]) as src:
                tile_data = src.read(window=item["window_obj"])[:3, ...].transpose(
                    1, 2, 0
                )
                img = Image.fromarray(tile_data)
            tensor_img = self.transform(img)

            meta = {
                "tile_id": item["tile_id"],
                "folder": item["folder"],
                "raster_path": str(item["raster_path"]),
                "x_px": item["x_px"],
                "y_px": item["y_px"],
                "size": item["size"],
                "center_lat": item["center_lat"],
                "center_lon": item["center_lon"],
            }
            return tensor_img, meta
        else:
            img = Image.open(item["img_path"]).convert("RGB")
            tensor_img = self.transform(img)

            meta = {
                "filename": item["filename"],
                "folder": item["folder"],
                "lat": item["lat"],
                "lon": item["lon"],
            }
            return tensor_img, meta


class UAVTripletDataset(Dataset):
    def __init__(self, dataset_path: Path, folder_list: list, crop_size: int = 518):
        self.crop_size = crop_size
        self.dataset_path = dataset_path

        dfs = []
        for folder in folder_list:
            folder_str = f"{int(folder):02d}"
            csv_file = self.dataset_path / folder_str / f"{folder_str}.csv"
            df_tmp = pd.read_csv(csv_file)
            df_tmp["folder"] = folder_str
            dfs.append(df_tmp)
        self.df = pd.concat(dfs, ignore_index=True)

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        self.drone_transform = T.Compose(
            [
                T.Resize((crop_size, crop_size)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomRotation(15),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

        self.sat_transform = T.Compose(
            [
                T.Resize((crop_size, crop_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self):
        return len(self.df)

    def _crop_sat_patch(self, src, x_px, y_px):
        x_min = max(0, int(x_px - self.crop_size // 2))
        y_min = max(0, int(y_px - self.crop_size // 2))
        x_min = min(x_min, src.width - self.crop_size)
        y_min = min(y_min, src.height - self.crop_size)

        window = Window(x_min, y_min, self.crop_size, self.crop_size)
        crop = src.read(window=window)[:3, ...].transpose(1, 2, 0)
        return Image.fromarray(crop)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder = row["folder"]

        drone_path = self.dataset_path / folder / "drone" / row["filename"]
        anchor_img = Image.open(drone_path).convert("RGB")
        anchor_tensor = self.drone_transform(anchor_img)

        sat_path = self.dataset_path / folder / f"satellite{folder}.tif"

        with rasterio.open(sat_path) as src:
            gt_y_px, gt_x_px = rasterio.transform.rowcol(
                src.transform, row["lon"], row["lat"]
            )

            pos_img = self._crop_sat_patch(src, gt_x_px, gt_y_px)
            positive_tensor = self.sat_transform(pos_img)

            neg_x, neg_y = gt_x_px, gt_y_px
            for _ in range(50):
                neg_x = random.randint(
                    self.crop_size // 2, src.width - self.crop_size // 2
                )
                neg_y = random.randint(
                    self.crop_size // 2, src.height - self.crop_size // 2
                )
                if (
                    abs(neg_x - gt_x_px) >= self.crop_size
                    or abs(neg_y - gt_y_px) >= self.crop_size
                ):
                    break

            neg_img = self._crop_sat_patch(src, neg_x, neg_y)
            negative_tensor = self.sat_transform(neg_img)

        return anchor_tensor, positive_tensor, negative_tensor


def download_dataset(url: str, destination_path: Path, chunk_size: int = 8192):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        with open(destination_path, "wb") as file:
            iter_file = response.iter_content(chunk_size=chunk_size)
            for chunk in tqdm(iter_file, desc="Downloading dataset: "):
                if chunk:
                    file.write(chunk)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parents[1]

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

    random.seed(seed)

    destination_file = PROJECT_ROOT / data_folder / raw_folder / zip_file_name

    if not destination_file.exists():
        destination_file.parent.mkdir(parents=True, exist_ok=True)

        download_dataset(url=dataset_link, destination_path=destination_file)
        print("Dataset downloaded successfully")
    else:
        print("Dataset already downloaded")

    extract_path = PROJECT_ROOT / data_folder / raw_folder / extract_folder
    shutil.unpack_archive(destination_file, extract_path)
