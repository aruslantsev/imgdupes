from typing import List, Tuple, Optional, Callable, Union

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        transforms: Callable[..., torch.Tensor],
    ):
        self.image_paths = image_paths
        self.length = len(image_paths)
        self.transforms = transforms

    def __len__(self):
        return self.length

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[Optional[str], Optional[torch.tensor]]:
        image_path = self.image_paths[idx]

        try:
            image = Image.open(image_path)
            image.load()
        except Exception:
            return None, None

        if image is not None:
            if image.mode in ("1", "L"):
                image = image.convert("RGB")
            if image.mode != "RGB":
                image = image.convert("RGBA")
                back = Image.new("RGB", image.size, (255, 255, 255))
                back.paste(image, mask=image)
                image = back
            image = self.transforms(image).float()

        return image_path, image


def images_collate(batch):
    batch_paths = []
    batch_images = []
    for image_path, image in batch:
        if image is not None:
            batch_paths.append(image_path)
            batch_images.append(image)
    if len(batch_paths) == 0:
        return None, None
    return batch_paths, torch.stack(batch_images)


@torch.no_grad()
def vectorize(
    image_paths: List[str],
    model: Union[torch.nn.Module, Callable],
    transform: Callable[..., torch.Tensor],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    process_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = ImageDataset(
        image_paths=image_paths,
        transforms=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=images_collate,
        shuffle=False,
        num_workers=num_workers,
    )

    all_paths = []
    all_embeddings = []

    bar = tqdm(
        desc=process_name,
        total=len(dataset),
        mininterval=10,
        # Do not let tqdm readjust update interval
        maxinterval=float("inf"),
    )
    for i, (batch_image_paths, batch_images) in enumerate(dataloader):
        if batch_image_paths is None:
            tqdm.write("Got empty batch when vectorizing images")
        else:
            batch_images = batch_images.to(device)
            batch_embeddings = model(batch_images).detach().cpu().numpy()
            all_paths.append(batch_image_paths)
            all_embeddings.append(batch_embeddings)
        bar.update(n=min(batch_size, len(dataset) - batch_size * i))

    if len(all_embeddings) == 0:
        return np.ndarray(0), np.ndarray((0, 0))

    all_paths = np.hstack(all_paths)
    all_embeddings = np.vstack(all_embeddings)

    return all_paths, all_embeddings

