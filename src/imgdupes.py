import multiprocessing
from argparse import ArgumentParser
from typing import List

import networkx
import networkx as nx
import numpy as np
import torch
from torchvision import transforms as pth_transforms

from src.files import walk_dir, walk_tree
from src.search import match
from src.vectorization import vectorize


def vectorization(images: List[str], queue: multiprocessing.Queue):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    transform = pth_transforms.Compose(
        [
            pth_transforms.Resize((448, 448)),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )
    if torch.cuda.is_available():
        print("Using CUDA for vectorization")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU for vectorization")
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    paths, embs = vectorize(
        image_paths=images,
        model=model,
        transform=transform,
        device=device,
        batch_size=8,
        num_workers=2,
        process_name="Vectorization",
    )
    norm = np.linalg.norm(embs, ord=2, axis=1, keepdims=True)
    embs = embs / (norm + 1e-6)
    queue.put([paths, embs])


def deduplicate(
    directory: str,
    recursive: bool = False
):
    if recursive:
        images = walk_tree(root=directory)
    else:
        images = walk_dir(root=directory)
    print(f"Found {len(images)} files")

    vectorization_queue = multiprocessing.Queue(maxsize=1)
    vectorization_process = multiprocessing.Process(
        target=vectorization,
        kwargs=dict(
            images=images,
            queue=vectorization_queue,
        )
    )
    vectorization_process.start()
    paths, embs = vectorization_queue.get()
    vectorization_process.join()
    vectorization_queue.close()

    print(f"Vectorized {len(paths)} files")
    idx_to_path = {i: path for i, path in enumerate(paths)}

    matched = match(embeddings=embs, n_neighbors=min(16, len(paths)))
    matched["pic_query"] = matched["pic_query"].map(idx_to_path)
    matched["pic_index"] = matched["pic_index"].map(idx_to_path)
    print(matched)
    matched = matched[matched["pic_index"] != matched["pic_query"]]
    matched = matched[matched["distance"] > 0.9]

    graph = nx.Graph()
    for _, row in matched.iterrows():
        graph.add_edge(row["pic_query"], row["pic_index"])
    for component in networkx.connected_components(graph):
        print("=" * 50)
        for file in component:
            print(file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="for given directory given follow subdirectories"
    )
    args = parser.parse_args()
    deduplicate(
        directory=args.directory, recursive=args.recursive
    )
