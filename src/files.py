import os

IMAGES = {".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".png"}


def walk_tree(root):
    images = []
    for directory, _, files in os.walk(root):
        for file in files:
            if os.path.splitext(file)[-1].lower() in IMAGES:
                images.append(os.path.join(directory, file))
    return images


def walk_dir(root):
    return [
        os.path.join(root, file) for file in os.listdir(root)
        if (
            os.path.isfile(os.path.join(root, file))
            and os.path.splitext(file)[-1].lower() in IMAGES
        )
    ]
