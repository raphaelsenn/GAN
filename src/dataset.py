"""
Author: Raphael Senn <raphaelsenn@gmx.de>
Initial coding: 2025-07-14
"""
import os
from typing import Callable, Any

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from ultralytics import YOLO


def detect_and_crop_faces(
        root: str, 
        src_dir: str, 
        dst_dir: str,
        model_path: str,
        conf: float=0.8,
    ) -> None:
    """
    Very simple function to detect and crop faces using a YOLO model (face detector).
    It creates a new folder and stores the cropped faces inside that folder,
    this is done for all faces of CelebFaces (where the YOLO model detects a face w.r.t conf).

    Parameters:
    -----------
    root : str
        Root path to the CelebFaces dataset.

    src_dir : str
        Folder name containing the orignal images (inside CelebFaces).

    dst_dir : str
        Folder name storing the cropped images (inside CelebFaces).

    model_path : str
        Path to the weights (.pt) of a YOLO faces detector model.

    conf : float (default=0.8)
        Confidence threshold of the YOLO model.

    Personal comment:
    ----------------
    Q1: Why doing this? 
    Q2: What is the purpose of this function?

    A: In the original GAN paper (Goodfellow et al., 2014), the authors used the TorontoFacesDataset (TFD),
    which is similar to CelebFaces. HOWEVER, the TFD images are much more tightly cropped,
    making it easier for neural networks to learn the facial structures.
    This function aims to replicate the TFD dataset.
    """
    src = os.path.join(root, src_dir)
    dst = os.path.join(root, dst_dir)
    assert  os.path.exists(src), f"{src} does not exist"

    os.makedirs(dst, exist_ok=True)
    images = sorted(os.listdir(src))

    model = YOLO(model_path)

    image_ids = []
    for img_name in images:
        image = Image.open(os.path.join(src, img_name))
        result = model.predict(image, conf=conf, verbose=False)[0]
        if not result.boxes:
            continue

        best_idx = torch.argmax(result.boxes.conf, dim=0)
        x1, y1, x2, y2 = map(int, result.boxes.xyxy[best_idx])
        image = image.crop((x1, y1, x2, y2))
        
        image.save(os.path.join(dst, img_name))
        image_ids.append(img_name)
    df_landmarks = pd.DataFrame({'image_id': image_ids})
    df_landmarks.to_csv(os.path.join(root, 'preprocessed_landmarks.csv'), index=False)


class CelebFaces(Dataset):
    """
    PyTorch Dataset wrapper for the CelebFaces (CelebA) dataset.

    Reference
    ---------
    CelebFaces Attributes Dataset (CelebA):
    Liu, Ziwei, et al. "Deep learning face attributes in the wild."
    Proceedings of the IEEE International Conference on Computer Vision. 2015.
    https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    """
    def __init__(
            self, 
            root_dir: str,
            data_folder: str,
            landmarks_file: str,
            transform: Callable |None=None
        ) -> None:
        self.root_dir = root_dir
        self.data_folder = data_folder
        self.landmakrs_file = landmarks_file
        self.df_landmarks = pd.read_csv(os.path.join(root_dir, landmarks_file))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df_landmarks)

    def __getitem__(self, index: int) -> Any:
        if torch.is_tensor(index):
            index = index.item()

        image_name = os.path.join(self.root_dir, self.data_folder, self.df_landmarks.iloc[index, 0])
        image = Image.open(image_name)

        if self.transform:
            image = self.transform(image)
        return image