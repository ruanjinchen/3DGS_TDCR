#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from PIL import Image
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_image(image_path, white_background=False):
    if image_path.split(".")[-1] == "jpg":
        image_path = image_path.replace("jpg", "png")

    image = Image.open(image_path)

    im_data = np.array(image.convert("RGBA"))

    bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
    #
    norm_data = im_data / 255.0

    arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    # reconstruct the image to rgba
    arr = np.concatenate((arr, norm_data[:, :, 3:4]), axis=2)

    arr_u8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(arr_u8, mode="RGBA")


    return image
