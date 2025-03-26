# utils/edge.py

import torch
import numpy as np
import cv2

def rgb_to_gray_tensor(image_tensor):
    """RGB [3, H, W] → Gray [H, W] (0~1 범위 유지)"""
    assert image_tensor.shape[0] == 3, "Input must be 3-channel RGB tensor"
    return image_tensor[0]*0.2989 + image_tensor[1]*0.5870 + image_tensor[2]*0.1140

def extract_edges(gray_tensor, method='canny', low_thresh=100, high_thresh=200):
    """
    엣지 검출: gray_tensor은 [H, W] 텐서 (0~1), 반환값은 (y, x) 좌표 리스트
    """
    img_np = gray_tensor.numpy()
    img_np = (img_np * 255).astype(np.uint8)

    if method == 'canny':
        edges = cv2.Canny(img_np, low_thresh, high_thresh)
    elif method == 'sobel':
        grad_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.hypot(grad_x, grad_y).astype(np.uint8)
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown edge detection method: {method}")

    edge_coords = np.argwhere(edges > 0)  # (y, x)
    return edge_coords

def get_edge_gradient_direction(gray_tensor):
    """
    Gray Tensor (H, W)에서 각 픽셀의 gradient 방향(rad)을 반환
    반환 shape: [H, W], 값은 (-pi, pi)
    """
    img_np = gray_tensor.numpy()
    img_np = (img_np * 255).astype(np.uint8)

    grad_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)

    angle = np.arctan2(grad_y, grad_x)  # [-pi, pi]
    return angle  # shape: (H, W)
import math

def get_neighbor_pixels_by_gradient(y, x, angle, step=1, H=32, W=32):
    """
    엣지 위치 (y, x), 방향 angle(rad)에 따라
    양 옆 픽셀 좌표를 반환 → [(y1, x1), (y2, x2)]
    """
    dx = math.cos(angle)
    dy = math.sin(angle)

    y1 = int(round(y - step * dy))
    x1 = int(round(x - step * dx))
    y2 = int(round(y + step * dy))
    x2 = int(round(x + step * dx))

    if 0 <= y1 < H and 0 <= x1 < W and 0 <= y2 < H and 0 <= x2 < W:
        return (y1, x1), (y2, x2)
    else:
        return None
