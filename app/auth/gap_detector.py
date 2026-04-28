"""
滑块验证码缺口位置检测
两种方案组合：
1. 轮廓特征匹配（Canny + 形态学闭运算 + 轮廓筛选）—— 主方案，大部分场景可用
2. Sobel连续性 × 梯度强度 —— 备用方案，云朵等高融合场景
"""

import cv2
import numpy as np


def detect_gap_contour(bg_image: np.ndarray) -> int | None:
    """轮廓特征匹配（test.py方案）"""
    height, width = bg_image.shape[:2]
    gray = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        if bbox_area == 0:
            continue
        area = cv2.contourArea(contour)
        extent = area / bbox_area

        if 30 < w < int(width * 0.4) and 30 < h < int(height * 0.4):
            aspect_ratio = float(w) / h
            if 0.7 < aspect_ratio < 1.3 and area > 1000:
                if extent > 0.4:
                    if x > width * 0.15 and (y + h) < height - 10:
                        score = abs(aspect_ratio - 1.0) + (1.0 - extent)
                        candidates.append((score, x))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
    return None


def detect_gap_sobel(bg_image: np.ndarray) -> int | None:
    """Sobel连续性 × 梯度强度（备用方案）"""
    gray = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel = np.abs(sobel_x).astype(np.uint8)

    best_x = None
    best_score = 0

    for x in range(50, min(280, bg_image.shape[1] - 10)):
        col = abs_sobel[20:180, x] if bg_image.shape[0] > 180 else abs_sobel[:, x]
        binary = col > 20
        max_consecutive = 0
        current = 0
        for val in binary:
            if val:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        mean_grad = np.mean(col)
        score = max_consecutive * mean_grad

        if score > best_score:
            best_score = score
            best_x = x

    return best_x


def detect_gap(bg_image: np.ndarray) -> int | None:
    """组合方案：先轮廓匹配，失败则用Sobel"""
    x = detect_gap_contour(bg_image)
    if x is not None:
        return x
    return detect_gap_sobel(bg_image)


def detect_gap_from_bytes(image_bytes: bytes) -> int | None:
    """从图片字节流检测缺口位置"""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return detect_gap(img)
