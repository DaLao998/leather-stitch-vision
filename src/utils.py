from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def odd_at_least(value: int, minimum: int) -> int:
    value = max(int(value), int(minimum))
    if value % 2 == 0:
        value += 1
    return value


def smooth_projection(values: np.ndarray, window: int) -> np.ndarray:
    window = odd_at_least(window, 3)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def longest_active_span(mask: np.ndarray) -> tuple[int, int]:
    best_start = 0
    best_end = -1
    start = None

    for idx, active in enumerate(mask.tolist()):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            end = idx - 1
            if end - start > best_end - best_start:
                best_start, best_end = start, end
            start = None

    if start is not None:
        end = len(mask) - 1
        if end - start > best_end - best_start:
            best_start, best_end = start, end

    return best_start, best_end


def load_bgr_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return image


def load_binary_mask(mask_path: Path) -> np.ndarray:
    image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {mask_path}")
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary


def iter_image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted([item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES])


def iter_mask_paths(path: Path, suffix: str = "*_holes_bw.png") -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob(suffix))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image)


def build_hole_response(image: np.ndarray, kernel_divisor: int = 80, min_kernel: int = 9) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv_v = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
    lab_l = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]

    h, w = gray.shape
    kernel_size = odd_at_least(min(h, w) // kernel_divisor, min_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    responses: list[np.ndarray] = []
    for channel in (gray, hsv_v, lab_l):
        blurred = cv2.GaussianBlur(channel, (3, 3), 0)
        blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
        responses.append(cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX))

    return np.maximum.reduce(responses).astype(np.uint8)


def derive_roi_path(hole_path: Path, roi_root: Path) -> Path:
    stem = hole_path.stem
    if stem.endswith("_holes_bw"):
        stem = stem[:-9]
    return roi_root / f"{stem}.jpg"
