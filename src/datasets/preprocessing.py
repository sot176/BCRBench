"""
DICOM mammogram preprocessing pipeline.

Walks a directory of raw DICOM files, and for each one:
  1. Reads the DICOM and applies VOI/window-level normalization, converting
     to 8-bit for contour detection.
  2. Segments out the breast tissue (largest contour) from the background.
  3. Resizes the breast crop onto a fixed canvas, aligned left/right by
     laterality (read from the DICOM itself).
  4. Rescales to full 16-bit range and saves the result under the *same*
     filename it had, in the output directory.

Usage:
    python preprocess_dicom.py

Or import and call directly:
    from preprocess_dicom import preprocess_images
    preprocess_images("raw_dicoms/", "processed_images/")
"""

import os
from typing import Optional

import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut as apply_windowing


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resize_with_alignment(
    image: np.ndarray,
    target_height: int,
    target_width: int,
    laterality: str,
) -> np.ndarray:
    """
    Resize `image` to fit within (target_height, target_width) while preserving
    aspect ratio, then paste it onto a blank canvas of that exact size, aligned
    to the left or right edge depending on breast laterality.

    Right-laterality images are aligned to the right edge of the canvas;
    everything else (including unknown laterality) defaults to left alignment.
    """
    h, w = image.shape
    scale = min(target_height / h, target_width / w)

    resized = cv2.resize(
        image,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((target_height, target_width), dtype=np.uint8)

    if laterality == "R":
        canvas[:resized.shape[0], -resized.shape[1]:] = resized
    else:
        canvas[:resized.shape[0], :resized.shape[1]] = resized

    return canvas


def segment_breast(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Isolate the breast tissue from the background using the largest external
    contour in a binary threshold of `image`. Returns the masked image, or
    None if no contour was found.
    """
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    binary_image = binary_image.astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return cv2.bitwise_and(image, mask)


def load_and_normalize(dicom_img: pydicom.Dataset) -> Optional[np.ndarray]:
    """
    Apply VOI windowing to a DICOM's raw pixel array and rescale to 8-bit,
    inverting MONOCHROME1 images so tissue is bright on a dark background.
    Returns None if the resulting image is flat (no contrast).
    """
    img_array = dicom_img.pixel_array.astype(float)

    if dicom_img.PhotometricInterpretation == "MONOCHROME1":
        img_array = np.amax(img_array) - img_array

    img = apply_windowing(img_array, dicom_img)

    if img.max() == img.min():
        return None

    img = (img - img.min()) / (img.max() - img.min()) * 255

    return img.astype(np.uint8)


def preprocess_one(
    dicom_path: str,
    target_height: int = 2048,
    target_width: int = 1664,
) -> Optional[np.ndarray]:
    """
    Run the full preprocessing pipeline on a single DICOM file and return the
    final 16-bit image, or None if any step fails/produces an empty result.
    """
    dicom_img = pydicom.dcmread(dicom_path)
    laterality = str(getattr(dicom_img, "ImageLaterality", "U")).strip() or "U"

    img = load_and_normalize(dicom_img)
    if img is None:
        print(f"Flat image in {os.path.basename(dicom_path)}, skipping.")
        return None

    breast_only = segment_breast(img)
    if breast_only is None:
        print(f"No contours found in {os.path.basename(dicom_path)}, skipping.")
        return None

    resized_image = resize_with_alignment(
        breast_only,
        target_height=target_height,
        target_width=target_width,
        laterality=laterality,
    )

    if resized_image.max() == 0:
        print(f"Empty resized image in {os.path.basename(dicom_path)}, skipping.")
        return None

    final_image = (resized_image.astype(np.float32) / resized_image.max()) * 65535

    return final_image.astype(np.uint16)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def preprocess_images(
    img_dir_in: str,
    img_dir_out: str,
    target_height: int = 2048,
    target_width: int = 1664,
) -> None:
    """
    Preprocess every .dcm file found under `img_dir_in` (recursively) and save
    each result to `img_dir_out` under its original filename (with a .png
    extension in place of .dcm).
    """
    os.makedirs(img_dir_out, exist_ok=True)

    for root, _, files in os.walk(img_dir_in):
        for dicom_filename in sorted(files):
            if not dicom_filename.lower().endswith(".dcm"):
                continue

            dicom_path = os.path.join(root, dicom_filename)
            out_name = os.path.splitext(dicom_filename)[0] + ".png"
            out_path = os.path.join(img_dir_out, out_name)

            if os.path.exists(out_path):
                continue

            try:
                final_image = preprocess_one(
                    dicom_path,
                    target_height,
                    target_width,
                )
            except Exception as e:
                print(f"Failed to preprocess {dicom_filename}: {e}")
                continue

            if final_image is None:
                continue

            cv2.imwrite(out_path, final_image)


