"""Preprocess mammography DICOM files into aligned 16-bit PNG images.

The pipeline is intentionally conservative and easy to audit:

1. read DICOM images listed by an exam-level metadata CSV,
2. apply DICOM modality/VOI windowing when available,
3. normalize to 8-bit for foreground segmentation,
4. keep the largest breast contour,
5. resize while preserving aspect ratio,
6. align right breasts to the right edge and left breasts to the left edge,
7. save as 16-bit PNG files with metadata-rich filenames.

This script is designed for research preprocessing workflows. Review DICOM
metadata and local governance requirements before publishing or sharing outputs.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image

LOGGER = logging.getLogger("mammography_preprocessing")
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for a mammography preprocessing run."""

    csv_file: Path
    dicom_root: Path
    output_dir: Path
    existing_output_dir: Path | None = None
    csv_separator: str = ";"
    exam_id_column: str = "inviv"
    patient_id_column: str = "pid"
    study_date_column: str = "scr_date"
    target_height: int = 2048
    target_width: int = 1664
    threshold_mode: str = "nonzero"
    overwrite: bool = False
    dry_run: bool = False


@dataclass
class RunSummary:
    """Counters collected during preprocessing."""

    processed: int = 0
    skipped_existing: int = 0
    skipped_missing_folder: int = 0
    skipped_read_error: int = 0
    skipped_empty_or_flat: int = 0
    skipped_no_contour: int = 0
    failed_processing: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "processed": self.processed,
            "skipped_existing": self.skipped_existing,
            "skipped_missing_folder": self.skipped_missing_folder,
            "skipped_read_error": self.skipped_read_error,
            "skipped_empty_or_flat": self.skipped_empty_or_flat,
            "skipped_no_contour": self.skipped_no_contour,
            "failed_processing": self.failed_processing,
        }


def sanitize_filename_part(value: object, fallback: str = "unknown") -> str:
    """Return a filesystem-safe filename token."""

    if value is None or pd.isna(value):
        value = fallback
    text = str(value).strip()
    if not text:
        text = fallback
    return SAFE_FILENAME_RE.sub("-", text)


def normalize_date(value: object) -> str:
    """Normalize a CSV date value to YYYY-MM-DD, or ``unknown`` if invalid."""

    study_date = pd.to_datetime(value, errors="coerce")
    if pd.isna(study_date):
        return "unknown"
    return str(study_date.strftime("%Y-%m-%d"))


def resize_with_alignment(
    image: np.ndarray,
    target_height: int,
    target_width: int,
    laterality: str,
) -> np.ndarray:
    """Resize a 2D image and align it left or right by breast laterality.

    Right breasts are right-aligned; all other values, including missing or
    unknown laterality, are left-aligned.
    """

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}.")

    height, width = image.shape
    if height <= 0 or width <= 0:
        raise ValueError("Cannot resize an empty image.")

    scale = min(target_height / height, target_width / width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image,
        dsize=(new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((target_height, target_width), dtype=image.dtype)
    if str(laterality).strip().upper() == "R":
        canvas[:new_height, target_width - new_width :] = resized
    else:
        canvas[:new_height, :new_width] = resized

    return canvas


def scale_to_uint8(image: np.ndarray) -> np.ndarray | None:
    """Scale an image to uint8, returning ``None`` for flat images."""

    image = np.asarray(image, dtype=np.float32)
    image_min = float(np.nanmin(image))
    image_max = float(np.nanmax(image))
    image_range = image_max - image_min

    if not np.isfinite(image_range) or image_range <= 0:
        return None

    normalized = (image - image_min) / image_range * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def scale_to_uint16(image: np.ndarray) -> np.ndarray | None:
    """Scale a non-empty image to the full uint16 intensity range."""

    image = np.asarray(image, dtype=np.float32)
    image_max = float(np.nanmax(image))
    if not np.isfinite(image_max) or image_max <= 0:
        return None

    normalized = np.maximum(image, 0) / image_max * 65535.0
    return np.clip(normalized, 0, 65535).astype(np.uint16)


def build_foreground_mask(image: np.ndarray, threshold_mode: str = "nonzero") -> np.ndarray:
    """Build a binary foreground mask for contour detection."""

    if threshold_mode == "nonzero":
        _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError("threshold_mode must be 'nonzero'.")
    return mask.astype(np.uint8)


def keep_largest_contour(image: np.ndarray, threshold_mode: str = "nonzero") -> np.ndarray | None:
    """Mask the image to its largest external contour."""

    binary_image = build_foreground_mask(image, threshold_mode=threshold_mode)
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


def apply_dicom_windowing(dicom_dataset: object) -> np.ndarray:
    """Return a windowed floating-point pixel array from a pydicom dataset."""

    try:
        from pydicom.pixels import apply_modality_lut, apply_voi_lut
    except ImportError as exc:
        raise RuntimeError(
            "pydicom is required for DICOM preprocessing. Install it with "
            "`pip install pydicom`."
        ) from exc

    image = dicom_dataset.pixel_array
    image = apply_modality_lut(image, dicom_dataset)
    image = apply_voi_lut(image, dicom_dataset)
    return np.asarray(image, dtype=np.float32)


def dicom_to_uint8(dicom_dataset: object) -> np.ndarray | None:
    """Convert a DICOM dataset to a display-normalized uint8 image."""

    image = apply_dicom_windowing(dicom_dataset)

    photometric = str(getattr(dicom_dataset, "PhotometricInterpretation", "")).upper()
    if photometric == "MONOCHROME1":
        image = np.nanmax(image) - image

    return scale_to_uint8(image)


def iter_dicom_files(folder: Path) -> Iterable[Path]:
    """Yield DICOM files in deterministic order."""

    return sorted(path for path in folder.iterdir() if path.suffix.lower() == ".dcm")


def load_existing_filenames(existing_output_dir: Path | None) -> set[str]:
    """Load filenames from a prior preprocessing output directory."""

    if existing_output_dir is None or not existing_output_dir.exists():
        return set()
    return {path.name for path in existing_output_dir.iterdir() if path.is_file()}


def build_output_filename(
    patient_id: object,
    exam_id: object,
    laterality: object,
    view: object,
    study_date: object,
    image_index: int,
) -> str:
    """Build a stable, metadata-rich PNG filename."""

    parts = [
        sanitize_filename_part(patient_id),
        sanitize_filename_part(exam_id),
        sanitize_filename_part(laterality, fallback="U"),
        sanitize_filename_part(view),
        normalize_date(study_date),
        str(image_index),
    ]
    return "_".join(parts) + ".png"


def preprocess_dicom_file(
    dicom_path: Path,
    laterality: str,
    target_height: int,
    target_width: int,
    threshold_mode: str,
) -> tuple[np.ndarray | None, str | None]:
    """Preprocess one DICOM file into a uint16 PNG-ready array."""

    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError(
            "pydicom is required for DICOM preprocessing. Install it with "
            "`pip install pydicom`."
        ) from exc

    dicom_dataset = pydicom.dcmread(dicom_path)
    image = dicom_to_uint8(dicom_dataset)
    if image is None:
        return None, "flat"

    breast_only = keep_largest_contour(image, threshold_mode=threshold_mode)
    if breast_only is None:
        return None, "no_contour"

    resized = resize_with_alignment(
        breast_only,
        target_height=target_height,
        target_width=target_width,
        laterality=laterality,
    )
    final_image = scale_to_uint16(resized)
    if final_image is None:
        return None, "empty"
    return final_image, None


def preprocess_images(config: PreprocessConfig) -> RunSummary:
    """Preprocess all DICOM images referenced by ``config.csv_file``."""

    summary = RunSummary()
    dataframe = pd.read_csv(config.csv_file, sep=config.csv_separator)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    existing_filenames = load_existing_filenames(config.existing_output_dir)

    required_columns = {config.exam_id_column}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {sorted(missing_columns)}")

    for _, row in dataframe.iterrows():
        exam_id = sanitize_filename_part(row[config.exam_id_column])
        patient_id = row.get(config.patient_id_column, "unknown")
        study_date = row.get(config.study_date_column, "unknown")
        folder_path = config.dicom_root / exam_id

        if not folder_path.exists():
            LOGGER.warning("Folder not found for exam %s: %s", exam_id, folder_path)
            summary.skipped_missing_folder += 1
            continue

        image_counter: defaultdict[tuple[str, str, str], int] = defaultdict(int)

        for dicom_path in iter_dicom_files(folder_path):
            try:
                import pydicom

                dicom_dataset = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                laterality = str(getattr(dicom_dataset, "ImageLaterality", "U")).strip() or "U"
                view = str(getattr(dicom_dataset, "ViewPosition", "Unknown")).strip() or "Unknown"
                study_date_str = normalize_date(study_date)
                key = (laterality, view, study_date_str)
                image_counter[key] += 1
                image_index = image_counter[key]
                filename = build_output_filename(
                    patient_id=patient_id,
                    exam_id=exam_id,
                    laterality=laterality,
                    view=view,
                    study_date=study_date,
                    image_index=image_index,
                )

                if not config.overwrite and (
                    filename in existing_filenames or (config.output_dir / filename).exists()
                ):
                    LOGGER.info("Already exists, skipping: %s", filename)
                    summary.skipped_existing += 1
                    continue

                if config.dry_run:
                    LOGGER.info("Would process %s -> %s", dicom_path.name, filename)
                    continue

                final_image, skip_reason = preprocess_dicom_file(
                    dicom_path=dicom_path,
                    laterality=laterality,
                    target_height=config.target_height,
                    target_width=config.target_width,
                    threshold_mode=config.threshold_mode,
                )
                if final_image is None:
                    LOGGER.warning("Skipped %s image: %s", skip_reason, dicom_path)
                    if skip_reason == "no_contour":
                        summary.skipped_no_contour += 1
                    else:
                        summary.skipped_empty_or_flat += 1
                    continue

                output_path = config.output_dir / filename
                Image.fromarray(final_image).save(output_path)
                LOGGER.info("Processed %s -> %s", dicom_path.name, output_path)
                summary.processed += 1

            except RuntimeError:
                raise
            except Exception as exc:
                LOGGER.exception("Error processing %s: %s", dicom_path, exc)
                summary.failed_processing += 1

    return summary


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(description="Preprocess mammography DICOM folders into aligned 16-bit PNGs.",)
    parser.add_argument("--csv-file", required=True, type=Path, help="Exam metadata CSV.")
    parser.add_argument("--dicom-root",required=True,type=Path, help="Directory containing one DICOM subfolder per exam/inviv id.",)
    parser.add_argument("--output-dir", required=True, type=Path, help="PNG output directory.")
    parser.add_argument("--existing-output-dir",type=Path,default=None,help="Optional prior output directory used only for duplicate filename skipping.",)
    parser.add_argument("--csv-separator", default=";", help="CSV separator. Default: ';'.")
    parser.add_argument("--exam-id-column", default="inviv", help="CSV exam id column.")
    parser.add_argument("--patient-id-column", default="pid", help="CSV patient id column.")
    parser.add_argument("--study-date-column", default="scr_date", help="CSV study date column.")
    parser.add_argument("--target-height", type=positive_int, default=2048)
    parser.add_argument("--target-width", type=positive_int, default=1664)
    parser.add_argument("--threshold-mode",choices=("nonzero", "otsu"),default="nonzero",help="Foreground thresholding mode. Default matches the original script.",)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files.")
    parser.add_argument("--dry-run", action="store_true", help="Log planned outputs without saving.")
    parser.add_argument("--log-level",default="INFO",choices=("DEBUG", "INFO", "WARNING", "ERROR"),help="Logging verbosity.",)
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),format="%(asctime)s | %(levelname)s | %(message)s",)

    return PreprocessConfig(
        csv_file=args.csv_file,
        dicom_root=args.dicom_root,
        output_dir=args.output_dir,
        existing_output_dir=args.existing_output_dir,
        csv_separator=args.csv_separator,
        exam_id_column=args.exam_id_column,
        patient_id_column=args.patient_id_column,
        study_date_column=args.study_date_column,
        target_height=args.target_height,
        target_width=args.target_width,
        threshold_mode=args.threshold_mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def main() -> None:
    config = parse_args()
    summary = preprocess_images(config)
    LOGGER.info("Finished preprocessing: %s", summary.as_dict())


if __name__ == "__main__":
    main()
