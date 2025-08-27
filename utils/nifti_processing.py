import os
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
import torch
from typing import Dict, List
from dotenv import load_dotenv
import logging

load_dotenv()

# تنظیم لاگ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تنظیمات
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "segmented_crops"))
ORGAN_MAPPING = {
    "pancreas": 10, "spleen": 1, "kidney_right": 2, "kidney_left": 3,
    "gallbladder": 4, "liver": 5, "stomach": 6
}
PAD = 5
TARGET_SIZE = 256

def load_mask(mask_path: Path) -> np.ndarray:
    logger.info(f"Loading mask: {mask_path}")
    img = Image.open(mask_path).convert("L")
    mask = np.array(img) > 0
    mask = torch.tensor(mask).long()
    mask = torch.flip(mask, dims=[0])
    mask = torch.rot90(mask, k=-1, dims=(0, 1))
    return mask.numpy().astype(np.uint8)

def get_bounding_box(mask: np.ndarray) -> tuple | None:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def hu_to_uint8(slice_np: np.ndarray, hu_min: int = -200, hu_max: int = 250) -> np.ndarray:
    slice_np = np.clip(slice_np, hu_min, hu_max)
    norm = (slice_np - hu_min) / (hu_max - hu_min)
    return (norm * 255).astype(np.uint8)

def apply_clahe(image_uint8: np.ndarray) -> np.ndarray:
    image_normalized = image_uint8 / 255.0
    clahe_image = equalize_adapthist(
        image_normalized,
        clip_limit=2.0,
        kernel_size=8
    )
    return (clahe_image * 255).astype(np.uint8)

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    image_uint8 = hu_to_uint8(image)
    return apply_clahe(image_uint8)

def process_masks(nifti_path: str, main_mask_dir: str, organ: str, patient_id: str) -> Dict[str, List[Path]]:
    logger.info(f"\n=== Processing masks in {main_mask_dir} for {organ} ===")

    # بارگذاری داده NIfTI
    logger.info(f"Loading NIfTI file: {nifti_path}")
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata().astype(np.float32)
    logger.info(f"NIfTI data loaded with shape: {data.shape}")

    # ایجاد دایرکتوری خروجی
    output_base_dir = OUTPUT_DIR / patient_id
    output_base_dir.mkdir(parents=True, exist_ok=True)

    organ_batches = {}
    organs_to_process = ORGAN_MAPPING.keys() if organ == "whole" else [organ]

    for current_organ in organs_to_process:
        organ_dir = output_base_dir / current_organ
        organ_dir.mkdir(parents=True, exist_ok=True)

        # مسیر زیرپوشه
        mask_dir = Path(main_mask_dir) / f"MONAI_{current_organ}"
        logger.info(f"Loading masks from: {mask_dir}")
        if not mask_dir.exists():
            logger.warning(f"No mask directory found at {mask_dir}, skipping {current_organ}")
            continue

        mask_files = sorted(mask_dir.glob("*_OUT.png"))
        if not mask_files:
            logger.warning(f"No mask files found in {mask_dir}, skipping {current_organ}")
            continue

        batch = []
        for idx, mask_path in enumerate(mask_files):
            logger.info(f"Processing mask: {mask_path.name}")
            mask = load_mask(mask_path)
            bbox = get_bounding_box(mask)
            if not bbox:
                logger.warning(f"No valid bounding box for mask {mask_path.name}, skipping")
                continue

            rmin, rmax, cmin, cmax = bbox
            rmin = max(0, rmin - PAD)
            rmax = min(mask.shape[0], rmax + PAD)
            cmin = max(0, cmin - PAD)
            cmax = min(mask.shape[1], cmax + PAD)
            slice_data_idx = idx
            if slice_data_idx < 0 or slice_data_idx >= data.shape[2]:
                logger.warning(f"Adjusted slice_idx {slice_data_idx} out of range [0, {data.shape[2]-1}], skipping")
                continue

            # برش از اسلایس CT
            crop = data[rmin:rmax, cmin:cmax, slice_data_idx]
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                logger.warning(f"Empty crop for slice {slice_data_idx}, skipping")
                continue
            if np.any(np.isnan(crop)) or np.any(np.isinf(crop)):
                logger.warning(f"NaN/Inf in crop for slice {slice_data_idx}, skipping")
                continue

            # بهبود کنتراست
            crop_enhanced = enhance_contrast(crop)
            crop_img = Image.fromarray(crop_enhanced)
            crop_img = crop_img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
            crop_img = np.array(crop_img)

            # ذخیره برش
            mask_filename = mask_path.stem
            crop_filename = f"{mask_filename}_crop.png"
            crop_path = organ_dir / crop_filename
            plt.imsave(crop_path, crop_img, cmap="gray")
            batch.append(crop_path)
            logger.info(f"Saved crop: {crop_path}")

        organ_batches[current_organ] = batch
        logger.info(f"[{current_organ}] Created {len(batch)} crops, saved to {organ_dir}")

    # ذخیره batch‌ها
    batch_output_path = output_base_dir / "batches.json"
    with open(batch_output_path, "w") as f:
        json.dump({k: [str(p) for p in v] for k, v in organ_batches.items()}, f, indent=2)
    logger.info(f"Batch saved to {batch_output_path}")

    return organ_batches