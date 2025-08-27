import os
import SimpleITK as sitk
import numpy as np
import cv2
from pathlib import Path

def extract_slices_from_nifti_mask(nifti_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load NIfTI
    mask_nii = sitk.ReadImage(str(nifti_path))
    mask_array = sitk.GetArrayFromImage(mask_nii)  # shape: [D, H, W]

    for idx, mask in enumerate(mask_array):
        mask_uint8 = (mask > 0).astype(np.uint8) * 255  # Convert to binary mask
        filename = output_dir / f"slice_{idx:03d}_pred.png"
        cv2.imwrite(str(filename), mask_uint8)

    print(f"[✓] Extracted {len(mask_array)} slices from {nifti_path}")
