from configs.app_config import AppConfig
import numpy as np
import cv2
import logging
from pathlib import Path


def apply_ct_window(image: np.ndarray, level: int = 40, width: int = 400) -> np.ndarray:
    lower = level - (width / 2)
    upper = level + (width / 2)
    windowed = np.clip(image, lower, upper)
    windowed = (windowed - lower) / (upper - lower)
    return (np.clip(windowed, 0, 1) * 255).astype(np.uint8)


def create_overlay_image(ct_slice_path: str, selected_organs: list, patient_id: str, slice_idx: int, session_path_str: str) -> np.ndarray:
    ct_img = cv2.imread(ct_slice_path, cv2.IMREAD_GRAYSCALE)
    if ct_img is None:
        logging.warning(f"Could not read CT slice: {ct_slice_path}")
        return np.zeros((512, 512, 3), dtype=np.uint8)

    overlay = cv2.cvtColor(ct_img, cv2.COLOR_GRAY2BGR)

    if not patient_id or not session_path_str:
        return overlay

    session_path = Path(session_path_str)

    # ➤ MONAI organ masks
    organ_mask_dir = session_path / "organ_masks"
    for organ in selected_organs or []:
        mask_path = organ_mask_dir / patient_id / organ / f'slice_{slice_idx:03d}_OUT.png'
        if not mask_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        mask_bool = mask > 0
        color = AppConfig.ORGAN_COLORS.get(organ, (255, 255, 255))
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask_bool] = color

        overlay[mask_bool] = cv2.addWeighted(
            overlay[mask_bool], 0.5,
            colored_mask[mask_bool], 0.5, 0
        )

    # ➤ MedSAM2 predicted masks
    pred_mask_path = session_path / "medsam2_outputs" / "png_masks" / f"slice_{slice_idx:03d}_pred.png"
    if pred_mask_path.exists():
        pred_mask = cv2.imread(str(pred_mask_path), cv2.IMREAD_GRAYSCALE)
        if pred_mask is not None:
            pred_bool = pred_mask > 0
            color = (0, 0, 255)  # 🔵 MedSAM2 mask color: Blue
            medsam_mask = np.zeros_like(overlay)
            medsam_mask[pred_bool] = color
            overlay[pred_bool] = cv2.addWeighted(
                overlay[pred_bool], 0.5,
                medsam_mask[pred_bool], 0.5, 0
            )

    return overlay
