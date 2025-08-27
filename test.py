import inspect

from sam2.build_sam import build_sam2_video_predictor_npz
from configs.app_config import AppConfig


predictor = build_sam2_video_predictor_npz(
    AppConfig.MEDSAM_CONFIG_PATH, AppConfig.MEDSAM_CHECKPOINT_PATH)

sig = inspect.signature(predictor.add_new_points_or_box)
print(f"Signature: {sig}")
