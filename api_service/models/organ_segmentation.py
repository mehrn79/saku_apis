from configs.app_config import AppConfig
import uuid
from pathlib import Path

from segmentation import segment


def run_segmentation(file_path: Path):
    session_root = AppConfig.BASE_OUTPUT_DIR
    session_root.mkdir(parents=True, exist_ok=True)
    session_path = session_root / f"session_{uuid.uuid4().hex}"
    session_path.mkdir(parents=True, exist_ok=True)

    nifti_path = segment(file_path, session_path)
    segmentation_output_dir = nifti_path.parent.parent

    patient_id = file_path.name.replace(".nii.gz", "").replace(".nii", "")

    return segmentation_output_dir, patient_id, session_path
