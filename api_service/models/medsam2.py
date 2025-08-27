from pathlib import Path
import uuid
import shutil
from PIL import Image

from configs.app_config import AppConfig
from utils.annotation import run_medsam2_prediction


def run_annotation_segmentation(tool: str, slice_idx: int, brush_np, box_data, file: Path):
    session_root = AppConfig.BASE_OUTPUT_DIR
    session_path = session_root / f"session_{uuid.uuid4().hex}"
    uploaded_files_dir = session_path / "uploaded_files"
    annotation_path_dir = session_path / "annotations"

    uploaded_files_dir.mkdir(parents=True, exist_ok=True)
    annotation_path_dir.mkdir(parents=True, exist_ok=True)

    uploaded_file_path = uploaded_files_dir / file.name
    shutil.copy(str(file), str(uploaded_file_path))

    patient_id = file.name.replace(".nii.gz", "").replace(".nii", "")

    brush_mask_path = None
    if tool == "Brush" and brush_np is not None:
        brush_mask_path = annotation_path_dir / \
            f"{patient_id}_slice_{slice_idx:03d}_mask.png"
        Image.fromarray(brush_np).save(brush_mask_path)

    seg_status_msg, output_nifti_path, _ = run_medsam2_prediction(
        ct_path=uploaded_file_path,
        session_path_str=str(session_path),
        patient_id=patient_id,
        slice_idx=slice_idx,
        tool=tool,
        annotation_path=brush_mask_path,
        box_data=box_data
    )

    if output_nifti_path:
        png_masks_dir = Path(output_nifti_path).parent / "png_masks"
        return str(png_masks_dir), patient_id
    else:
        raise RuntimeError(f"MedSAM2 segmentation failed: {seg_status_msg}")
