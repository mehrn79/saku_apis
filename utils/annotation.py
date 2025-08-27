import subprocess
from pathlib import Path

from configs.app_config import AppConfig

BASE_DIR = Path(__file__).resolve().parents[1]


def run_medsam2_prediction(ct_path, slice_idx, session_path_str, patient_id, tool, annotation_path, box_data):
    output_path = Path(session_path_str) / "medsam2_outputs"
    output_path.mkdir(parents=True, exist_ok=True)
    output_mask_path = output_path / f"{patient_id}_pred_mask.nii.gz"

    cmd_base = [
        "python", str(BASE_DIR / "utils" / "med_sam.py"),
        "--ct_path", str(ct_path),
        "--key_slice_idx", str(slice_idx),
        "--save_path", str(output_mask_path),
        "--checkpoint", str(AppConfig.MEDSAM_CHECKPOINT_PATH),
        "--cfg", str(AppConfig.MEDSAM_CONFIG_PATH)
    ]

    if tool == "Bounding Box":
        if not box_data:
            return "❌ Bounding Box tool selected but no box data provided.", None, None
        try:
            x1 = int(round(box_data["xmin"]))
            y1 = int(round(box_data["ymin"]))
            x2 = int(round(box_data["xmax"]))
            y2 = int(round(box_data["ymax"]))
            box_str = f"{x1} {y1} {x2} {y2}"
            cmd = cmd_base + ["--box", *box_str.split()]
        except (KeyError, TypeError) as e:
            return f"❌ Failed to parse bounding box data: {e}", None, None

    elif tool == "Brush":
        if not annotation_path or not Path(annotation_path).exists():
            return "❌ Brush tool selected but no mask path was provided or found.", None, None
        cmd = cmd_base + ["--mask_path", str(annotation_path)]

    else:
        return f"❌ Unknown annotation tool: {tool}", None, None

    print("[INFO] Running command:", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True)
        print("[STDOUT]\n", result.stdout)
        if result.stderr:
            print("[STDERR]\n", result.stderr)

        return f"✅ MedSAM2 segmentation completed.", str(output_mask_path), patient_id

    except subprocess.CalledProcessError as e:
        print("[ERROR] Subprocess failed:", e)
        print("[STDERR]\n", e.stderr)
        return f"❌ MedSAM2 failed:\n{e.stderr}", None, None
    except Exception as ex:
        print("[ERROR] Unexpected failure:", ex)
        return f"❌ Unexpected error: {ex}", None, None
