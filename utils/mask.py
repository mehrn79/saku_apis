import base64
from pathlib import Path
from typing import Dict
import logging

def load_masks_as_base64(mask_root_dir: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
   
    output = {}
    for patient_dir in mask_root_dir.iterdir():
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        output[patient_id] = {}
        for organ_dir in patient_dir.iterdir():
            if not organ_dir.is_dir():
                continue
            organ_name = organ_dir.name
            output[patient_id][organ_name] = {}
            for mask_file in organ_dir.glob("*.png"):
                try:
                    with open(mask_file, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode("utf-8")
                        output[patient_id][organ_name][mask_file.name] = f"data:image/png;base64,{encoded}"
                except Exception as e:
                    logging.error(f"Error encoding {mask_file}: {e}")
    return output


def load_flat_masks_as_base64(mask_dir: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for mask_file in mask_dir.glob("*.png"):
        try:
            with open(mask_file, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            output[mask_file.name] = f"data:image/png;base64,{encoded}"
        except Exception as e:
            logging.error(f"Error encoding {mask_file}: {e}")
    return output
