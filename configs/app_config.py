import logging
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class AppConfig:
    MONAI_CONFIG_PATH = Path(os.getenv("MONAI_CONFIG_PATH"))
    BASE_OUTPUT_DIR = Path(os.getenv("BASE_OUTPUT_DIR", "./output"))
    TEMP_UPLOAD_DIR = Path(os.getenv("TEMP_UPLOAD_DIR", "./uploads"))
    MEDSAM_CHECKPOINT_PATH = Path(os.getenv("MEDSAM_CHECKPOINT_PATH"))
    MEDSAM_CONFIG_PATH = Path(os.getenv("MEDSAM_CONFIG_PATH"))

    ALL_ORGANS = [
        "background", "spleen", "kidney_right", "kidney_left", "gallbladder",
        "liver", "stomach", "pancreas", "adrenal_gland_right",
        "adrenal_gland_left", "lung_upper_lobe_left", "lung_lower_lobe_left",
        "lung_upper_lobe_right", "lung_middle_lobe_right",
        "lung_lower_lobe_right", "esophagus", "trachea", "thyroid_gland",
        "small_bowel", "duodenum", "colon", "urinary_bladder", "prostate",
        "kidney_cyst_left", "kidney_cyst_right", "sacrum", "vertebrae_S1",
        "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2",
        "vertebrae_L1", "vertebrae_T12", "vertebrae_T11", "vertebrae_T10",
        "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6",
        "vertebrae_T5", "vertebrae_T4", "vertebrae_T3", "vertebrae_T2",
        "vertebrae_T1", "vertebrae_C7", "vertebrae_C6", "vertebrae_C5",
        "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1",
        "heart", "aorta", "pulmonary_artery", "pulmonary_vein",
        "brachiocephalic_trunk", "subclavian_artery_right",
        "subclavian_artery_left", "common_carotid_artery_right",
        "common_carotid_artery_left", "brachiocephalic_vein_left",
        "brachiocephalic_vein_right", "atrial_appendage_left",
        "superior_vena_cava", "inferior_vena_cava", "portal_vein_and_splenic_vein",
        "iliac_artery_left", "iliac_artery_right", "iliac_vein_left",
        "iliac_vein_right", "humerus_left", "humerus_right", "scapula_left",
        "scapula_right", "clavicle_left", "clavicle_right", "femur_left",
        "femur_right", "hip_left", "hip_right", "spinal_cord", "gluteus_maximus_left",
        "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right",
        "gluteus_minimus_left", "gluteus_minimus_right", "autochthon_left",
        "autochthon_right", "iliopsoas_left", "iliopsoas_right",
        "pectoral_muscle_major_left", "pectoral_muscle_major_right",
        "pectoral_muscle_minor_left", "pectoral_muscle_minor_right",
        "latissimus_dorsi_left", "latissimus_dorsi_right", "serratus_anterior_left",
        "serratus_anterior_right", "trapezius_left", "trapezius_right",
        "rectus_abdominis_left", "rectus_abdominis_right",
        "abdominal_oblique_external_left", "abdominal_oblique_external_right",
        "abdominal_oblique_internal_left", "abdominal_oblique_internal_right"
    ]
    TARGET_ORGANS = ["liver", "spleen", "kidney_right", "kidney_left",
                     "gallbladder", "stomach", "pancreas"]
    ORGAN_COLORS = {
        "liver": (0, 0, 255), "spleen": (0, 255, 0), "kidney_right": (255, 0, 0),
        "kidney_left": (255, 255, 0), "gallbladder": (255, 0, 255),
        "stomach": (0, 255, 255), "pancreas": (128, 0, 128)
    }

    IMPL_CLASS_UID_PREFIX = "1.2.826.0.1.3680043.9.7434."

    @classmethod
    def setup_directories(cls):
        for path in [cls.BASE_OUTPUT_DIR, cls.TEMP_UPLOAD_DIR]:
            path.mkdir(parents=True, exist_ok=True)
