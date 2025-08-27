import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T
from featup.util import norm, pca
from pathlib import Path
from typing import Dict

ORGAN_MAPPING = {
    "pancreas": 10, "spleen": 1, "kidney_right": 2, "kidney_left": 3,
    "gallbladder": 4, "liver": 5, "stomach": 6
}

def process_folder_with_featup(png_folder: Path, out_folder: Path, model, featup, save_rgb: bool = False, num_pca: int = 10) -> None:
    os.makedirs(out_folder, exist_ok=True)
    if save_rgb:
        os.makedirs(out_folder / "rgb", exist_ok=True)

    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        norm
    ])

    filenames = sorted([f for f in os.listdir(png_folder) if f.endswith(".png")])
    for fname in tqdm(filenames, desc=f"Processing {os.path.basename(png_folder)}"):
        fpath = os.path.join(png_folder, fname)
        img_gray = np.array(Image.open(fpath).convert("L"))
        if img_gray is None:
            continue
        img_rgb = np.stack([img_gray] * 3, axis=-1)
        img_pil = Image.fromarray(img_rgb)
        image_tensor = transform(img_pil).unsqueeze(0).to(next(model.parameters()).device)
        
        with torch.no_grad():
            hr_feat = featup(image_tensor).squeeze(0).contiguous()
            pca_out, _ = pca([hr_feat.unsqueeze(0)], num_pca)
            pca_tensor = pca_out[0]
        
        out_file = out_folder / fname.replace(".png", ".pt")
        torch.save(pca_tensor.cpu(), out_file)

def extract_features(nifti_path: str, model, featup) -> None:
    print(f"\n=== Extracting features for NIfTI file: {nifti_path} ===")
    patient_id = os.path.basename(nifti_path).replace(".nii.gz", "")
    input_base_dir = Path(os.getenv("SEGMENTED_CROPS_DIR", "segmented_crops")) / patient_id
    output_base_dir = Path(os.getenv("FEATURES_DIR", "features")) / patient_id

    for organ_name in ORGAN_MAPPING.keys():
        organ_folder = input_base_dir / organ_name
        if organ_folder.exists() and any(organ_folder.glob("*.png")):
            out_folder = output_base_dir / organ_name
            print(f"Processing folder: {organ_folder}")
            process_folder_with_featup(organ_folder, out_folder, model, featup, save_rgb=False, num_pca=10)
            print(f"[{organ_name}] Feature extraction completed, saved to {out_folder}")
        else:
            print(f"Warning: Folder {organ_folder} not found or empty, skipping")

if __name__ == "__main__":
    nifti_path = "/path/to/sample.nii.gz"
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).cuda()
    model = upsampler.model
    featup = upsampler
    extract_features(nifti_path, model, featup)