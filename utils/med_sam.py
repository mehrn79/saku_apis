import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
from med_sam_masks import extract_slices_from_nifti_mask
import argparse
from sam2.build_sam import build_sam2_video_predictor_npz
from PIL import Image
import SimpleITK as sitk
import torch
import numpy as np
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]


def apply_flip_and_rotation(slice_img):
    slice_img = np.flipud(slice_img)
    slice_img = np.rot90(slice_img, k=-1)
    return slice_img


def preprocess_volume(volume, image_size):
    d, h, w = volume.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    for i in range(d):
        transformed = apply_flip_and_rotation(volume[i])
        img_pil = Image.fromarray(transformed.astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)
        resized_array[i] = img_array
    return resized_array


def mask_to_box(mask):
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("❌ Mask is empty. Cannot extract bounding box.")
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    return np.array([x_min, y_min, x_max, y_max])


def save_box_on_image(image_3ch, box, save_path="debug_box_input.png"):
    img = image_3ch.transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    x1, y1, x2, y2 = box
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.title("Input image with bounding box")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ct_path', required=True)
    parser.add_argument('--save_path', default='output_mask.nii.gz')
    parser.add_argument('--key_slice_idx', type=int, required=True)
    parser.add_argument('--box', nargs=4, type=int)
    parser.add_argument('--mask_path')
    parser.add_argument(
        '--checkpoint', default=f'{BASE_DIR}/MedSAM2/checkpoints/MedSAM2_latest.pt')
    parser.add_argument('--cfg', default='configs/sam2.1_hiera_t512.yaml')
    args = parser.parse_args()

    nii_image = sitk.ReadImage(args.ct_path)
    volume = sitk.GetArrayFromImage(nii_image)
    volume = np.clip(volume, -1024, 1024)
    volume = ((volume - np.min(volume)) / (np.max(volume) -
              np.min(volume)) * 255).astype(np.uint8)

    if args.box:
        box = np.array(args.box)
    elif args.mask_path:
        if args.mask_path.endswith('.npy'):
            mask = np.load(args.mask_path)
        else:
            mask = np.array(Image.open(args.mask_path).convert('L')) > 0
        mask = mask.astype(np.uint8)
        if mask.shape != (512, 512):
            mask = cv2.resize(mask, (512, 512),
                              interpolation=cv2.INTER_NEAREST)
        box = mask_to_box(mask)
    else:
        raise ValueError("You must provide either --box or --mask_path.")

    resized_volume = preprocess_volume(volume, 512)
    resized_volume = resized_volume / 255.0
    img_tensor = torch.from_numpy(resized_volume).float()

    img_mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    img_std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    img_tensor = (img_tensor - img_mean) / img_std

    save_box_on_image(img_tensor[args.key_slice_idx].cpu().numpy(), box)

    predictor = build_sam2_video_predictor_npz(args.cfg, args.checkpoint)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(img_tensor, 512, 512)

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=args.key_slice_idx,
            obj_id=1,
            box=box
        )

        seg_volume = np.zeros(volume.shape, dtype=np.uint8)

        for i, _, mask_logits in predictor.propagate_in_video(inference_state):
            seg_volume[i] = (mask_logits[0] > 0.0).cpu().numpy()[0]

        for i, _, mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            seg_volume[i] = (mask_logits[0] > 0.0).cpu().numpy()[0]

    seg_image = sitk.GetImageFromArray(seg_volume)
    seg_image.CopyInformation(nii_image)
    sitk.WriteImage(seg_image, args.save_path)

    output_mask_path = Path(args.save_path)
    png_output_dir = output_mask_path.parent / "png_masks"
    png_output_dir.mkdir(parents=True, exist_ok=True)

    extract_slices_from_nifti_mask(str(output_mask_path), str(png_output_dir))

    print(f"[✓] Extracted PNG slices to: {png_output_dir}")


if __name__ == "__main__":
    main()
