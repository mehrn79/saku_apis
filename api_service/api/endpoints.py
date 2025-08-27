import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import os
import re
import torch
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from api_service.models.organ_segmentation import run_segmentation
from api_service.models.medsam2 import run_annotation_segmentation
from configs.app_config import AppConfig
from utils.nifti import create_png_masks_from_nifti
from utils.mask import load_flat_masks_as_base64, load_masks_as_base64
from utils.feature_extraction import extract_features
from utils.anomaly_detection import detect_anomalies
from utils.nifti_processing import process_masks, ORGAN_MAPPING  
from utils.generate_medical_report import generate_medical_report
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

router = APIRouter()

class AnomalyPayload(BaseModel):
    organ: str

@router.post(
    "/segment",
    summary="Segmentation Endpoint",
    response_class=JSONResponse
)
def segmentation_endpoint(file: UploadFile = File(...)):
    file_path = AppConfig.TEMP_UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    segmentation_output_dir, patient_id, session_path = run_segmentation(
        file_path)

    png_masks_dir = session_path / "organ_masks"
    png_masks_dir.mkdir(parents=True, exist_ok=True)

    create_png_masks_from_nifti(segmentation_output_dir, png_masks_dir)

    masks = load_masks_as_base64(Path(png_masks_dir))

    return JSONResponse(content={
        "patient_id": patient_id,
        "masks": masks
    })

@router.post(
    "/annotate-segment",
    summary="Annotation-based Segmentation with MedSAM2",
)
def annotation_segmentation_endpoint(
    file: UploadFile = File(...),
    slice_idx: int = Form(...),
    image: UploadFile = File(None),
    box: str = Form(None)
):
    tool = None
    box_data = None
    image_np = None

    if box:
        tool = "Bounding Box"
        try:
            box_data = json.loads(box)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail="Invalid JSON format in 'box'.")
    elif image:
        tool = "Brush"

    if not tool:
        raise HTTPException(
            status_code=400, detail="Either 'box' or 'image' (for brush mask) must be provided.")

    file_path = AppConfig.TEMP_UPLOAD_DIR / file.filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if image is not None:
        image_path = AppConfig.TEMP_UPLOAD_DIR / image.filename
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        brush_image = Image.open(image_path).convert("RGBA")
        image_np = np.array(brush_image)

    try:
        png_masks_dir, patient_id = run_annotation_segmentation(
            tool=tool,
            slice_idx=slice_idx,
            brush_np=image_np,
            box_data=box_data,
            file=file_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    masks = load_flat_masks_as_base64(Path(png_masks_dir))

    return {
        "patient_id": patient_id,
        "masks": masks
    }


class AnomalyPayload(BaseModel):
    organ: str  # رشته‌ای از ارگان‌ها جدا شده با کاما (مثل "liver,spleen")
    main_mask_dir: str
    patient_id: str

@router.post(
    "/anomaly_detection",
    summary="Anomaly Detection using Mask Directory and NIfTI",
    response_class=JSONResponse
)
async def anomaly_detection(file: UploadFile = File(...), payload: str = Form(...)):
    try:
        logger.info(f"Received request for anomaly detection")

        if not payload:
            raise HTTPException(status_code=400, detail="Payload is empty")

        try:
            payload_dict = json.loads(payload)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON format in payload: {str(e)}")

        organ_input = payload_dict.get("organ", "")
        main_mask_dir = payload_dict.get("main_mask_dir")
        patient_id = payload_dict.get("patient_id")

        # تبدیل رشته ارگان‌ها به لیست
        organs = [o.strip() for o in organ_input.split(",")] if organ_input else []
        valid_organs = list(ORGAN_MAPPING.keys()) + ["whole"]
        if not organs or any(o not in valid_organs for o in organs):
            raise HTTPException(status_code=400, detail=f"Invalid organ(s). Must be one or more of {valid_organs}, separated by commas")

        if not main_mask_dir:
            raise HTTPException(status_code=400, detail="main_mask_dir is required")
        if not Path(main_mask_dir).exists():
            raise HTTPException(status_code=400, detail=f"Main mask directory {main_mask_dir} does not exist")

        if not patient_id:
            raise HTTPException(status_code=400, detail="patient_id is required")

        # ذخیره موقت فایل NIfTI
        nifti_path = Path("temp") / file.filename
        nifti_path.parent.mkdir(parents=True, exist_ok=True)
        with nifti_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"NIfTI file saved temporarily at {nifti_path}")

        logger.info(f"Processing masks for organs {organs} in {main_mask_dir}")

        # مرحله 1: پردازش ماسک‌ها
        organ_batches = process_masks(str(nifti_path), main_mask_dir, ",".join(organs), patient_id)  # ارسال همه ارگان‌ها به صورت رشته

        # مرحله 2: Feature Extraction
        upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).to(os.getenv("DEVICE", "cuda"))
        model = upsampler.model
        featup = upsampler
        logger.info(f"Extracting features for selected organs")
        extract_features(main_mask_dir, model, featup)  # فقط یک بار اجرا می‌شه

        # مرحله 3: Anomaly Detection
        model_dir = Path(os.getenv("MODEL_DIR", "/path/to/xgboost_models"))
        results = {}
        if "whole" in organs:
            # پردازش یکپارچه برای همه ارگان‌ها
            logger.info(f"Detecting anomalies for all organs")
            raw_results = detect_anomalies(main_mask_dir, "whole", model_dir)
            for current_organ in ORGAN_MAPPING.keys():
                results[current_organ] = {
                    "has_anomaly": raw_results.get(current_organ, {}).get("has_anomaly", False),
                    "suspicious_slices": [Path(slice_path).stem.split('_')[0] for slice_path in raw_results.get(current_organ, {}).get("suspicious_slices", [])]
                }
        else:
            # پردازش برای ارگان‌های انتخابی
            for current_organ in organs:
                logger.info(f"Detecting anomalies for {current_organ}")
                raw_results = detect_anomalies(main_mask_dir, current_organ, model_dir)
                results[current_organ] = {
                    "has_anomaly": raw_results[current_organ]["has_anomaly"],
                    "suspicious_slices": [Path(slice_path).stem.split('_')[0] for slice_path in raw_results[current_organ]["suspicious_slices"]]
                }

        # تولید گزارش پزشکی فقط برای ارگان‌های انتخابی
        medical_report = generate_medical_report({k: v for k, v in results.items() if k in organs}, patient_id)

        # حذف فایل موقت
        nifti_path.unlink()

        # اضافه کردن گزارش به خروجی JSON
        final_results = {"results": results, "medical_report": medical_report}

        return JSONResponse(content=final_results)

    except Exception as e:
        logger.error(f"Unexpected error in anomaly_detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")