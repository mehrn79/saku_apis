from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api_service.api import endpoints
from configs.app_config import AppConfig
import torch
import os

app = FastAPI(
    title="Medical AI Segmentation API",
    description="segmentation and annotation apis",
    version="1.0.0"
)

upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).to(os.getenv("DEVICE", "cuda"))
model = upsampler.model
featup = upsampler

AppConfig.setup_directories()

app.include_router(endpoints.router)
