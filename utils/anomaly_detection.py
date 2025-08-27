import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, List, Dict

ORGAN_LIST = ["pancreas", "spleen", "stomach", "gallbladder", "liver", "kidney_right", "kidney_left"]
SPP_LEVELS = [(1, 1), (2, 2), (3, 3), (4, 4)]

def load_feature_array(path: str) -> np.ndarray:
    t = torch.load(path, map_location="cpu")
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    arr = np.asarray(t, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] == 10:
        return arr.astype(np.float32)
    raise ValueError(f"Unexpected feature shape {arr.shape} in {path}")

def spp_features(t: np.ndarray, levels: List[Tuple[int, int]] = SPP_LEVELS) -> np.ndarray:
    C, H, W = t.shape
    feats = []
    for (gh, gw) in levels:
        hbounds = [H // gh * i for i in range(gh + 1)]
        wbounds = [W // gw * i for i in range(gw + 1)]
        for i in range(gh):
            for j in range(gw):
                hs, he = hbounds[i], hbounds[i + 1]
                ws, we = wbounds[j], wbounds[j + 1]
                patch = t[:, hs:he, ws:we]
                flat = patch.reshape(C, -1)
                m = flat.mean(axis=1)
                s = flat.std(axis=1)
                feats.append(m)
                feats.append(s)
    return np.concatenate(feats, axis=0).astype(np.float32)

def load_model_and_threshold(organ: str, model_dir: Path) -> Tuple[xgb.XGBClassifier, float]:
    model_path = model_dir / organ / f"xgb_model_{organ}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    results_path = model_dir / organ / f"results_{organ}.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found at {results_path}")
    with open(results_path, "r") as f:
        results = json.load(f)
    thr_star = results["test"]["thr_used"]
    return model, thr_star

def infer_organ(organ: str, features_dir: Path, model_dir: Path, output_dir: Path) -> Tuple[bool, List[str]]:
    organ_dir = features_dir / organ
    if not organ_dir.exists() or not any(organ_dir.glob("*.pt")):
        print(f"Warning: No features found for {organ}, skipping")
        return False, []

    feature_paths = sorted(organ_dir.glob("*.pt"))
    feature_vectors = []
    for path in tqdm(feature_paths, desc=f"Processing {organ} features"):
        feat = load_feature_array(path)
        vec = spp_features(feat, levels=SPP_LEVELS)
        feature_vectors.append(vec)

    X = np.vstack(feature_vectors)
    scaler = joblib.load(model_dir / organ / "scaler.pkl")
    X = scaler.transform(X)

    model, thr_star = load_model_and_threshold(organ, model_dir)
    proba = model.predict_proba(X)[:, 1]
    has_anomaly = any(p >= thr_star for p in proba)
    suspicious_slices = [str(path) for path, p in zip(feature_paths, proba) if p >= thr_star]

    results = {
        "organ": organ,
        "has_anomaly": has_anomaly,
        "probability_threshold": thr_star,
        "suspicious_slices": suspicious_slices,
        "all_probabilities": proba.tolist()
    }
    output_path = output_dir / f"inference_{organ}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[{organ}] Has anomaly: {has_anomaly}, Suspicious slices: {len(suspicious_slices)}")
    return has_anomaly, suspicious_slices

def detect_anomalies(nifti_path: str, organ: str, model_dir: Path) -> Dict:
    features_dir = Path(os.getenv("FEATURES_DIR", "features")) / os.path.basename(nifti_path).replace(".nii.gz", "")
    output_dir = Path(os.getenv("INFERENCE_DIR", "inference_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    organs_to_process = ORGAN_LIST if organ == "whole" else [organ]
    
    for org in organs_to_process:
        if org not in ORGAN_LIST:
            print(f"Warning: Invalid organ {org}, skipping")
            continue
        has_anomaly, suspicious_slices = infer_organ(org, features_dir, model_dir, output_dir)
        all_results[org] = {"has_anomaly": has_anomaly, "suspicious_slices": suspicious_slices}

    output_path = output_dir / "inference_summary.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    return all_results