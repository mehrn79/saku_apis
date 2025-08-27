
# Medical image APIs

An interactive **image segmentation** demo built with **Gradio**, wired to a medical/biomedical segmentation backend (via the `MedSAM2` submodule) and an enhanced API layer for anomaly detection and medical reporting.

> **Note:** This README has been updated based on the latest developments in the repository, including the integration of an anomaly detection API and medical report generation. If any details differ from your actual code, please let me know, and I’ll update it immediately.

---

## ✨ Features

- 🔌 **Point-and-click / box prompts** for interactive segmentation (Gradio UI)
- 🧠 **Model backend via `MedSAM2`** submodule for segmentation
- 🧰 **Utilities & configs** for reproducible runs (`utils/`, `configs/`)
- 🌐 **Enhanced API service** (`api_service/`) for decoupled UI, anomaly detection, and medical reporting
- 📝 **Automated medical report generation** based on detected anomalies
- 🧪 **Quick tests** / examples in `test.py`
- 🎯 **Support for multiple organs** (e.g., pancreas, liver, spleen) or whole-body analysis



## 📦 Repository Structure


segmentation_gradio/
├─ MedSAM2/           # model submodule (pulled via git submodules) for segmentation
├─ api_service/       # REST API server for inference, anomaly detection, and reporting
│  ├─ main.py         # API entry point and orchestration
│  └─ utils/          # utility functions including medical report generation
├─ configs/           # model / app config files
├─ utils/             # helpers and common utilities (e.g., nifti_processing, medical_report_generator)
├─ app.py             # Gradio UI app (entry point)
├─ main.py            # alternate entry / orchestration script
├─ segmentation.py    # segmentation logic / pipeline wrapper
├─ test.py            # quick tests / example usage
├─ requirements.txt   # Python dependencies
├─ .env-sample        # sample environment variables
└─ .gitmodules        # submodule definitions
```

---

## 🚀 Quickstart

### 1) Clone with Submodules

```bash
git clone --recurse-submodules https://github.com/mehrn79/segmentation_gradio.git
cd segmentation_gradio
git submodule update --init --recursive
```

### 2) Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3) Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Configure Environment Variables

```bash
cp .env-sample .env
```
Common variables (adjust to your setup):

```
DEVICE=cuda            # or "cpu"
MODEL_NAME=medsam2     # model id / variant for segmentation
MODEL_DIR=/path/to/xgboost_models  # directory for anomaly detection models
API_URL=http://127.0.0.1:8000     # if using the api_service
# HF_TOKEN=...         # if the backend pulls weights from Hugging Face
```

### 5) Obtain Model Weights

- `MedSAM2` is included as a submodule. Follow its README to download or place the required segmentation weights.
- For anomaly detection, ensure XGBoost models are placed in `MODEL_DIR` as specified in `.env`.

### 6) Run the App

```bash
# Option A: launch the Gradio UI directly
python app.py

# Option B: use the main orchestrator (if that’s how you run it)
python main.py
```
By default, Gradio serves on `http://127.0.0.1:7860/` — the console will print the exact URL.

> If you’re using the API service, run it first (see below), then launch the UI.

---

## 🌐 API Service

The API service in `api_service/` has been enhanced to include anomaly detection and medical report generation, in addition to segmentation.

### Run
```bash
cd api_service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Environment Variables (Suggested)
```
DEVICE=cuda            # or cpu
MODEL_NAME=medsam2
WEIGHTS_PATH=/path/to/weights
MODEL_DIR=/path/to/xgboost_models  # for anomaly detection
API_PORT=8000
API_HOST=0.0.0.0
```

### Endpoints
- `GET /healthz` — Health check
  ```json
  {"status":"ok","device":"cuda","model":"medsam2"}
  ```
- `POST /anomaly_detection` — Run anomaly detection and generate medical report
  - **Input**:
    - `multipart/form-data` with:
      - `file`: NIfTI file (e.g., `/media/.../000007_03_01.nii.gz`)
      - `payload` (JSON):
        ```json
        {
          "organ": "liver,spleen"  # or "pancreas", "whole", comma-separated for multiple organs
          "main_mask_dir": "/media/.../MONAI/000007_03_01",
          "patient_id": "000007_03_01"
        }
        ```
  - **Output**:
    ```json
    {
      "results": {
        "liver": {
          "has_anomaly": true,
          "suspicious_slices": ["043", "049", "050"]
        },
        "spleen": {
          "has_anomaly": false,
          "suspicious_slices": []
        }
      },
      "medical_report": "Abdominal CT Scan Radiology Report\n\nDate of Examination: August 20, 2025\nDate of Report: August 20, 2025\nExamination Type: Multidetector CT Scan of the Abdomen...\n\nFindings:\n- **Liver:** Abnormalities noted with suspicious findings...\n- **Spleen:** Normal appearance with no abnormalities detected...\n\nImpression:\n1. Anomaly detected in the liver...\n2. No abnormalities identified in the spleen...\n\nNote: This report is based on automated anomaly detection from CT imaging. All findings should be correlated with clinical history and confirmed by a qualified physician."
    }
    ```

#### Call Examples
**cURL (File Upload):**
```bash
curl -X POST "http://127.0.0.1:8000/anomaly_detection" \
  -F "file=@/media/.../000007_03_01.nii.gz" \
  -F 'payload={"organ":"liver,spleen","main_mask_dir":"/media/.../MONAI/000007_03_01","patient_id":"000007_03_01"}'
```

**Python Client:**
```python
import requests, json
url = "http://127.0.0.1:8000/anomaly_detection"
files = {"file": open("/media/.../000007_03_01.nii.gz", "rb")}
payload = {"organ": "liver,spleen", "main_mask_dir": "/media/.../MONAI/000007_03_01", "patient_id": "000007_03_01"}
resp = requests.post(url, files=files, data={"payload": json.dumps(payload)})
print(resp.json())
```

### Detailed Explanation of the New API
The newly developed API in `api_service/main.py` is designed to detect anomalies in abdominal organs using pre-segmented masks and generate professional medical reports. It operates through the following phases:

1. **Input Reception and Mask Processing**:
   - Accepts a NIfTI file and mask directory as input.
   - Processes masks for specified organs (single or multiple, e.g., "liver,spleen", or "whole").
   - Utilizes `nibabel` for NIfTI loading and custom logic in `utils/nifti_processing.py` to handle rotations and cropping.

2. **Feature Extraction**:
   - Employs the FeatUp model (via `torch.hub.load`) to extract high-resolution features from mask slices.
   - Executes once for all organs to optimize performance, storing results as `.pt` files.

3. **Anomaly Detection**:
   - Leverages an XGBoost model to identify anomalies.
   - For "whole" mode, processes all organs in a single run; otherwise, targets only selected organs.
   - Outputs include `has_anomaly` (boolean) and `suspicious_slices` (list of affected slice numbers).

4. **Medical Report Generation**:
   - Generates a detailed report using a custom template in `utils/medical_report_generator.py`.
   - The template was crafted after analyzing numerous radiology reports from sources like radiologytemplates.com.au and xradiologist.com.
   - Includes sections: header (date, exam type), clinical info, technique, findings (per organ), impression (summary with recommendations), and a note for physician review.
   - Filters content to include only the organs specified in the `organ` input.

This API enhances the original segmentation functionality by adding diagnostic capabilities, making it a comprehensive tool for medical imaging analysis.

---

## 🖼️ Using the UI

1. Upload a medical image (e.g., CT scan NIfTI file).
2. (Optional) Provide prompts: click foreground points or draw a box for segmentation.
3. Hit **Segment** to generate a segmentation mask.
4. Use the API integration to run anomaly detection and view the generated medical report.

---

## ⚙️ Configuration

- **Configs:** YAML/JSON files in `configs/`, referenced by CLI args or `.env`.
- **Runtime:** Switch devices (`cpu`/`cuda`), tweak model variant, and set paths for weights and XGBoost models.

---

## ✅ Testing

```bash
python test.py
```

Add CI later with GitHub Actions to lint and run smoke tests.

---

## 🧪 Example: Headless Inference

```python
from segmentation import segment_image
mask = segment_image(
    image_path="/path/to/image.nii.gz",
    points=[(x1, y1), (x2, y2)],   # or box=(x0, y0, x1, y1)
    device="cuda",
    model_name="medsam2",
)
# For anomaly detection, use the API endpoint
import requests
resp = requests.post("http://127.0.0.1:8000/anomaly_detection", files={"file": open("/path/to/image.nii.gz", "rb")}, data={"payload": json.dumps({"organ": "whole", "main_mask_dir": "/path/to/masks", "patient_id": "test"})})
print(resp.json())
```

---

## 🧩 Troubleshooting

- **CUDA not found** → Set `DEVICE=cpu` or install a CUDA-enabled PyTorch.
- **Model weights missing** → Ensure `MedSAM2` weights and XGBoost models are in place.
- **Gradio not opening** → Check the terminal URL/port and firewall on `7860`.
- **API 404s** → Verify the API server is running and `API_URL` is set.
- **Invalid slices in report** → Check `main_mask_dir` for correct file naming (e.g., `043_OUT_crop.pt`).

---

## 🗺️ Roadmap

- [ ] Precise API docs with full schemas
- [ ] Demo notebooks (batch inference, evaluation)
- [ ] Dockerfile + Compose (UI + API)
- [ ] Basic unit tests and CI
- [ ] Example datasets and masks for quick trials
- [ ] Multilingual medical report support

---

## 🤝 Contributing

PRs welcome! Please open an issue with context (data type, model variant, reproduction steps) before large changes.

---

## 📜 License

Add a license file (e.g., MIT) at the project root if you intend others to use or extend the code.

---

## 🙏 Acknowledgments

- `MedSAM2` authors and maintainers
- The Gradio team for the UI toolkit
- xAI for inspiration in medical AI development
```