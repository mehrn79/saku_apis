import torch
import gradio as gr
from gradio_image_annotation import image_annotator
import numpy as np
from datetime import datetime
import uuid
from pathlib import Path
import shutil
from PIL import Image

from configs.app_config import AppConfig
from utils.image import create_overlay_image
from segmentation_gradio.segmentation import segment
from utils.nifti import prepare_nifti_slices
from utils.annotation import handle_annotation_and_segmentation

DEFAULT_SLICE_SHAPE = (512, 512, 3)
white_placeholder = 255 * np.ones(DEFAULT_SLICE_SHAPE, dtype=np.uint8)


def handle_file_upload(file):
    if file is None:
        return None, None, None, None, None, None

    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    session_path = AppConfig.BASE_OUTPUT_DIR / session_id

    uploaded_files_dir = session_path / "uploaded_files"
    temp_slices_dir = session_path / "temp_slices"

    uploaded_files_dir.mkdir(parents=True, exist_ok=True)
    temp_slices_dir.mkdir(parents=True, exist_ok=True)

    uploaded_file_path = uploaded_files_dir / Path(file.name).name
    shutil.copy(str(Path(file.name)), str(uploaded_file_path))

    slice_files, num_slices = prepare_nifti_slices(uploaded_file_path, temp_slices_dir)

    if not slice_files:
        return None, None, None, None, None, None

    initial_slice_idx = num_slices // 2
    patient_id = uploaded_file_path.name.split('.')[0]

    overlay_image = create_overlay_image(
        slice_files[initial_slice_idx], [], patient_id, initial_slice_idx, str(session_path))

    brush_image = np.array(Image.open(slice_files[initial_slice_idx]))

    return (
        gr.update(visible=True, maximum=num_slices - 1, value=initial_slice_idx),
        slice_files,
        {
            "image": overlay_image,
            "boxes": []
        },
        patient_id,
        str(session_path),
        brush_image
    )


def update_slice_view(slice_idx, slice_files, selected_organs, patient_id, session_path_str: str):
    if not slice_files or patient_id is None or not session_path_str:
        return None, None

    overlay_image = create_overlay_image(
        slice_files[slice_idx], selected_organs, patient_id, slice_idx, session_path_str
    )

    annotator_box_data = {
        "image": overlay_image, 
        "boxes": []
    }

    if overlay_image.shape[2] == 3:

        alpha_channel = np.full(overlay_image.shape[:2], 255, dtype=np.uint8)
        overlay_image = np.dstack((overlay_image, alpha_channel))  # تبدیل به RGBA

    return annotator_box_data, overlay_image


def run_segmentation(selected_organs, file, session_path_str: str):
    if not selected_organs:
        return "⚠️ Please select at least one organ.", gr.update(), gr.update()
    if file is None:
        return "⚠️ Please upload a NIfTI file first.", gr.update(), gr.update()

    session_path = Path(session_path_str)
    nii_path = session_path / "uploaded_files" / Path(file.name).name

    if not nii_path.exists():
        return f"Error: Uploaded file not found at {nii_path}", gr.update(), gr.update()

    status_message = segment(nii_path, selected_organs, session_path)
    patient_id = nii_path.name.split('.')[0]

    return status_message, selected_organs, patient_id


def switch_tool(tool):
    return (
        gr.update(visible=(tool == "Bounding Box")),
        gr.update(visible=(tool == "Brush"))
    )

def dummy_return(image_np):
    return image_np

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("## 🧠 CT Abdomen Viewer & Segmentation + Annotation")

    slice_files_state = gr.State()
    patient_id_state = gr.State()
    selected_organs_state = gr.State([])
    session_path_state = gr.State()

    with gr.Row():
        with gr.Column(scale=3):
            file_input = gr.File(label="Upload NIfTI file (.nii or .nii.gz)")
            with gr.Accordion("Segmentation", open=True):
                organs_checkbox = gr.CheckboxGroup(
                    choices=AppConfig.TARGET_ORGANS,
                    label="Select Target Organs"
                )
                segment_button = gr.Button("▶️ Run Segmentation", variant="primary")
            output_text = gr.Textbox(label="Status", interactive=False)

            with gr.Accordion("✍️ Annotation Tool", open=True):
                tool_selector = gr.Radio(choices=["Brush", "Bounding Box"], value="Bounding Box", label="Choose Tool")
                submit_btn = gr.Button("✅ Submit Annotation")

        with gr.Column(scale=7):
            annotator_box = image_annotator(
                {"image": white_placeholder, "boxes": []},
                label_list=["target"],
                label_colors=[(0, 255, 0)],
                boxes_alpha=0.4,
                disable_edit_boxes=True,  
                handle_size=7, 
                visible=True
            )

            annotator_brush = gr.ImageEditor(
                type="numpy",
                label="Draw with Brush",
                value=white_placeholder.copy(),
                image_mode="RGBA",
                height=600,
                brush=gr.Brush(),
                visible=False
            )

            hidden_output = gr.Image(visible=False)

            slice_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Slice", visible=False)

    file_input.upload(
        fn=handle_file_upload,
        inputs=[file_input],
        outputs=[slice_slider, slice_files_state, annotator_box, patient_id_state, session_path_state, annotator_brush]
    )

    segment_button.click(
        fn=run_segmentation,
        inputs=[organs_checkbox, file_input, session_path_state],
        outputs=[output_text, selected_organs_state, patient_id_state]
    ).then(
        fn=update_slice_view,
        inputs=[slice_slider, slice_files_state, selected_organs_state, patient_id_state, session_path_state],
        outputs=[annotator_box, annotator_brush]
    )

    inputs_for_updates = [slice_slider, slice_files_state, selected_organs_state, patient_id_state, session_path_state]

    slice_slider.change(fn=update_slice_view, inputs=inputs_for_updates, outputs=[annotator_box, annotator_brush])
    organs_checkbox.change(fn=update_slice_view, inputs=inputs_for_updates, outputs=[annotator_box, annotator_brush])

    tool_selector.change(fn=switch_tool, inputs=[tool_selector], outputs=[annotator_box, annotator_brush])
    annotator_brush.change(fn=lambda x: x, inputs=annotator_brush, outputs=hidden_output)

    submit_btn.click(
    fn=handle_annotation_and_segmentation,
    inputs=[
        tool_selector,          # tool: "Brush" یا "Bounding Box"
        annotator_box,          # annotator_box_data
        annotator_brush,        # brush_data
        slice_slider,           # slice_idx
        patient_id_state,
        session_path_state,
        file_input              # فایل CT
    ],
    outputs=[output_text]
    )


demo.launch()
