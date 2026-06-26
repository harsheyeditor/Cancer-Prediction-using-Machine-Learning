"""
Gradio Web Interface for Cancer Prediction
==========================================

Provides a premium, interactive medical dashboard for users to input 
cell measurements and receive real-time malignancy predictions.
"""

import sys
import os
import time
import gradio as gr

from src import config
from ui.components import get_header_html, get_footer_html
from ui.stats_card import get_stats_html
from ui.prediction_card import get_prediction_html
from ui.model_card import get_model_info_html

# Load CSS
with open(os.path.join(os.path.dirname(__file__), "ui", "style.css"), "r") as f:
    custom_css = f.read()

# Ensure ML artifacts are present before starting the server.
pipeline_path = os.path.join(config.MODEL_DIR, config.PIPELINE_FILE)
medians_path = os.path.join(config.MODEL_DIR, config.MEDIANS_FILE)

if not os.path.exists(pipeline_path) or not os.path.exists(medians_path):
    print("==================================================")
    print("[ERROR] ML Artifacts are missing!")
    print(f"Could not find pipeline at: {pipeline_path}")
    print("Please run the training script first to generate the model and data:")
    print("    python -m src.train")
    print("==================================================")
    sys.exit(1)

try:
    from src.predict import CancerPredictor
    predictor = CancerPredictor(model_dir=config.MODEL_DIR)
    print("[PASS] Predictor initialized successfully.")
except Exception as e:
    print(f"[ERROR] Error initializing predictor: {e}")
    sys.exit(1)

def predict_cancer(
    concave_points_worst: float,
    perimeter_worst: float,
    concave_points_mean: float,
    radius_worst: float,
    area_worst: float
):
    """
    Process UI inputs, run inference, and format the results for the Blocks UI.
    """
    # Simulate loading state for premium feel
    time.sleep(0.5)
    
    patient_data = {
        'concave points_worst': concave_points_worst,
        'perimeter_worst': perimeter_worst,
        'concave points_mean': concave_points_mean,
        'radius_worst': radius_worst,
        'area_worst': area_worst
    }
    
    try:
        result = predictor.predict(patient_data)
        return get_prediction_html(
            diagnosis=result['diagnosis'],
            confidence=result['confidence_pct'],
            prob_mal=result['probabilities']['Malignant'],
            prob_ben=result['probabilities']['Benign']
        )
    except Exception as e:
        return get_prediction_html(error=str(e))

def reset_inputs():
    vals = [config.UI_SLIDER_BOUNDS[key]["default"] for key in ["concave points_worst", "perimeter_worst", "concave points_mean", "radius_worst", "area_worst"]]
    return vals + [get_prediction_html()]

with gr.Blocks(title="Breast Cancer Diagnostic Tool") as demo:
    
    gr.HTML(get_header_html())
    
    with gr.Row():
        # Sidebar Stats
        with gr.Column(scale=2, min_width=200):
            gr.HTML(get_stats_html())
            
        # Main Inputs
        with gr.Column(scale=4, min_width=300):
            with gr.Group(elem_classes="card"):
                gr.HTML("""
                <div class="card-header">
                    <span class="card-header-icon">👤</span>
                    <h3 class="card-title">Patient Measurements</h3>
                </div>
                <p class="card-desc">Adjust the sliders below based on the patient's FNA biopsy report.</p>
                """)
                
                inputs = []
                slider_info = [
                    ("concave points_worst", "Concave Points (Worst)"),
                    ("perimeter_worst", "Perimeter (Worst)"),
                    ("concave points_mean", "Concave Points (Mean)"),
                    ("radius_worst", "Radius (Worst)"),
                    ("area_worst", "Area (Worst)")
                ]
                
                for key, label in slider_info:
                    gr.HTML(f"""
                    <div class="slider-label-box">
                        <span class="text-label">{label}</span>
                    </div>
                    """)
                    s = gr.Slider(
                        minimum=config.UI_SLIDER_BOUNDS[key]["minimum"],
                        maximum=config.UI_SLIDER_BOUNDS[key]["maximum"],
                        value=config.UI_SLIDER_BOUNDS[key]["default"],
                        step=config.UI_SLIDER_BOUNDS[key]["step"],
                        show_label=False,
                        container=False
                    )
                    inputs.append(s)
                
                with gr.Row():
                    gr.HTML("<div class='spacer-md'></div>") # spacer
                with gr.Row():
                    reset_btn = gr.Button("↺ Reset Inputs", elem_classes="reset-btn")
                    predict_btn = gr.Button("🔍 Predict Diagnosis", elem_classes="predict-btn")
                
                gr.HTML("<div class='text-instruction'>Move sliders to update values. Click Predict to see the result.</div>")
                    
        # Prediction Panel
        with gr.Column(scale=5, min_width=400):
            with gr.Group(elem_classes="card"):
                gr.HTML("""
                <div class="card-header">
                    <span class="card-header-icon">📋</span>
                    <h3 class="card-title">Diagnostic Result</h3>
                </div>
                """)
                diagnosis_output = gr.HTML(get_prediction_html())
            
            with gr.Group(elem_classes="card"):
                gr.HTML("""
                <div class="card-header">
                    <span class="card-header-icon">🧑‍⚕️</span>
                    <h3 class="card-title">Model Information</h3>
                </div>
                """)
                gr.HTML(get_model_info_html())

    gr.HTML(get_footer_html())

    predict_btn.click(
        fn=predict_cancer,
        inputs=inputs,
        outputs=[diagnosis_output]
    )
    
    reset_btn.click(
        fn=reset_inputs,
        inputs=[],
        outputs=inputs + [diagnosis_output]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", css=custom_css)
