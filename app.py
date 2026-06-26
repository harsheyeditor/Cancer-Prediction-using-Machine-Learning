"""
Gradio Web Interface for Cancer Prediction
==========================================

Provides an interactive UI for users to input cell measurements 
and receive real-time malignancy predictions.
"""

import sys
import os
import gradio as gr

from src import config

# Ensure ML artifacts are present before starting the server.
# This prevents the app from crashing during inference if the model hasn't been trained yet.
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
) -> str:
    """
    Process UI inputs, run inference, and format the results.

    Args:
        concave_points_worst: Feature input from UI
        perimeter_worst: Feature input from UI
        concave_points_mean: Feature input from UI
        radius_worst: Feature input from UI
        area_worst: Feature input from UI

    Returns:
        A Markdown-formatted string with the diagnosis and confidence level.
    """
    
    # Map UI inputs to the feature names expected by the trained pipeline
    patient_data = {
        'concave points_worst': concave_points_worst,
        'perimeter_worst': perimeter_worst,
        'concave points_mean': concave_points_mean,
        'radius_worst': radius_worst,
        'area_worst': area_worst
    }
    
    try:
        # Use the trained prediction pipeline to generate the result
        result = predictor.predict(patient_data)
        
        # Format the output into a readable Markdown block for the Gradio frontend
        diagnosis = result['diagnosis']
        confidence = result['confidence_pct']
        icon = "🔴" if diagnosis == "Malignant" else "🟢"
        
        output_text = f"### {icon} Diagnosis: **{diagnosis}**\n**Confidence:** {confidence}%"
        return output_text
        
    except Exception as e:
        return f"**Error processing prediction:** {str(e)}"

# Define the Gradio interface
# Using UI bounds from config.py to avoid magic numbers and keep the interface easily adjustable.
demo = gr.Interface(
    fn=predict_cancer,
    inputs=[
        gr.Slider(
            config.UI_SLIDER_BOUNDS["concave points_worst"]["minimum"], 
            config.UI_SLIDER_BOUNDS["concave points_worst"]["maximum"], 
            value=config.UI_SLIDER_BOUNDS["concave points_worst"]["default"], 
            step=config.UI_SLIDER_BOUNDS["concave points_worst"]["step"], 
            label="Concave Points (Worst)"
        ),
        gr.Slider(
            config.UI_SLIDER_BOUNDS["perimeter_worst"]["minimum"], 
            config.UI_SLIDER_BOUNDS["perimeter_worst"]["maximum"], 
            value=config.UI_SLIDER_BOUNDS["perimeter_worst"]["default"], 
            step=config.UI_SLIDER_BOUNDS["perimeter_worst"]["step"], 
            label="Perimeter (Worst)"
        ),
        gr.Slider(
            config.UI_SLIDER_BOUNDS["concave points_mean"]["minimum"], 
            config.UI_SLIDER_BOUNDS["concave points_mean"]["maximum"], 
            value=config.UI_SLIDER_BOUNDS["concave points_mean"]["default"], 
            step=config.UI_SLIDER_BOUNDS["concave points_mean"]["step"], 
            label="Concave Points (Mean)"
        ),
        gr.Slider(
            config.UI_SLIDER_BOUNDS["radius_worst"]["minimum"], 
            config.UI_SLIDER_BOUNDS["radius_worst"]["maximum"], 
            value=config.UI_SLIDER_BOUNDS["radius_worst"]["default"], 
            step=config.UI_SLIDER_BOUNDS["radius_worst"]["step"], 
            label="Radius (Worst)"
        ),
        gr.Slider(
            config.UI_SLIDER_BOUNDS["area_worst"]["minimum"], 
            config.UI_SLIDER_BOUNDS["area_worst"]["maximum"], 
            value=config.UI_SLIDER_BOUNDS["area_worst"]["default"], 
            step=config.UI_SLIDER_BOUNDS["area_worst"]["step"], 
            label="Area (Worst)"
        )
    ],
    outputs=gr.Markdown(),
    title="Breast Cancer Diagnostic Tool",
    description=(
        "Enter the cellular measurements from the FNA biopsy. "
        "The model will predict whether the mass is **Benign** or **Malignant** "
        "using a Random Forest classifier."
    ),
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch(share=False)
