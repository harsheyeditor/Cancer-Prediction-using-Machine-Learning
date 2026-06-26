import sys
import os
import gradio as gr

from src import config

# Check if artifacts exist before starting
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

def predict_cancer(concave_points_worst, perimeter_worst, concave_points_mean, radius_worst, area_worst):
    """
    Gradio interface function.
    Collects inputs from the UI, passes them to the predictor, and formats the output.
    """
    
    # Bundle inputs into dictionary matching the feature names
    patient_data = {
        'concave points_worst': concave_points_worst,
        'perimeter_worst': perimeter_worst,
        'concave points_mean': concave_points_mean,
        'radius_worst': radius_worst,
        'area_worst': area_worst
    }
    
    try:
        # Get prediction
        result = predictor.predict(patient_data)
        
        # Format output text
        diagnosis = result['diagnosis']
        confidence = result['confidence_pct']
        icon = "🔴" if diagnosis == "Malignant" else "🟢"
        
        output_text = f"### {icon} Diagnosis: **{diagnosis}**\n**Confidence:** {confidence}%"
        return output_text
        
    except Exception as e:
        return f"**Error processing prediction:** {str(e)}"

# Define the Gradio interface
# Using min/max/median from the dataset for the top 5 features
demo = gr.Interface(
    fn=predict_cancer,
    inputs=[
        gr.Slider(0.0, 0.3, value=0.10, step=0.01, label="Concave Points (Worst)"),
        gr.Slider(50.0, 255.0, value=97.66, step=1.0, label="Perimeter (Worst)"),
        gr.Slider(0.0, 0.25, value=0.03, step=0.01, label="Concave Points (Mean)"),
        gr.Slider(7.0, 40.0, value=14.97, step=0.1, label="Radius (Worst)"),
        gr.Slider(100.0, 4500.0, value=686.5, step=10.0, label="Area (Worst)")
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
