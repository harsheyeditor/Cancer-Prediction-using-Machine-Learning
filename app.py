import sys
import os
import gradio as gr

# Robust paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Check if artifacts exist before starting
model_path = os.path.join(MODEL_DIR, "cancer_prediction_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    print("==================================================")
    print("[ERROR] ML Artifacts are missing!")
    print(f"Could not find model at: {model_path}")
    print("Please run the training pipeline first to generate the model and data:")
    print("    python src/train.py")
    print("==================================================")
    sys.exit(1)

# Add src to path so we can import our predictor
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from predict import CancerPredictor
    predictor = CancerPredictor(model_dir=MODEL_DIR, data_dir=DATA_DIR)
    print("[PASS] Predictor initialized successfully.")
except Exception as e:
    print(f"[ERROR] Error initializing predictor: {e}")
    sys.exit(1)

def predict_cancer(concave_points_worst, perimeter_worst, concave_points_mean, radius_worst, area_worst):
    """Gradio interface function."""
    
    # Bundle inputs into dictionary matching the feature names
    patient_data = {
        'concave points_worst': concave_points_worst,
        'perimeter_worst': perimeter_worst,
        'concave points_mean': concave_points_mean,
        'radius_worst': radius_worst,
        'area_worst': area_worst
    }
    
    # Get prediction
    result = predictor.predict(patient_data)
    
    # Format output text
    diagnosis = result['diagnosis']
    confidence = result['confidence_pct']
    icon = "[Malignant]" if diagnosis == "Malignant" else "[Benign]"
    
    output_text = f"{icon} **Diagnosis:** {diagnosis}\n**Confidence:** {confidence}%"
    
    return output_text

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
    description="Enter the cellular measurements from the FNA biopsy. The model will predict whether the mass is Benign or Malignant based on the Random Forest classifier trained on the Wisconsin Breast Cancer Dataset."
)

if __name__ == "__main__":
    demo.launch(share=False)
