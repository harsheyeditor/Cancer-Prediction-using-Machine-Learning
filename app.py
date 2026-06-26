import sys
import os
import gradio as gr

# Add src to path so we can import our predictor
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from predict import CancerPredictor

# Initialize the predictor
try:
    predictor = CancerPredictor(model_dir="models", data_dir="data")
    print("✅ Predictor initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing predictor: {e}")
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
    emoji = "🔴" if diagnosis == "Malignant" else "🟢"
    
    output_text = f"{emoji} **Diagnosis:** {diagnosis}\n**Confidence:** {confidence}%"
    
    return output_text

# Define the Gradio interface
# Using min/max/median from the dataset for the top 5 features
# concave points_worst: 0 to 0.29 (median 0.10)
# perimeter_worst: 50 to 251 (median 97.66)
# concave points_mean: 0 to 0.20 (median 0.03)
# radius_worst: 7.9 to 36 (median 14.97)
# area_worst: 185 to 4254 (median 686.5)

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
    title="🩺 Breast Cancer Diagnostic Tool",
    description="Enter the cellular measurements from the FNA biopsy. The model will predict whether the mass is Benign or Malignant based on the Random Forest classifier trained on the Wisconsin Breast Cancer Dataset.",
    theme="huggingface",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(share=False)
