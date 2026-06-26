def get_model_info_html():
    return """
    <div>
        <div class="specs-panel">
            <div class="spec-item">
                <span class="spec-label">Model</span>
                <span class="spec-value">Random Forest</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Dataset</span>
                <span class="spec-value">Wisconsin Breast Cancer</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Training Samples</span>
                <span class="spec-value">569</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Features</span>
                <span class="spec-value">30</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Purpose</span>
                <span class="spec-value">Educational</span>
            </div>
        </div>
        
        <div class="disclaimer-card">
            <div class="disclaimer-icon">⚠️</div>
            <div class="disclaimer-text">
                <h4>Important Disclaimer</h4>
                <p>Generated for educational purposes only. Not a substitute for professional medical advice.</p>
            </div>
        </div>
    </div>
    """
