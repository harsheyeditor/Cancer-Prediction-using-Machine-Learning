def get_prediction_html(diagnosis=None, confidence=None, prob_mal=None, prob_ben=None, error=None):
    if error:
        return f"""
        <div class="result-container">
            <div class="result-hero-box malignant">
                <div class="result-icon">⚠️</div>
                <div class="result-text">
                    <h2 class="text-danger">ERROR</h2>
                    <p class="text-danger">{error}</p>
                </div>
            </div>
        </div>
        """
        
    if diagnosis is None:
        return """
        <div class="result-container card card-flat state-loading box-awaiting">
            <div class="icon-awaiting">📊</div>
            <div class="text-h1 text-lg">Awaiting Input</div>
            <div class="text-subtitle mt-sm">Adjust the measurements on the left and click Predict.</div>
        </div>
        """
        
    if diagnosis == "Malignant":
        top_class = "malignant"
        icon = """<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon-svg icon-svg-danger"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>"""
        title = "MALIGNANT"
        title_class = "text-danger"
        desc = "The model predicts the mass is likely malignant."
        risk_class = "risk-high"
        risk_text = "High Risk"
        prob = prob_mal
        prob_text = "Probability of Malignant"
        prob_other_text = f"Benign Probability: <span class='text-success'>{prob_ben/100:.3f} ({prob_ben:.1f}%)</span>"
        fill_color = "var(--danger)"
    else:
        top_class = ""
        icon = """<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon-svg icon-svg-success"><polyline points="20 6 9 17 4 12"></polyline></svg>"""
        title = "BENIGN"
        title_class = "text-success"
        desc = "The model predicts the mass is likely benign."
        risk_class = "risk-low"
        risk_text = "Low Risk"
        prob = prob_ben
        prob_text = "Probability of Benign"
        prob_other_text = f"Malignant Probability: <span class='text-danger'>{prob_mal/100:.3f} ({prob_mal:.1f}%)</span>"
        fill_color = "var(--success)"
        
    return f"""
    <div class="result-container">
        <div class="result-hero-box {top_class}">
            <div class="result-icon">{icon}</div>
            <div class="result-text">
                <h2 class="{title_class}">{title}</h2>
                <p>{desc}</p>
                <span class="risk-badge {risk_class}">{risk_text}</span>
            </div>
        </div>
        
        <div class="metrics-row">
            <div class="card card-flat flex-1">
                <div class="text-value-large text-accent mb-xs">{confidence}%</div>
                <div class="text-label mb-sm">Confidence</div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: {confidence}%; background: {fill_color};"></div>
                </div>
                <div class="text-subtitle" style="font-size: 0.8rem;">High confidence in prediction</div>
            </div>
            
            <div class="card card-flat flex-1">
                <div class="text-value-large mb-xs">{prob/100:.3f}</div>
                <div class="text-label mb-sm">Prediction Probability</div>
                <div class="text-subtitle mt-lg" style="font-size: 0.8rem;">{prob_text}</div>
            </div>
        </div>
        
        <div class="card card-flat" style="padding: var(--space-sm);">
            {prob_other_text}
        </div>
    </div>
    """
