def get_stats_html():
    return """
    <div>
        <div class="card stat-item">
            <div class="stat-header">
                <span>🗄</span> Dataset
            </div>
            <div class="text-value-large">569</div>
            <div class="text-subtitle text-sm">Samples<br>Wisconsin Breast Cancer Dataset</div>
        </div>
        <div class="card stat-item">
            <div class="stat-header">
                <span>📈</span> Features
            </div>
            <div class="text-value-large">30</div>
            <div class="text-subtitle text-sm">Total Features<br>5 Selected for Prediction</div>
        </div>
        <div class="card stat-item">
            <div class="stat-header">
                <span>🧠</span> Algorithm
            </div>
            <div class="text-value-large text-xl">Random<br>Forest</div>
            <div class="text-subtitle text-sm mt-sm">Ensemble Learning Classifier</div>
        </div>
        <div class="card stat-item">
            <div class="stat-header">
                <span>🎯</span> Model Accuracy
            </div>
            <div class="text-value-large">96%+</div>
            <div class="text-subtitle text-sm">On Test Set<br>(Run training for exact metrics)</div>
        </div>
    </div>
    """
