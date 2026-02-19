document.addEventListener('DOMContentLoaded', function () {
    const locationSelect = document.getElementById('user_location');
    for (let i = 1; i <= 100; i++) {
        const option = document.createElement('option');
        option.value = 'City' + i;
        option.textContent = 'City' + i;
        locationSelect.appendChild(option);
    }

    const today = new Date().toISOString().split('T')[0];
    document.getElementById('order_date').value = today;

    const form = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');
    const resultBody = document.getElementById('resultBody');

    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        const formData = {
            product_category: document.getElementById('product_category').value,
            product_price: document.getElementById('product_price').value,
            order_quantity: document.getElementById('order_quantity').value,
            order_date: document.getElementById('order_date').value,
            user_age: document.getElementById('user_age').value,
            user_gender: document.getElementById('user_gender').value,
            user_location: document.getElementById('user_location').value,
            payment_method: document.getElementById('payment_method').value,
            shipping_method: document.getElementById('shipping_method').value,
            discount_applied: document.getElementById('discount_applied').value
        };

        predictBtn.disabled = true;
        predictBtn.innerHTML = '<div class="spinner"></div> Analyzing...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (result.success) {
                showResult(result);
            } else {
                showError(result.error || 'Prediction failed');
            }
        } catch (err) {
            showError('Connection error. Is the server running?');
        } finally {
            predictBtn.disabled = false;
            predictBtn.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
                Predict Return Probability
            `;
        }
    });

    function showResult(result) {
        const prob = result.return_probability;
        const isHighRisk = prob >= 50;
        const color = isHighRisk ? '#ef4444' : '#10b981';
        const circumference = 2 * Math.PI * 65;
        const offset = circumference - (prob / 100) * circumference;

        resultBody.innerHTML = `
            <div class="result-display">
                <div class="probability-ring">
                    <svg width="160" height="160" viewBox="0 0 160 160">
                        <circle class="ring-bg" cx="80" cy="80" r="65"></circle>
                        <circle class="ring-fill" cx="80" cy="80" r="65"
                            stroke="${color}"
                            stroke-dasharray="${circumference}"
                            stroke-dashoffset="${circumference}"
                            id="ringFill">
                        </circle>
                    </svg>
                    <div class="probability-value">
                        <div class="percentage" style="color: ${color}" id="probText">0%</div>
                        <div class="label">Return Risk</div>
                    </div>
                </div>
                <div class="prediction-badge ${isHighRisk ? 'high-risk' : 'low-risk'}">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        ${isHighRisk
                ? '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line>'
                : '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline>'}
                    </svg>
                    ${result.prediction}
                </div>
                <div class="result-details">
                    <div class="detail-row">
                        <span class="detail-label">Probability</span>
                        <span class="detail-value">${prob}%</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Risk Level</span>
                        <span class="detail-value" style="color: ${color}">${isHighRisk ? 'High' : 'Low'}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Model</span>
                        <span class="detail-value">Logistic Regression</span>
                    </div>
                </div>
            </div>
        `;

        requestAnimationFrame(function () {
            const ring = document.getElementById('ringFill');
            const probText = document.getElementById('probText');
            if (ring) {
                ring.style.strokeDashoffset = offset;
            }
            animateCounter(probText, 0, prob, 800);
        });
    }

    function animateCounter(element, start, end, duration) {
        const startTime = performance.now();
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(start + (end - start) * eased * 100) / 100;
            element.textContent = current.toFixed(1) + '%';
            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                element.textContent = end + '%';
            }
        }
        requestAnimationFrame(update);
    }

    function showError(message) {
        resultBody.innerHTML = `
            <div class="error-display">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 8px;">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="15" y1="9" x2="9" y2="15"></line>
                    <line x1="9" y1="9" x2="15" y2="15"></line>
                </svg>
                <p>${message}</p>
            </div>
        `;
    }
});
