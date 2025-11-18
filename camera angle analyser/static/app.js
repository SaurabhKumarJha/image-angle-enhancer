const photoInput = document.getElementById('photoInput');
const uploadBox = document.getElementById('uploadBox');
const loadingSpinner = document.getElementById('loadingSpinner');
const originalImg = document.getElementById('originalImg');
const overlayImg = document.getElementById('overlayImg');
const suggestionsList = document.getElementById('suggestionsList');
const downloadOverlay = document.getElementById('downloadOverlay');
const retryBtn = document.getElementById('retryBtn');
const resultsSection = document.getElementById('resultsSection');
const analyzeBtn = document.getElementById('analyzeBtn');

let lastOverlayDataUrl = null;
let lastAnalysisData = null;

// ========== DRAG AND DROP ==========
uploadBox.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
  uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadBox.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    photoInput.files = files;
    analyzePhoto();
  }
});

uploadBox.addEventListener('click', () => {
  photoInput.click();
});

photoInput.addEventListener('change', () => {
  if (photoInput.files.length > 0) {
    uploadBox.classList.add('file-selected');
    analyzePhoto();
  }
});

// ========== ANALYZE BUTTON CLICK ==========
if (analyzeBtn) {
  analyzeBtn.addEventListener('click', analyzePhoto);
}

// ========== ANALYZE PHOTO ==========
async function analyzePhoto() {
  if (!photoInput.files || photoInput.files.length === 0) {
    alert('Please choose an image first');
    return;
  }

  const file = photoInput.files[0];
  originalImg.src = URL.createObjectURL(file);

  const fd = new FormData();
  fd.append('photo', file);

  loadingSpinner.style.display = 'block';
  resultsSection.style.display = 'none';

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: fd });
    const data = await res.json();
    
    if (data.error) {
      alert('Error: ' + data.error);
      loadingSpinner.style.display = 'none';
      return;
    }

    lastAnalysisData = data;
    displayResults(data);
    
  } catch (err) {
    alert('Analysis failed: ' + err);
    loadingSpinner.style.display = 'none';
  }
}

// ========== DISPLAY RESULTS ==========
function displayResults(data) {
  // Show overlay image
  overlayImg.src = data.overlay_image;
  lastOverlayDataUrl = data.overlay_image;

  // Update metric bars with smooth animation
  updateMetricBar('angleBar', Math.abs(data.tilt_degrees), 90, 'Angle');
  updateMetricBar('lightingBar', data.lighting_score * 100, 100, 'Lighting');
  updateMetricBar('focusBar', data.blur_score * 100, 100, 'Focus');
  updateMetricBar('vibrancyBar', data.saturation_score * 100, 100, 'Vibrancy');
  updateMetricBar('compositionBar', data.composition_score * 100, 100, 'Composition');
  
  // Eye focus - only show for portraits
  if (data.scene_type === 'portrait') {
    updateMetricBar('eyeBar', data.eye_focus_score * 100, 100, 'Eye Focus');
  } else {
    const eyeValue = document.getElementById('eyeValue');
    if (eyeValue) {
      eyeValue.textContent = 'N/A (Not a portrait)';
    }
  }

  // Display suggestions
  displaySuggestions(data.suggestions);

  // Show results
  loadingSpinner.style.display = 'none';
  resultsSection.style.display = 'block';
  resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function updateMetricBar(barId, value, max, label) {
  const bar = document.getElementById(barId);
  const valueId = barId.replace('Bar', 'Value');
  const percentage = Math.min((value / max) * 100, 100);
  
  if (bar) {
    // Animate bar fill
    bar.style.width = '0%';
    setTimeout(() => {
      bar.style.transition = 'width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)';
      bar.style.width = percentage + '%';
    }, 100);
  }
  
  const valueEl = document.getElementById(valueId);
  if (valueEl) {
    const displayValue = label === 'Angle' ? Math.abs(value).toFixed(1) + '°' : Math.round(percentage) + '%';
    valueEl.textContent = displayValue;
  }
}

function displaySuggestions(suggestions) {
  suggestionsList.innerHTML = '';
  
  suggestions.forEach((s) => {
    const li = document.createElement('li');
    li.className = 'suggestion-item';
    
    const scoreHtml = s.score !== undefined ? 
      `<div class="suggestion-score">Score: ${Math.round(s.score * 100)}%</div>` : '';
    
    const categoryBadge = s.category ? 
      `<span class="suggestion-category">${s.category}</span>` : '';
    
    li.innerHTML = `
      <div class="suggestion-header">
        <span class="suggestion-icon">${s.icon || '💡'}</span>
        ${categoryBadge}
      </div>
      <p class="suggestion-message">${s.message}</p>
      ${scoreHtml}
    `;
    
    li.style.animation = `slideInLeft 0.5s ease-out forwards`;
    li.style.opacity = '0';
    
    suggestionsList.appendChild(li);
  });
}

// ========== RETRY / UPLOAD ANOTHER ==========
retryBtn.addEventListener('click', () => {
  photoInput.value = '';
  uploadBox.classList.remove('file-selected');
  resultsSection.style.display = 'none';
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// ========== INITIALIZE ==========
console.log('✅ Smart Photo Analyzer loaded and ready');
