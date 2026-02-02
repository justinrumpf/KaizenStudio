// State
let currentImageId = null;
let originalSize = [0, 0];
let processedSize = [0, 0];
let paintMode = null; // 'keep' or 'remove'
let isPainting = false;
let paintPoints = { keep: [], remove: [], brushSize: 15 };

// Fullscreen state
let fullscreenMode = false;

// GoPro state
let goproAvailable = false;
let previewAutoRefresh = false;
let previewInterval = 5000;
let previewTimer = null;
let isPreviewRefreshing = false;
let captureInProgress = false;

// Stream state
let streamActive = false;
let streamStatusCheckInterval = null;

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const uploadSection = document.getElementById('upload-section');
const progressSection = document.getElementById('progress-section');
const resultsSection = document.getElementById('results-section');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const timingDetails = document.getElementById('timing-details');
const originalImage = document.getElementById('original-image');
const resultImage = document.getElementById('result-image');
const paintCanvas = document.getElementById('paint-canvas');
const sensitivitySlider = document.getElementById('sensitivity-slider');
const sensitivityValue = document.getElementById('sensitivity-value');
const brushSizeSlider = document.getElementById('brush-size');
const brushSizeValue = document.getElementById('brush-size-value');
const keepBtn = document.getElementById('keep-btn');
const removeBtn = document.getElementById('remove-btn');
const clearPaintBtn = document.getElementById('clear-paint-btn');
const applyBtn = document.getElementById('apply-btn');
const downloadTransparentBtn = document.getElementById('download-transparent-btn');
const downloadWhiteBtn = document.getElementById('download-white-btn');
const newImageBtn = document.getElementById('new-image-btn');
const brushCursor = document.getElementById('brush-cursor');

// Subject Zoom DOM Elements
const subjectZoomSlider = document.getElementById('subject-zoom-slider');
const subjectZoomValue = document.getElementById('subject-zoom-value');
const applyZoomBtn = document.getElementById('apply-zoom-btn');
const resetImageBtn = document.getElementById('reset-image-btn');
const fullscreenBtn = document.getElementById('fullscreen-btn');
const resultWrapper = document.getElementById('result-wrapper');

// Color Correction DOM Elements
const colorPresetButtons = document.querySelectorAll('.btn-preset');
const resetColorsBtn = document.getElementById('reset-colors-btn');

// Fullscreen Modal DOM Elements
const fullscreenModal = document.getElementById('fullscreen-modal');
const fsResultImage = document.getElementById('fs-result-image');
const fsPaintCanvas = document.getElementById('fs-paint-canvas');
const fsBrushCursor = document.getElementById('fs-brush-cursor');
const fsKeepBtn = document.getElementById('fs-keep-btn');
const fsRemoveBtn = document.getElementById('fs-remove-btn');
const fsClearBtn = document.getElementById('fs-clear-btn');
const fsBrushSize = document.getElementById('fs-brush-size');
const fsBrushSizeValue = document.getElementById('fs-brush-size-value');
const fsZoomSlider = document.getElementById('fs-zoom-slider');
const fsZoomValue = document.getElementById('fs-zoom-value');
const fsApplyBtn = document.getElementById('fs-apply-btn');
const fsCloseBtn = document.getElementById('fs-close-btn');
const fsPreviewBtn = document.getElementById('fs-preview-btn');
const fsRecenterBtn = document.getElementById('fs-recenter-btn');
const recenterBtn = document.getElementById('recenter-btn');
const fsImageWrapper = document.getElementById('fs-image-wrapper');
const fsCanvasContainer = document.getElementById('fs-canvas-container');

// GoPro DOM Elements
const goproPreviewSection = document.getElementById('gopro-preview-section');
const previewPlaceholder = document.getElementById('preview-placeholder');
const previewImage = document.getElementById('preview-image');
const goproCaptureBtn = document.getElementById('gopro-capture-btn');
const goproStatus = document.getElementById('gopro-status');

// Stream DOM Elements
const streamToggleBtn = document.getElementById('stream-toggle-btn');
const streamStatusIndicator = document.getElementById('stream-status-indicator');
const streamFeed = document.getElementById('stream-feed');
const cameraZoomSlider = document.getElementById('camera-zoom-slider');
const cameraZoomValue = document.getElementById('camera-zoom-value');
const whiteBalanceSelect = document.getElementById('white-balance-select');

// Initialize
document.addEventListener('DOMContentLoaded', init);

function init() {
    // File input handlers
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    // GoPro handlers
    goproCaptureBtn.addEventListener('click', captureWithGoPro);

    // Stream handlers
    streamToggleBtn.addEventListener('click', toggleStream);
    cameraZoomSlider.addEventListener('input', handleCameraZoomInput);
    cameraZoomSlider.addEventListener('change', setCameraZoom);
    whiteBalanceSelect.addEventListener('change', setWhiteBalance);

    // Check GoPro status on load
    checkGoproStatus();

    // Drag and drop handlers
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    dropZone.addEventListener('click', (e) => {
        // Don't open file browser if clicking on the button
        if (e.target !== browseBtn) {
            fileInput.click();
        }
    });

    // Slider handlers
    sensitivitySlider.addEventListener('input', (e) => {
        sensitivityValue.textContent = e.target.value;
    });

    brushSizeSlider.addEventListener('input', (e) => {
        brushSizeValue.textContent = e.target.value;
        paintPoints.brushSize = parseInt(e.target.value);
        updateBrushCursorAppearance();
    });

    // Paint mode buttons
    keepBtn.addEventListener('click', () => setPaintMode('keep'));
    removeBtn.addEventListener('click', () => setPaintMode('remove'));
    clearPaintBtn.addEventListener('click', clearPaint);

    // Canvas painting
    paintCanvas.addEventListener('mousedown', startPainting);
    paintCanvas.addEventListener('mousemove', paint);
    paintCanvas.addEventListener('mouseup', stopPainting);
    paintCanvas.addEventListener('mouseleave', handleCanvasLeave);
    paintCanvas.addEventListener('mouseenter', handleCanvasEnter);
    paintCanvas.addEventListener('mousemove', updateBrushCursor);

    // Touch support
    paintCanvas.addEventListener('touchstart', handleTouchStart);
    paintCanvas.addEventListener('touchmove', handleTouchMove);
    paintCanvas.addEventListener('touchend', stopPainting);

    // Action buttons
    applyBtn.addEventListener('click', applyAdjustments);
    downloadTransparentBtn.addEventListener('click', () => downloadImage('transparent'));
    downloadWhiteBtn.addEventListener('click', () => downloadImage('white'));
    newImageBtn.addEventListener('click', resetUI);

    // Subject Zoom handlers
    subjectZoomSlider.addEventListener('input', (e) => {
        subjectZoomValue.textContent = e.target.value + '%';
    });
    applyZoomBtn.addEventListener('click', applySubjectZoom);
    recenterBtn.addEventListener('click', recenterSubject);
    resetImageBtn.addEventListener('click', resetToOriginal);

    // Color Correction preset button handlers
    colorPresetButtons.forEach(btn => {
        btn.addEventListener('click', handleColorPreset);
    });
    resetColorsBtn.addEventListener('click', resetColors);

    // Fullscreen handlers
    fullscreenBtn.addEventListener('click', openFullscreen);
    fsCloseBtn.addEventListener('click', closeFullscreen);
    fsApplyBtn.addEventListener('click', applyFullscreenChanges);
    fsPreviewBtn.addEventListener('click', previewFullscreenChanges);
    fsRecenterBtn.addEventListener('click', recenterInFullscreen);
    fsKeepBtn.addEventListener('click', () => setFullscreenPaintMode('keep'));
    fsRemoveBtn.addEventListener('click', () => setFullscreenPaintMode('remove'));
    fsClearBtn.addEventListener('click', clearFullscreenPaint);
    fsBrushSize.addEventListener('input', updateFullscreenBrush);
    fsZoomSlider.addEventListener('input', handleFullscreenZoom);

    // Fullscreen canvas events
    fsPaintCanvas.addEventListener('mousedown', startFullscreenPaint);
    fsPaintCanvas.addEventListener('mousemove', fullscreenPaint);
    fsPaintCanvas.addEventListener('mouseup', stopFullscreenPaint);
    fsPaintCanvas.addEventListener('mouseleave', handleFullscreenCanvasLeave);
    fsPaintCanvas.addEventListener('mouseenter', handleFullscreenCanvasEnter);
    fsPaintCanvas.addEventListener('mousemove', updateFullscreenBrushCursor);

    // Keyboard shortcut for fullscreen
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && fullscreenMode) {
            closeFullscreen();
        }
    });
}

// Drag and Drop
function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        uploadImage(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        uploadImage(file);
    }
}

// Upload
async function uploadImage(file) {
    showProgress();
    updateProgress(10, 'Uploading image...');

    const formData = new FormData();
    formData.append('image', file);

    try {
        updateProgress(30, 'Processing image...');

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        updateProgress(70, 'Removing background...');

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.error || `Upload failed (${response.status})`);
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        updateProgress(90, 'Finalizing...');

        currentImageId = data.image_id;
        originalSize = data.original_size;
        processedSize = data.processed_size;

        // Display results
        displayResults(data);

        updateProgress(100, 'Complete!');

        setTimeout(() => {
            showResults();
        }, 500);

    } catch (error) {
        alert('Error processing image: ' + error.message);
        resetUI();
    }
}

function showProgress() {
    uploadSection.classList.add('hidden');
    progressSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
}

function updateProgress(percent, text) {
    progressFill.style.width = percent + '%';
    progressText.textContent = text;
}

function showResults() {
    uploadSection.classList.add('hidden');
    progressSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');
}

function displayResults(data) {
    // Display timing information
    const timings = data.timings;
    timingDetails.innerHTML = `
        <div class="timing-item">
            <span class="label">Read</span>
            <span class="value">${timings.read}ms</span>
        </div>
        <div class="timing-item">
            <span class="label">Resize</span>
            <span class="value">${timings.resize}ms</span>
        </div>
        <div class="timing-item">
            <span class="label">Remove BG</span>
            <span class="value">${timings.remove_bg}ms</span>
        </div>
        <div class="timing-item">
            <span class="label">Save</span>
            <span class="value">${timings.save}ms</span>
        </div>
        <div class="timing-item total">
            <span class="label">Total</span>
            <span class="value">${timings.total}ms</span>
        </div>
    `;

    // Add cache-busting timestamp
    const timestamp = Date.now();
    // Display images
    originalImage.src = `/image/original/${currentImageId}?t=${timestamp}`;
    resultImage.src = `/image/output-white/${currentImageId}?t=${timestamp}`;

    // Setup canvas after image loads
    resultImage.onload = () => {
        setupCanvas();
    };

    // Reset adjustments
    sensitivitySlider.value = 50;
    sensitivityValue.textContent = '50';
    clearPaint();

    // Reset subject zoom
    subjectZoomSlider.value = 100;
    subjectZoomValue.textContent = '100%';
}

// Canvas Painting
function setupCanvas() {
    const wrapper = resultWrapper;
    const wrapperRect = wrapper.getBoundingClientRect();
    const imgRect = resultImage.getBoundingClientRect();

    // Position canvas over the image
    const imgLeft = imgRect.left - wrapperRect.left;
    const imgTop = imgRect.top - wrapperRect.top;

    paintCanvas.width = resultImage.offsetWidth;
    paintCanvas.height = resultImage.offsetHeight;
    paintCanvas.style.left = imgLeft + 'px';
    paintCanvas.style.top = imgTop + 'px';
    paintCanvas.classList.remove('hidden');
}

function setPaintMode(mode) {
    if (paintMode === mode) {
        paintMode = null;
        keepBtn.classList.remove('active');
        removeBtn.classList.remove('active');
        brushCursor.classList.add('hidden');
    } else {
        paintMode = mode;
        keepBtn.classList.toggle('active', mode === 'keep');
        removeBtn.classList.toggle('active', mode === 'remove');
        updateBrushCursorAppearance();
    }
}

function startPainting(e) {
    if (!paintMode) return;
    isPainting = true;
    paint(e);
}

function paint(e) {
    if (!isPainting || !paintMode) return;

    const rect = paintCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Scale to original image coordinates
    const scaleX = originalSize[0] / paintCanvas.width;
    const scaleY = originalSize[1] / paintCanvas.height;

    paintPoints[paintMode].push([x * scaleX, y * scaleY]);

    // Draw on canvas
    const ctx = paintCanvas.getContext('2d');
    ctx.beginPath();
    ctx.arc(x, y, paintPoints.brushSize / 2, 0, Math.PI * 2);
    ctx.fillStyle = paintMode === 'keep' ? 'rgba(0, 255, 100, 0.5)' : 'rgba(255, 50, 50, 0.5)';
    ctx.fill();
}

function stopPainting() {
    isPainting = false;
}

function handleCanvasEnter() {
    if (paintMode) {
        updateBrushCursorAppearance();
        brushCursor.classList.remove('hidden');
    }
}

function handleCanvasLeave() {
    stopPainting();
    brushCursor.classList.add('hidden');
}

function updateBrushCursor(e) {
    if (!paintMode) {
        brushCursor.classList.add('hidden');
        return;
    }

    const rect = paintCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Position relative to the result wrapper
    const wrapperRect = resultWrapper.getBoundingClientRect();
    const canvasLeft = parseFloat(paintCanvas.style.left) || 0;
    const canvasTop = parseFloat(paintCanvas.style.top) || 0;

    brushCursor.style.left = (canvasLeft + x) + 'px';
    brushCursor.style.top = (canvasTop + y) + 'px';
    brushCursor.classList.remove('hidden');
}

function updateBrushCursorAppearance() {
    const size = parseInt(brushSizeSlider.value);
    brushCursor.style.width = size + 'px';
    brushCursor.style.height = size + 'px';

    brushCursor.classList.remove('keep', 'remove');
    if (paintMode) {
        brushCursor.classList.add(paintMode);
    }
}

function handleTouchStart(e) {
    e.preventDefault();
    const touch = e.touches[0];
    startPainting({ clientX: touch.clientX, clientY: touch.clientY });
}

function handleTouchMove(e) {
    e.preventDefault();
    const touch = e.touches[0];
    paint({ clientX: touch.clientX, clientY: touch.clientY });
}

function clearPaint() {
    paintPoints = { keep: [], remove: [], brushSize: parseInt(brushSizeSlider.value) };
    const ctx = paintCanvas.getContext('2d');
    ctx.clearRect(0, 0, paintCanvas.width, paintCanvas.height);
    paintMode = null;
    keepBtn.classList.remove('active');
    removeBtn.classList.remove('active');
    brushCursor.classList.add('hidden');
}

// Apply Adjustments
async function applyAdjustments() {
    applyBtn.disabled = true;
    applyBtn.textContent = 'Applying...';

    try {
        const response = await fetch('/adjust', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_id: currentImageId,
                threshold: parseInt(sensitivitySlider.value),
                mask_data: paintPoints,
                original_size: originalSize
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Apply effects and refresh
        refreshResultImage();

        // Clear paint after applying
        clearPaint();

    } catch (error) {
        alert('Error applying adjustments: ' + error.message);
    } finally {
        applyBtn.disabled = false;
        applyBtn.textContent = 'Apply Adjustments';
    }
}

// ==================== SUBJECT ZOOM ====================

async function resetToOriginal() {
    resetImageBtn.disabled = true;
    resetImageBtn.textContent = 'Resetting...';

    try {
        const response = await fetch('/reset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_id: currentImageId
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Refresh result image
        const timestamp = Date.now();
        resultImage.src = `/image/output-white/${currentImageId}?t=${timestamp}`;

        // Reset zoom slider
        subjectZoomSlider.value = 100;
        subjectZoomValue.textContent = '100%';

    } catch (error) {
        alert('Error resetting image: ' + error.message);
    } finally {
        resetImageBtn.disabled = false;
        resetImageBtn.textContent = 'Reset';
    }
}

async function applySubjectZoom() {
    const zoomLevel = parseInt(subjectZoomSlider.value);

    if (zoomLevel === 100) {
        return; // No zoom to apply
    }

    applyZoomBtn.disabled = true;
    applyZoomBtn.textContent = 'Applying...';

    try {
        const response = await fetch('/adjust', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_id: currentImageId,
                threshold: 50,  // Don't change threshold
                subject_zoom: zoomLevel
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Apply effects and refresh
        refreshResultImage();

        // Reset zoom slider after applying
        subjectZoomSlider.value = 100;
        subjectZoomValue.textContent = '100%';

    } catch (error) {
        alert('Error applying zoom: ' + error.message);
    } finally {
        applyZoomBtn.disabled = false;
        applyZoomBtn.textContent = 'Apply Zoom';
    }
}

// ==================== COLOR CORRECTION ====================

async function handleColorPreset(e) {
    const btn = e.target;
    const type = btn.dataset.type;
    const value = parseInt(btn.dataset.value);

    if (!currentImageId) {
        alert('No image loaded');
        return;
    }

    // Disable button while processing
    btn.disabled = true;
    const originalText = btn.textContent;
    btn.textContent = '...';

    try {
        const payload = {
            image_id: currentImageId,
            temperature: 0,
            tint: 0,
            brightness: 0
        };

        // Set the appropriate value based on type
        payload[type] = value;

        const response = await fetch('/color_correct', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Apply effects and refresh
        refreshResultImage();

    } catch (error) {
        alert('Error applying color correction: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
}

async function resetColors() {
    if (!currentImageId) {
        alert('No image loaded');
        return;
    }

    resetColorsBtn.disabled = true;
    resetColorsBtn.textContent = 'Resetting...';

    try {
        const response = await fetch('/reset_colors', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_id: currentImageId
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Apply effects and refresh
        refreshResultImage();

    } catch (error) {
        alert('Error resetting colors: ' + error.message);
    } finally {
        resetColorsBtn.disabled = false;
        resetColorsBtn.textContent = 'Reset Colors';
    }
}

// ==================== IMAGE REFRESH ====================

function refreshResultImage() {
    if (!currentImageId) return;
    const timestamp = Date.now();
    resultImage.src = `/image/adjusted-white/${currentImageId}?t=${timestamp}`;
}

// ==================== FULLSCREEN EDIT MODE ====================

let fsPaintMode = null;
let fsPaintPoints = { keep: [], remove: [], brushSize: 15 };
let fsIsPainting = false;
let fsZoom = 100;

function openFullscreen() {
    if (!currentImageId) return;

    fullscreenMode = true;
    fullscreenModal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    // Load the current image
    const timestamp = Date.now();
    fsResultImage.src = `/image/adjusted-white/${currentImageId}?t=${timestamp}`;

    // Setup canvas after image loads
    fsResultImage.onload = setupFullscreenCanvas;

    // Reset state
    fsPaintMode = null;
    fsPaintPoints = { keep: [], remove: [], brushSize: parseInt(fsBrushSize.value) };
    fsZoom = 100;
    fsZoomSlider.value = 100;
    fsZoomValue.textContent = '100%';
    fsKeepBtn.classList.remove('active');
    fsRemoveBtn.classList.remove('active');
}

function closeFullscreen() {
    fullscreenMode = false;
    fullscreenModal.classList.add('hidden');
    document.body.style.overflow = '';
    fsPaintMode = null;
    fsBrushCursor.classList.add('hidden');
}

function setupFullscreenCanvas() {
    // Match canvas size to image
    fsPaintCanvas.width = fsResultImage.naturalWidth;
    fsPaintCanvas.height = fsResultImage.naturalHeight;
    fsPaintCanvas.style.width = fsResultImage.offsetWidth + 'px';
    fsPaintCanvas.style.height = fsResultImage.offsetHeight + 'px';
}

function setFullscreenPaintMode(mode) {
    if (fsPaintMode === mode) {
        fsPaintMode = null;
        fsKeepBtn.classList.remove('active');
        fsRemoveBtn.classList.remove('active');
        fsBrushCursor.classList.add('hidden');
    } else {
        fsPaintMode = mode;
        fsKeepBtn.classList.toggle('active', mode === 'keep');
        fsRemoveBtn.classList.toggle('active', mode === 'remove');
        updateFullscreenBrushAppearance();
    }
}

function updateFullscreenBrush() {
    const size = parseInt(fsBrushSize.value);
    fsBrushSizeValue.textContent = size;
    fsPaintPoints.brushSize = size;
    updateFullscreenBrushAppearance();
}

function updateFullscreenBrushAppearance() {
    const size = parseInt(fsBrushSize.value);
    fsBrushCursor.style.width = size + 'px';
    fsBrushCursor.style.height = size + 'px';
    fsBrushCursor.classList.remove('keep', 'remove');
    if (fsPaintMode) {
        fsBrushCursor.classList.add(fsPaintMode);
    }
}

function handleFullscreenZoom(e) {
    fsZoom = parseInt(e.target.value);
    fsZoomValue.textContent = fsZoom + '%';

    const scale = fsZoom / 100;
    fsImageWrapper.style.transform = `scale(${scale})`;
    fsImageWrapper.style.transformOrigin = 'center center';
}

function startFullscreenPaint(e) {
    if (!fsPaintMode) return;
    fsIsPainting = true;
    fullscreenPaint(e);
}

function fullscreenPaint(e) {
    if (!fsIsPainting || !fsPaintMode) return;

    const rect = fsPaintCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Scale to actual canvas coordinates
    const scaleX = fsPaintCanvas.width / rect.width;
    const scaleY = fsPaintCanvas.height / rect.height;

    // Store points in original image coordinates
    fsPaintPoints[fsPaintMode].push([x * scaleX, y * scaleY]);

    // Draw on canvas (visual feedback)
    const ctx = fsPaintCanvas.getContext('2d');
    const brushRadius = fsPaintPoints.brushSize / 2 * scaleX;
    ctx.beginPath();
    ctx.arc(x * scaleX, y * scaleY, brushRadius, 0, Math.PI * 2);
    ctx.fillStyle = fsPaintMode === 'keep' ? 'rgba(0, 255, 100, 0.5)' : 'rgba(255, 50, 50, 0.5)';
    ctx.fill();
}

function stopFullscreenPaint() {
    fsIsPainting = false;
}

function handleFullscreenCanvasLeave() {
    stopFullscreenPaint();
    fsBrushCursor.classList.add('hidden');
}

function updateFullscreenBrushCursor(e) {
    if (!fsPaintMode) {
        fsBrushCursor.classList.add('hidden');
        return;
    }

    const rect = fsPaintCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Position cursor relative to canvas
    fsBrushCursor.style.left = (x + fsPaintCanvas.offsetLeft) + 'px';
    fsBrushCursor.style.top = (y + fsPaintCanvas.offsetTop) + 'px';
    fsBrushCursor.classList.remove('hidden');
}

function handleFullscreenCanvasEnter() {
    if (fsPaintMode) {
        updateFullscreenBrushAppearance();
        fsBrushCursor.classList.remove('hidden');
    }
}

function clearFullscreenPaint() {
    fsPaintPoints = { keep: [], remove: [], brushSize: parseInt(fsBrushSize.value) };
    const ctx = fsPaintCanvas.getContext('2d');
    ctx.clearRect(0, 0, fsPaintCanvas.width, fsPaintCanvas.height);
    fsPaintMode = null;
    fsKeepBtn.classList.remove('active');
    fsRemoveBtn.classList.remove('active');
    fsBrushCursor.classList.add('hidden');
}

async function previewFullscreenChanges() {
    if (!fsPaintPoints.keep.length && !fsPaintPoints.remove.length) {
        return;
    }

    fsPreviewBtn.disabled = true;
    fsPreviewBtn.textContent = 'Updating...';

    try {
        const response = await fetch('/adjust', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_id: currentImageId,
                threshold: parseInt(sensitivitySlider.value),
                mask_data: fsPaintPoints,
                original_size: processedSize
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Refresh fullscreen image
        const timestamp = Date.now();
        fsResultImage.src = `/image/adjusted-white/${currentImageId}?t=${timestamp}`;

        // Clear paint points after preview (they've been applied)
        clearFullscreenPaint();

    } catch (error) {
        alert('Error previewing changes: ' + error.message);
    } finally {
        fsPreviewBtn.disabled = false;
        fsPreviewBtn.textContent = 'Preview';
    }
}

async function recenterInFullscreen() {
    fsRecenterBtn.disabled = true;
    fsRecenterBtn.textContent = 'Centering...';

    try {
        const response = await fetch('/recenter', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_id: currentImageId
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Refresh fullscreen image
        const timestamp = Date.now();
        fsResultImage.src = `/image/adjusted-white/${currentImageId}?t=${timestamp}`;

        // Re-setup canvas for new image
        fsResultImage.onload = setupFullscreenCanvas;

    } catch (error) {
        alert('Error recentering: ' + error.message);
    } finally {
        fsRecenterBtn.disabled = false;
        fsRecenterBtn.textContent = 'Recenter';
    }
}

async function recenterSubject() {
    recenterBtn.disabled = true;
    recenterBtn.textContent = 'Centering...';

    try {
        const response = await fetch('/recenter', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_id: currentImageId
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Refresh result image with effects
        refreshResultImage();

    } catch (error) {
        alert('Error recentering: ' + error.message);
    } finally {
        recenterBtn.disabled = false;
        recenterBtn.textContent = 'Recenter';
    }
}

async function applyFullscreenChanges() {
    if (!fsPaintPoints.keep.length && !fsPaintPoints.remove.length) {
        closeFullscreen();
        return;
    }

    fsApplyBtn.disabled = true;
    fsApplyBtn.textContent = 'Applying...';

    try {
        const response = await fetch('/adjust', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_id: currentImageId,
                threshold: parseInt(sensitivitySlider.value),
                mask_data: fsPaintPoints,
                original_size: processedSize
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Refresh with effects
        refreshResultImage();

        // Close fullscreen and reset main canvas
        closeFullscreen();
        clearPaint();
        setupCanvas();

    } catch (error) {
        alert('Error applying changes: ' + error.message);
    } finally {
        fsApplyBtn.disabled = false;
        fsApplyBtn.textContent = 'Apply & Close';
    }
}

// Download
function downloadImage(background) {
    const url = `/download/${currentImageId}?background=${background}`;
    window.location.href = url;
}

// ==================== GoPro Functions ====================

async function checkGoproStatus() {
    try {
        const response = await fetch('/gopro/status');
        const data = await response.json();

        if (data.available) {
            goproPreviewSection.classList.remove('hidden');

            if (data.connected) {
                goproAvailable = true;
                goproCaptureBtn.classList.add('connected');
                goproStatus.textContent = `Connected at ${data.ip_address}`;
                goproStatus.classList.remove('error', 'warning');
                goproStatus.classList.add('success');

                // Also check stream status
                checkStreamStatus();
            } else {
                goproAvailable = false;
                goproCaptureBtn.classList.remove('connected');
                goproStatus.textContent = 'GoPro not reachable. Check camera is on and connected to WiFi.';
                goproStatus.classList.remove('success');
                goproStatus.classList.add('error');
            }
        } else {
            // GoPro not configured
            goproPreviewSection.classList.add('hidden');
        }
    } catch (error) {
        console.error('Error checking GoPro status:', error);
        goproPreviewSection.classList.add('hidden');
    }
}

// ==================== Stream Functions ====================

async function toggleStream() {
    if (streamActive) {
        await stopStream();
    } else {
        await startStream();
    }
}

async function startStream() {
    streamToggleBtn.disabled = true;
    streamToggleBtn.textContent = 'Starting...';
    updateStreamStatus('connecting');

    try {
        const response = await fetch('/gopro/stream/start', { method: 'POST' });
        const data = await response.json();
        console.log('Stream start response:', data);

        if (data.success) {
            streamActive = true;
            streamToggleBtn.textContent = 'Stop Stream';
            streamToggleBtn.classList.add('active');
            updateStreamStatus('live');

            // Show the stream feed with cache-busting
            streamFeed.src = '/gopro/stream/feed?t=' + Date.now();
            streamFeed.classList.remove('hidden');
            previewPlaceholder.classList.add('hidden');
            previewImage.classList.add('hidden');

            // Start status checking
            startStreamStatusCheck();

            goproStatus.textContent = 'Preview active (updates every ~2 seconds)';
            goproStatus.classList.remove('error');
            goproStatus.classList.add('success');
        } else {
            const errorMsg = data.error || (data.status === 'error' ? 'Stream failed to start' : 'Unknown error');
            throw new Error(errorMsg);
        }
    } catch (error) {
        console.error('Error starting stream:', error);
        updateStreamStatus('error', error.message);
        goproStatus.textContent = `Stream error: ${error.message}`;
        goproStatus.classList.remove('success');
        goproStatus.classList.add('error');
        streamToggleBtn.textContent = 'Start Stream';
        streamToggleBtn.classList.remove('active');
    } finally {
        streamToggleBtn.disabled = false;
    }
}

async function stopStream() {
    streamToggleBtn.disabled = true;
    streamToggleBtn.textContent = 'Stopping...';

    try {
        await fetch('/gopro/stream/stop', { method: 'POST' });
    } catch (error) {
        console.error('Error stopping stream:', error);
    }

    streamActive = false;
    streamToggleBtn.textContent = 'Start Stream';
    streamToggleBtn.classList.remove('active');
    streamToggleBtn.disabled = false;
    updateStreamStatus('offline');

    // Hide stream feed, show placeholder
    streamFeed.src = '';
    streamFeed.classList.add('hidden');
    previewPlaceholder.classList.remove('hidden');

    // Stop status checking
    stopStreamStatusCheck();
}

function updateStreamStatus(status, message = '') {
    const indicator = streamStatusIndicator;
    const dot = indicator.querySelector('.status-dot');
    const text = indicator.querySelector('.status-text');

    // Remove all status classes
    indicator.classList.remove('connecting', 'live', 'error', 'offline');
    indicator.classList.add(status);

    switch (status) {
        case 'connecting':
            text.textContent = 'Connecting...';
            break;
        case 'live':
            text.textContent = 'Live';
            break;
        case 'error':
            text.textContent = message || 'Error';
            break;
        case 'offline':
        default:
            text.textContent = 'Offline';
            break;
    }
}

async function checkStreamStatus() {
    try {
        const response = await fetch('/gopro/stream/status');
        const data = await response.json();

        if (data.streaming && !streamActive) {
            // Stream is running but UI doesn't know - sync up
            streamActive = true;
            streamToggleBtn.textContent = 'Stop Stream';
            streamToggleBtn.classList.add('active');
            updateStreamStatus('live');
            streamFeed.src = '/gopro/stream/feed';
            streamFeed.classList.remove('hidden');
            previewPlaceholder.classList.add('hidden');
            startStreamStatusCheck();
        } else if (!data.streaming && streamActive) {
            // Stream stopped unexpectedly
            streamActive = false;
            streamToggleBtn.textContent = 'Start Stream';
            streamToggleBtn.classList.remove('active');
            updateStreamStatus(data.status === 'error' ? 'error' : 'offline', data.error);
            streamFeed.classList.add('hidden');
            previewPlaceholder.classList.remove('hidden');
            stopStreamStatusCheck();
        }
    } catch (error) {
        console.error('Error checking stream status:', error);
    }
}

function startStreamStatusCheck() {
    stopStreamStatusCheck();
    streamStatusCheckInterval = setInterval(checkStreamStatus, 3000);
}

function stopStreamStatusCheck() {
    if (streamStatusCheckInterval) {
        clearInterval(streamStatusCheckInterval);
        streamStatusCheckInterval = null;
    }
}

function handleCameraZoomInput(e) {
    cameraZoomValue.textContent = e.target.value + '%';
}

async function setCameraZoom() {
    const percent = parseInt(cameraZoomSlider.value);

    try {
        const response = await fetch('/gopro/zoom', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ percent })
        });

        const data = await response.json();
        if (!data.success) {
            console.error('Zoom error:', data.error);
        }
    } catch (error) {
        console.error('Error setting zoom:', error);
    }
}

async function setWhiteBalance() {
    const option = parseInt(whiteBalanceSelect.value);

    try {
        const response = await fetch('/gopro/white_balance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ option })
        });

        const data = await response.json();
        if (!data.success) {
            console.error('White balance error:', data.error);
            alert('Failed to set white balance: ' + data.error);
        } else {
            console.log('White balance set to option:', option);
        }
    } catch (error) {
        console.error('Error setting white balance:', error);
        alert('Error setting white balance: ' + error.message);
    }
}

async function captureWithGoPro() {
    if (!goproAvailable) {
        await checkGoproStatus();
        if (!goproAvailable) {
            alert('GoPro is not connected. Please ensure the camera is on and connected to your WiFi network.');
            return;
        }
    }

    captureInProgress = true;
    showProgress();

    // Use stream snapshot if stream is active (instant capture)
    if (streamActive) {
        updateProgress(10, 'Capturing from stream...');

        try {
            const response = await fetch('/gopro/stream/snapshot', { method: 'POST' });

            updateProgress(50, 'Processing image...');

            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                throw new Error(data.error || `Capture failed (${response.status})`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            updateProgress(90, 'Finalizing...');

            currentImageId = data.image_id;
            originalSize = data.original_size;
            processedSize = data.processed_size;

            displayStreamCaptureResults(data);

            updateProgress(100, 'Complete!');

            setTimeout(() => {
                showResults();
            }, 500);

        } catch (error) {
            alert('Error capturing from stream: ' + error.message);
            resetUI();
        } finally {
            captureInProgress = false;
        }
        return;
    }

    // Traditional capture (shutter trigger)
    updateProgress(5, 'Connecting to GoPro...');

    try {
        updateProgress(15, 'Setting photo mode...');

        const response = await fetch('/gopro/capture', {
            method: 'POST'
        });

        updateProgress(40, 'Capturing photo...');

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.error || `Capture failed (${response.status})`);
        }

        updateProgress(70, 'Processing image...');

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        updateProgress(90, 'Finalizing...');

        currentImageId = data.image_id;
        originalSize = data.original_size;
        processedSize = data.processed_size;

        // Display results with GoPro-specific timing
        displayGoproResults(data);

        updateProgress(100, 'Complete!');

        setTimeout(() => {
            showResults();
        }, 500);

    } catch (error) {
        alert('Error capturing with GoPro: ' + error.message);
        resetUI();
        checkGoproStatus();
    } finally {
        captureInProgress = false;
    }
}

function displayGoproResults(data) {
    // Display timing information (GoPro has extra steps)
    const timings = data.timings;
    const usedPreview = data.used_preview;

    let timingHtml = '';
    if (usedPreview) {
        timingHtml = `
            <div class="timing-item">
                <span class="label">Source</span>
                <span class="value">Preview</span>
            </div>
        `;
    } else {
        timingHtml = `
            <div class="timing-item">
                <span class="label">Capture</span>
                <span class="value">${timings.capture}ms</span>
            </div>
            <div class="timing-item">
                <span class="label">Download</span>
                <span class="value">${timings.download}ms</span>
            </div>
        `;
    }

    timingDetails.innerHTML = timingHtml + `
        <div class="timing-item">
            <span class="label">Remove BG</span>
            <span class="value">${timings.remove_bg}ms</span>
        </div>
        <div class="timing-item total">
            <span class="label">Total</span>
            <span class="value">${timings.total}ms</span>
        </div>
    `;

    // Add cache-busting timestamp
    const timestamp = Date.now();
    // Display images
    originalImage.src = `/image/original/${currentImageId}?t=${timestamp}`;
    resultImage.src = `/image/output-white/${currentImageId}?t=${timestamp}`;

    // Setup canvas after image loads
    resultImage.onload = () => {
        setupCanvas();
    };

    // Reset adjustments
    sensitivitySlider.value = 50;
    sensitivityValue.textContent = '50';
    clearPaint();

    // Reset subject zoom
    subjectZoomSlider.value = 100;
    subjectZoomValue.textContent = '100%';
}

function displayStreamCaptureResults(data) {
    // Display timing information for stream capture (instant)
    const timings = data.timings;

    timingDetails.innerHTML = `
        <div class="timing-item">
            <span class="label">Snapshot</span>
            <span class="value">${timings.snapshot}ms</span>
        </div>
        <div class="timing-item">
            <span class="label">Remove BG</span>
            <span class="value">${timings.remove_bg}ms</span>
        </div>
        <div class="timing-item total">
            <span class="label">Total</span>
            <span class="value">${timings.total}ms</span>
        </div>
    `;

    // Add cache-busting timestamp
    const timestamp = Date.now();
    // Display images
    originalImage.src = `/image/original/${currentImageId}?t=${timestamp}`;
    resultImage.src = `/image/output-white/${currentImageId}?t=${timestamp}`;

    // Setup canvas after image loads
    resultImage.onload = () => {
        setupCanvas();
    };

    // Reset adjustments
    sensitivitySlider.value = 50;
    sensitivityValue.textContent = '50';
    clearPaint();

    // Reset subject zoom
    subjectZoomSlider.value = 100;
    subjectZoomValue.textContent = '100%';
}

// Reset
function resetUI() {
    currentImageId = null;
    originalSize = [0, 0];
    processedSize = [0, 0];
    paintMode = null;
    isPainting = false;
    paintPoints = { keep: [], remove: [], brushSize: 15 };

    // Close fullscreen if open
    if (fullscreenMode) {
        closeFullscreen();
    }

    uploadSection.classList.remove('hidden');
    progressSection.classList.add('hidden');
    resultsSection.classList.add('hidden');

    progressFill.style.width = '0%';
    fileInput.value = '';

    // Reset subject zoom
    subjectZoomSlider.value = 100;
    subjectZoomValue.textContent = '100%';

    // Recheck GoPro status when returning to upload screen
    checkGoproStatus();
}
