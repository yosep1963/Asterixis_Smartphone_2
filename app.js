/**
 * Asterixis Detection Web App v5.1
 * MediaPipe Hands + ONNX Runtime Web
 * Compatible with iOS Safari and Android Chrome
 *
 * v5.1 - Fingertip-focused Grade 1 sensitivity improvement
 * - Severity based on fingertip velocity (not whole-hand average)
 * - Acceleration adjustment added (aligned with Python pipeline)
 * - Reduced EMA smoothing for faster Grade 1 response
 * - Detection interval shortened (66ms)
 * - False positive protection: wrist-normalized coords ignore hand translation
 */

// ============================================================================
// Configuration (aligned with Python src/config.py)
// ============================================================================
const CONFIG = {
    // Feature dimensions
    NUM_LANDMARKS: 21,
    COORD_DIM: 42,
    ANGLE_DIM: 1,
    VELOCITY_DIM: 42,
    ACCELERATION_DIM: 42,
    SINGLE_HAND_FEATURE_DIM: 127,
    FEATURE_DIM: 254,

    // Window settings
    WINDOW_SIZE: 15,

    // Detection thresholds (aligned with Python config.py)
    CONFIDENCE_THRESHOLD: 0.5,
    MIN_HAND_SIZE: 0.01,

    // Motion detection (aligned with Python MOTION_THRESHOLDS)
    MIN_VELOCITY_FOR_DETECTION: 0.005,

    // Severity thresholds
    SEVERITY: {
        NORMAL: 25,
        GRADE1: 50,
        GRADE2: 75,
        GRADE3: 100
    },

    // Velocity thresholds for Grade — fingertip-based (lower than whole-hand)
    VELOCITY_THRESHOLDS: {
        GRADE1_MIN: 0.006,
        GRADE2_MIN: 0.018,
        GRADE3_MIN: 0.040
    },

    // Fingertip landmark indices (in 42-dim normalized coord space, x,y pairs)
    // Landmark 4(thumb), 8(index), 12(middle), 16(ring), 20(pinky)
    FINGERTIP_COORD_INDICES: [8,9, 16,17, 24,25, 32,33, 40,41],

    // Buffer fill requirement before detection starts
    MIN_BUFFER_FILL: 15,

    // Severity history length (10s at ~15 detections/sec)
    SEVERITY_HISTORY_SIZE: 150,

    // Severity smoothing (EMA alpha) — higher = faster response
    SEVERITY_SMOOTHING: 0.5
};

// ============================================================================
// Global State
// ============================================================================
let hands = null;
let session = null;
let isRunning = false;
let facingMode = 'user';

// Feature buffers
let leftCoordsBuffer = [];
let leftAnglesBuffer = [];
let rightCoordsBuffer = [];
let rightAnglesBuffer = [];

// FPS calculation
let frameCount = 0;
let lastFpsTime = Date.now();
let currentFps = 0;

// Detection state
let lastDetectionTime = 0;
const DETECTION_INTERVAL = 66; // ms (~15 detections/sec)
let isDetecting = false;

// Hand tracking
let realHandFrameCount = 0;
let noHandFrameCount = 0;
const NO_HAND_RESET_THRESHOLD = 15;

// Severity history (for sparkline)
let severityHistory = [];

// Smoothed severity (EMA)
let smoothedSeverity = 0;

// ============================================================================
// DOM Elements
// ============================================================================
const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const startBtn = document.getElementById('start-btn');
const switchCameraBtn = document.getElementById('switch-camera-btn');
const fpsDisplay = document.getElementById('fps-display');
const handStatus = document.getElementById('hand-status');
const motionStatus = document.getElementById('motion-status');
const detectionResult = document.getElementById('detection-result');
const severityBar = document.getElementById('severity-bar');
const severityValue = document.getElementById('severity-value');
const historyCanvas = document.getElementById('history-canvas');
const historyCtx = historyCanvas.getContext('2d');

// ============================================================================
// Logging
// ============================================================================
function log(message) {
    console.log(`[Asterixis] ${message}`);
}

function logError(message, error) {
    console.error(`[Asterixis] ${message}`, error);
}

// ============================================================================
// Initialization
// ============================================================================
async function init() {
    log('Initializing v5.1...');
    updateLoadingText('Initializing...');

    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera not supported in this browser');
        }

        log('Loading MediaPipe Hands...');
        updateLoadingText('Loading Hand Detection...');

        hands = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`;
            }
        });

        hands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.5
        });

        hands.onResults(onResults);

        log('Warming up hand model...');
        updateLoadingText('Preparing Hand Model...');

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 640;
        tempCanvas.height = 480;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.fillStyle = '#000';
        tempCtx.fillRect(0, 0, 640, 480);
        await hands.send({ image: tempCanvas });
        log('MediaPipe Hands ready');

        log('Loading ONNX model...');
        updateLoadingText('Loading AI Model...');

        session = await ort.InferenceSession.create('./models/asterixis_model.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        log('ONNX model loaded');

        log('Ready');
        updateLoadingText('Ready!');
        loadingOverlay.classList.add('hidden');
        startBtn.disabled = false;
        startBtn.textContent = 'Start Detection';

    } catch (error) {
        logError('Initialization failed', error);
        updateLoadingText(`Error: ${error.message}`);
        startBtn.disabled = false;
        startBtn.textContent = 'Retry';
        startBtn.onclick = () => location.reload();
    }
}

function updateLoadingText(text) {
    if (loadingText) loadingText.textContent = text;
}

// ============================================================================
// Camera Control
// ============================================================================
async function startCamera() {
    log('Starting camera...');

    try {
        if (videoElement.srcObject) {
            videoElement.srcObject.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null;
        }

        const constraintsList = [
            { video: { facingMode: { ideal: facingMode }, width: { ideal: 720 }, height: { ideal: 1280 } }, audio: false },
            { video: { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
            { video: { facingMode: facingMode }, audio: false },
            { video: true, audio: false }
        ];

        let stream = null;
        for (const constraints of constraintsList) {
            try {
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                break;
            } catch (e) {
                continue;
            }
        }

        if (!stream) throw new Error('Could not access any camera');

        videoElement.srcObject = stream;

        await new Promise((resolve, reject) => {
            const timeoutId = setTimeout(() => reject(new Error('Video load timeout')), 10000);
            videoElement.onloadedmetadata = () => { clearTimeout(timeoutId); resolve(); };
            videoElement.onerror = (e) => { clearTimeout(timeoutId); reject(new Error('Video error')); };
        });

        try { await videoElement.play(); } catch (e) { /* may be ok on some browsers */ }

        canvasElement.width = videoElement.videoWidth || 640;
        canvasElement.height = videoElement.videoHeight || 480;

        isRunning = true;
        startBtn.textContent = 'Stop';
        log('Camera started');

        processVideoFrame();

    } catch (error) {
        logError('Camera start failed', error);

        let msg = 'Camera access failed.\n\n';
        if (error.name === 'NotAllowedError') msg += 'Please allow camera permission in browser settings.';
        else if (error.name === 'NotFoundError') msg += 'No camera found on this device.';
        else if (error.name === 'NotReadableError') msg += 'Camera is in use by another app.';
        else if (error.name === 'SecurityError') msg += 'Camera access blocked. HTTPS required.';
        else msg += error.message || 'Unknown error';

        alert(msg);
    }
}

async function processVideoFrame() {
    if (!isRunning) return;

    try {
        if (hands && videoElement.readyState >= 2) {
            await hands.send({ image: videoElement });
        }
    } catch (e) {
        // Ignore frame processing errors
    }

    requestAnimationFrame(processVideoFrame);
}

async function stopCamera() {
    log('Stopping camera...');
    isRunning = false;

    if (videoElement.srcObject) {
        videoElement.srcObject.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
    }

    resetBuffers();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    startBtn.textContent = 'Start Detection';
    updateUI(null, 0, 0);
    handStatus.textContent = 'Hands: --';
    motionStatus.textContent = 'Motion: --';
    motionStatus.className = 'motion-idle';
    fpsDisplay.textContent = 'FPS: --';

    log('Camera stopped');
}

function resetBuffers() {
    leftCoordsBuffer = [];
    leftAnglesBuffer = [];
    rightCoordsBuffer = [];
    rightAnglesBuffer = [];
    realHandFrameCount = 0;
    noHandFrameCount = 0;
    smoothedSeverity = 0;
    severityHistory = [];
    drawSeverityHistory();
}

async function switchCamera() {
    facingMode = facingMode === 'user' ? 'environment' : 'user';
    log(`Switching to ${facingMode} camera`);
    if (isRunning) {
        await stopCamera();
        await startCamera();
    }
}

// ============================================================================
// MediaPipe Results Handler
// ============================================================================
function onResults(results) {
    if (!isRunning) return;

    const now = Date.now();

    // Update FPS
    frameCount++;
    if (now - lastFpsTime >= 1000) {
        currentFps = frameCount;
        frameCount = 0;
        lastFpsTime = now;
        fpsDisplay.textContent = `FPS: ${currentFps}`;
    }

    // Draw frame
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (facingMode === 'user') {
        canvasCtx.translate(canvasElement.width, 0);
        canvasCtx.scale(-1, 1);
    }

    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    let leftHand = null;
    let rightHand = null;
    let numHands = 0;

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        numHands = results.multiHandLandmarks.length;

        for (let i = 0; i < results.multiHandLandmarks.length; i++) {
            const landmarks = results.multiHandLandmarks[i];

            let label = 'Right';
            if (results.multiHandedness && results.multiHandedness[i] &&
                results.multiHandedness[i].classification &&
                results.multiHandedness[i].classification[0]) {
                label = results.multiHandedness[i].classification[0].label;
            }

            // MediaPipe mirror: Left label = actual right hand
            if (label === 'Left') {
                rightHand = landmarks;
            } else {
                leftHand = landmarks;
            }

            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                color: label === 'Left' ? '#00FF00' : '#FF6600',
                lineWidth: 3
            });
            drawLandmarks(canvasCtx, landmarks, {
                color: '#FFFFFF',
                fillColor: label === 'Left' ? '#00FF00' : '#FF6600',
                lineWidth: 1,
                radius: 4
            });
        }
    }

    canvasCtx.restore();

    // Update hand status
    handStatus.textContent = `Hands: ${numHands}/2`;
    handStatus.style.color = numHands > 0 ? '#64ffda' : '#ff5252';

    if (numHands === 0) {
        noHandFrameCount++;
        if (noHandFrameCount >= NO_HAND_RESET_THRESHOLD) {
            updateUI(null, 0, 0);
            resetBuffers();
        }
        return;
    }

    noHandFrameCount = 0;

    // Update feature buffers
    updateFeatureBuffers(leftHand, rightHand);

    // Run detection at controlled interval
    if (leftCoordsBuffer.length >= CONFIG.MIN_BUFFER_FILL &&
        realHandFrameCount >= CONFIG.MIN_BUFFER_FILL &&
        !isDetecting &&
        now - lastDetectionTime >= DETECTION_INTERVAL) {
        lastDetectionTime = now;
        runDetection();
    }
}

// ============================================================================
// Feature Extraction (aligned with Python preprocessor.py)
// ============================================================================
function normalizeHandLandmarks(landmarks) {
    if (!landmarks || landmarks.length !== CONFIG.NUM_LANDMARKS) return null;

    const wristX = landmarks[0].x;
    const wristY = landmarks[0].y;

    const middleTipX = landmarks[12].x;
    const middleTipY = landmarks[12].y;
    const handSize = Math.sqrt(
        Math.pow(middleTipX - wristX, 2) +
        Math.pow(middleTipY - wristY, 2)
    );

    if (handSize < CONFIG.MIN_HAND_SIZE) return null;

    const normalized = [];
    for (const lm of landmarks) {
        normalized.push((lm.x - wristX) / handSize, (lm.y - wristY) / handSize);
    }

    return normalized;
}

function calculateWristAngle(landmarks) {
    if (!landmarks) return 0;

    const wrist = [landmarks[0].x, landmarks[0].y];
    const indexMcp = [landmarks[5].x, landmarks[5].y];
    const middleMcp = [landmarks[9].x, landmarks[9].y];

    const palmCenterX = (indexMcp[0] + middleMcp[0]) / 2;
    const palmCenterY = (indexMcp[1] + middleMcp[1]) / 2;

    const angleRad = Math.atan2(palmCenterX - wrist[0], -(palmCenterY - wrist[1]));
    return angleRad * (180 / Math.PI);
}

function updateFeatureBuffers(leftHand, rightHand) {
    const leftCoords = normalizeHandLandmarks(leftHand);
    const leftAngle = calculateWristAngle(leftHand);
    const rightCoords = normalizeHandLandmarks(rightHand);
    const rightAngle = calculateWristAngle(rightHand);

    if (leftCoords !== null || rightCoords !== null) {
        realHandFrameCount++;
    } else {
        realHandFrameCount = 0;
    }

    // Left hand buffer
    if (leftCoords) {
        leftCoordsBuffer.push(leftCoords);
        leftAnglesBuffer.push(leftAngle);
    } else if (leftCoordsBuffer.length > 0) {
        leftCoordsBuffer.push([...leftCoordsBuffer[leftCoordsBuffer.length - 1]]);
        leftAnglesBuffer.push(leftAnglesBuffer[leftAnglesBuffer.length - 1]);
    } else {
        leftCoordsBuffer.push(new Array(CONFIG.COORD_DIM).fill(0));
        leftAnglesBuffer.push(0);
    }

    // Right hand buffer
    if (rightCoords) {
        rightCoordsBuffer.push(rightCoords);
        rightAnglesBuffer.push(rightAngle);
    } else if (rightCoordsBuffer.length > 0) {
        rightCoordsBuffer.push([...rightCoordsBuffer[rightCoordsBuffer.length - 1]]);
        rightAnglesBuffer.push(rightAnglesBuffer[rightAnglesBuffer.length - 1]);
    } else {
        rightCoordsBuffer.push(new Array(CONFIG.COORD_DIM).fill(0));
        rightAnglesBuffer.push(0);
    }

    // Limit buffer size
    const maxSize = CONFIG.WINDOW_SIZE + 15;
    while (leftCoordsBuffer.length > maxSize) {
        leftCoordsBuffer.shift();
        leftAnglesBuffer.shift();
        rightCoordsBuffer.shift();
        rightAnglesBuffer.shift();
    }
}

function buildSingleHandFeatures(coordsBuffer, anglesBuffer) {
    const start = coordsBuffer.length - CONFIG.WINDOW_SIZE;
    const coords = coordsBuffer.slice(start);
    const angles = anglesBuffer.slice(start);

    const velocities = [];
    const accelerations = [];

    for (let i = 0; i < CONFIG.WINDOW_SIZE; i++) {
        if (i === 0) {
            velocities.push(new Array(CONFIG.COORD_DIM).fill(0));
        } else {
            velocities.push(coords[i].map((v, j) => v - coords[i - 1][j]));
        }
    }

    for (let i = 0; i < CONFIG.WINDOW_SIZE; i++) {
        if (i === 0) {
            accelerations.push(new Array(CONFIG.COORD_DIM).fill(0));
        } else {
            accelerations.push(velocities[i].map((v, j) => v - velocities[i - 1][j]));
        }
    }

    const features = [];
    for (let i = 0; i < CONFIG.WINDOW_SIZE; i++) {
        features.push([...coords[i], angles[i], ...velocities[i], ...accelerations[i]]);
    }

    return features;
}

function buildFullFeatures() {
    const leftFeatures = buildSingleHandFeatures(leftCoordsBuffer, leftAnglesBuffer);
    const rightFeatures = buildSingleHandFeatures(rightCoordsBuffer, rightAnglesBuffer);

    const full = [];
    for (let i = 0; i < CONFIG.WINDOW_SIZE; i++) {
        full.push([...leftFeatures[i], ...rightFeatures[i]]);
    }

    return full;
}

// ============================================================================
// Motion & Joint Analysis
// Analyzes fingertip velocity (joint-level) separately from whole-hand velocity.
// Wrist-normalized coords mean pure hand translation produces zero velocity,
// so any velocity here reflects genuine joint/rotation movement.
// ============================================================================
function analyzeMotion(features) {
    // Per-hand offsets in the 254-dim feature vector
    const velOffset = CONFIG.COORD_DIM + CONFIG.ANGLE_DIM; // 43 (velocity start within a hand)
    const accelOffset = velOffset + CONFIG.VELOCITY_DIM;     // 85 (accel start within a hand)
    const handOffsets = [0, CONFIG.SINGLE_HAND_FEATURE_DIM]; // [0, 127]

    let bestMeanVel = 0;
    let bestFingertipVel = 0;
    let bestMeanAccel = 0;

    for (const handBase of handOffsets) {
        const absVelStart = handBase + velOffset;   // 43 or 170
        const absVelEnd = absVelStart + CONFIG.VELOCITY_DIM; // 85 or 212
        const absAccelStart = handBase + accelOffset; // 85 or 212
        const absAccelEnd = absAccelStart + CONFIG.ACCELERATION_DIM; // 127 or 254

        // Fingertip velocity indices within this hand's velocity block
        // CONFIG.FINGERTIP_COORD_INDICES are [8,9,16,17,24,25,32,33,40,41] in coord space
        // In feature vector: handBase + velOffset + coordIndex
        const fingertipAbsIndices = CONFIG.FINGERTIP_COORD_INDICES.map(ci => handBase + velOffset + ci);

        // 1. Mean velocity (all 42 coords) — for motion detection gate
        let velTotal = 0, velCount = 0;
        for (let i = 0; i < CONFIG.WINDOW_SIZE; i++) {
            for (let j = absVelStart; j < absVelEnd; j++) {
                velTotal += Math.abs(features[i][j]);
                velCount++;
            }
        }
        const meanVel = velCount > 0 ? velTotal / velCount : 0;

        // 2. Fingertip velocity (10 values: 5 tips × 2 coords) — for severity
        let ftTotal = 0, ftCount = 0;
        for (let i = 0; i < CONFIG.WINDOW_SIZE; i++) {
            for (const idx of fingertipAbsIndices) {
                ftTotal += Math.abs(features[i][idx]);
                ftCount++;
            }
        }
        const fingertipVel = ftCount > 0 ? ftTotal / ftCount : 0;

        // 3. Mean acceleration (all 42 coords) — for accel adjustment
        let accelTotal = 0, accelCount = 0;
        for (let i = 0; i < CONFIG.WINDOW_SIZE; i++) {
            for (let j = absAccelStart; j < absAccelEnd; j++) {
                accelTotal += Math.abs(features[i][j]);
                accelCount++;
            }
        }
        const meanAccel = accelCount > 0 ? accelTotal / accelCount : 0;

        // Take the higher hand (more severe side)
        if (meanVel > bestMeanVel) bestMeanVel = meanVel;
        if (fingertipVel > bestFingertipVel) bestFingertipVel = fingertipVel;
        if (meanAccel > bestMeanAccel) bestMeanAccel = meanAccel;
    }

    return {
        hasMotion: bestMeanVel >= CONFIG.MIN_VELOCITY_FOR_DETECTION,
        meanVelocity: bestMeanVel,
        fingertipVelocity: bestFingertipVel,
        meanAccel: bestMeanAccel
    };
}

// ============================================================================
// Severity Calculation — fingertip velocity + acceleration adjustment
// Uses fingertip velocity as primary indicator (joint-level, not whole-hand).
// Acceleration adjustment aligned with Python utils.SeverityCalculator.
// ============================================================================
function calculateSeverityScore(confidence, fingertipVelocity, meanAccel) {
    if (confidence < 0.5) return confidence * 50;

    const g1 = CONFIG.VELOCITY_THRESHOLDS.GRADE1_MIN;
    const g2 = CONFIG.VELOCITY_THRESHOLDS.GRADE2_MIN;
    const g3 = CONFIG.VELOCITY_THRESHOLDS.GRADE3_MIN;

    let severity;

    if (fingertipVelocity < g1) {
        severity = 25 + (fingertipVelocity / g1) * 20;
    } else if (fingertipVelocity < g2) {
        const ratio = (fingertipVelocity - g1) / (g2 - g1);
        severity = 30 + ratio * 20;
    } else if (fingertipVelocity < g3) {
        const ratio = (fingertipVelocity - g2) / (g3 - g2);
        severity = 50 + ratio * 25;
    } else {
        const ratio = Math.min(1.0, (fingertipVelocity - g3) / 0.03);
        severity = 75 + ratio * 25;
    }

    // Acceleration adjustment (aligned with Python: accel_adjust = min(5, mean_accel * 50))
    const accelAdjust = Math.min(5, meanAccel * 50);
    severity += accelAdjust;

    return Math.min(100, Math.max(25, Math.round(severity)));
}

// ============================================================================
// Detection (with motion check + ONNX inference)
// ============================================================================
async function runDetection() {
    if (!session || isDetecting) return;

    isDetecting = true;

    try {
        const features = buildFullFeatures();
        if (!features || features.length !== CONFIG.WINDOW_SIZE) {
            isDetecting = false;
            return;
        }

        // Motion & joint analysis
        const { hasMotion, fingertipVelocity, meanAccel } = analyzeMotion(features);

        if (!hasMotion) {
            // Stationary: force Normal regardless of model output
            motionStatus.textContent = 'Motion: Still';
            motionStatus.className = 'motion-idle';

            const severity = 0;
            smoothedSeverity = smoothedSeverity * (1 - CONFIG.SEVERITY_SMOOTHING) + severity * CONFIG.SEVERITY_SMOOTHING;
            pushSeverityHistory(smoothedSeverity);
            updateUI(false, Math.round(smoothedSeverity), 0);
            isDetecting = false;
            return;
        }

        motionStatus.textContent = 'Motion: Active';
        motionStatus.className = 'motion-active';

        // Flatten features for ONNX
        const inputData = new Float32Array(CONFIG.WINDOW_SIZE * CONFIG.FEATURE_DIM);
        for (let i = 0; i < CONFIG.WINDOW_SIZE; i++) {
            for (let j = 0; j < CONFIG.FEATURE_DIM; j++) {
                const val = features[i][j];
                inputData[i * CONFIG.FEATURE_DIM + j] = isNaN(val) ? 0 : val;
            }
        }

        const tensor = new ort.Tensor('float32', inputData, [1, CONFIG.WINDOW_SIZE, CONFIG.FEATURE_DIM]);
        const results = await session.run({ 'input': tensor });
        const confidence = results['output'].data[0];

        let isAsterixis = false;
        let severity = 0;

        if (confidence > CONFIG.CONFIDENCE_THRESHOLD) {
            isAsterixis = true;
            severity = calculateSeverityScore(confidence, fingertipVelocity, meanAccel);
        } else {
            severity = confidence * 50;
        }

        // Smooth severity with EMA
        smoothedSeverity = smoothedSeverity * (1 - CONFIG.SEVERITY_SMOOTHING) + severity * CONFIG.SEVERITY_SMOOTHING;
        pushSeverityHistory(smoothedSeverity);

        updateUI(isAsterixis, Math.round(smoothedSeverity), confidence);

    } catch (error) {
        logError('Detection failed', error);
    } finally {
        isDetecting = false;
    }
}

// ============================================================================
// Severity History Sparkline
// ============================================================================
function pushSeverityHistory(severity) {
    severityHistory.push(severity);
    if (severityHistory.length > CONFIG.SEVERITY_HISTORY_SIZE) {
        severityHistory.shift();
    }
    drawSeverityHistory();
}

function drawSeverityHistory() {
    const canvas = historyCanvas;
    const ctx = historyCtx;

    // Resize to container width
    canvas.width = canvas.parentElement.clientWidth || 300;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (severityHistory.length < 2) return;

    // Background grid lines at 25, 50, 75
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    for (const level of [25, 50, 75]) {
        const y = h - (level / 100) * h;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Draw severity line
    ctx.beginPath();
    ctx.lineWidth = 2;

    for (let i = 0; i < severityHistory.length; i++) {
        const x = (i / (CONFIG.SEVERITY_HISTORY_SIZE - 1)) * w;
        const y = h - (severityHistory[i] / 100) * h;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }

    // Gradient stroke color based on latest severity
    const latest = severityHistory[severityHistory.length - 1];
    if (latest < 25) ctx.strokeStyle = '#00e676';
    else if (latest < 50) ctx.strokeStyle = '#ffd600';
    else if (latest < 75) ctx.strokeStyle = '#ff9100';
    else ctx.strokeStyle = '#ff1744';

    ctx.stroke();
}

// ============================================================================
// UI Update
// ============================================================================
function updateUI(isAsterixis, severity, confidence) {
    document.querySelectorAll('.grade-item').forEach(el => el.classList.remove('active'));

    if (isAsterixis === null) {
        detectionResult.className = 'detection-result no-hand';
        detectionResult.querySelector('.result-label').textContent = 'No Hand';
        severityBar.style.width = '0%';
        severityValue.textContent = '--';
        return;
    }

    let gradeClass, gradeName;

    if (!isAsterixis || severity <= CONFIG.SEVERITY.NORMAL) {
        gradeClass = 'normal';
        gradeName = 'Normal';
        document.getElementById('grade-normal').classList.add('active');
    } else if (severity <= CONFIG.SEVERITY.GRADE1) {
        gradeClass = 'grade1 asterixis';
        gradeName = 'Grade 1';
        document.getElementById('grade-1').classList.add('active');
    } else if (severity <= CONFIG.SEVERITY.GRADE2) {
        gradeClass = 'grade2 asterixis';
        gradeName = 'Grade 2';
        document.getElementById('grade-2').classList.add('active');
    } else {
        gradeClass = 'grade3 asterixis';
        gradeName = 'Grade 3';
        document.getElementById('grade-3').classList.add('active');
    }

    detectionResult.className = `detection-result ${gradeClass}`;
    detectionResult.querySelector('.result-label').textContent = gradeName;

    severityBar.style.width = `${severity}%`;
    severityValue.textContent = severity;
}

// ============================================================================
// Event Listeners
// ============================================================================
startBtn.addEventListener('click', async () => {
    if (isRunning) await stopCamera();
    else await startCamera();
});

switchCameraBtn.addEventListener('click', switchCamera);

window.addEventListener('load', init);

document.addEventListener('visibilitychange', () => {
    if (document.hidden && isRunning) log('Page hidden, pausing...');
    else if (!document.hidden && isRunning) log('Page visible, resuming...');
});

window.addEventListener('beforeunload', () => { stopCamera(); });
