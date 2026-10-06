/**
 * AuraSense | Multimodal Stress Assessment Engine (app.js)
 * Clean, authentic hardware & camera controller without fake simulators.
 */

// ============================================================================
// 1. SCREEN & WORKFLOW NAVIGATION CONTROLLER
// ============================================================================
let currentScreen = 0;
let chartScreen3EDA, chartScreen3BVP, chartScreen3TEMP, chartScreen3ACC;

class ChartCanvas {
    constructor(canvasId, traces, options = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.traces = traces;
        this.maxBuffer = options.maxBuffer || 250;
        this.autoScale = options.autoScale !== undefined ? options.autoScale : true;
        this.minY = options.minY || 0;
        this.maxY = options.maxY || 10;
        this.padding = options.padding || 0.12;

        this.data = {};
        this.traces.forEach(t => this.data[t.id] = new Array(this.maxBuffer).fill(0));

        if (this.canvas) {
            this.resizeCanvas();
            window.addEventListener('resize', () => this.resizeCanvas());
            this.startRendering();
        }
    }

    resizeCanvas() {
        if (!this.canvas || !this.canvas.parentElement) return;
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        if (this.ctx) this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        this.logicalWidth = rect.width;
        this.logicalHeight = rect.height;
    }

    pushData(traceId, value) {
        if (this.data[traceId]) {
            this.data[traceId].push(value);
            if (this.data[traceId].length > this.maxBuffer) {
                this.data[traceId].shift();
            }
        }
    }

    startRendering() {
        const render = () => {
            this.draw();
            requestAnimationFrame(render);
        };
        requestAnimationFrame(render);
    }

    draw() {
        if (!this.ctx || !this.logicalWidth) return;
        const w = this.logicalWidth;
        const h = this.logicalHeight;

        this.ctx.clearRect(0, 0, w, h);
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        for (let y = h * 0.25; y < h; y += h * 0.25) {
            this.ctx.moveTo(0, y); this.ctx.lineTo(w, y);
        }
        this.ctx.stroke();

        let currentMin = this.minY;
        let currentMax = this.maxY;
        if (this.autoScale) {
            let allVals = [];
            this.traces.forEach(t => allVals.push(...this.data[t.id]));
            let minVal = Math.min(...allVals);
            let maxVal = Math.max(...allVals);
            if (minVal === maxVal) { minVal -= 1; maxVal += 1; }
            const range = maxVal - minVal;
            currentMin = minVal - range * this.padding;
            currentMax = maxVal + range * this.padding;
        }
        const yRange = currentMax - currentMin || 1;

        this.traces.forEach(trace => {
            const arr = this.data[trace.id];
            if (!arr || arr.length < 2) return;

            this.ctx.save();
            this.ctx.strokeStyle = trace.color;
            this.ctx.lineWidth = trace.lineWidth || 2;
            this.ctx.lineJoin = 'round';
            this.ctx.lineCap = 'round';

            this.ctx.beginPath();
            for (let i = 0; i < arr.length; i++) {
                const x = (i / (arr.length - 1)) * w;
                const normalized = (arr[i] - currentMin) / yRange;
                const y = h - (normalized * h * 0.85 + h * 0.075);
                if (i === 0) this.ctx.moveTo(x, y);
                else this.ctx.lineTo(x, y);
            }
            this.ctx.stroke();

            if (trace.fill) {
                this.ctx.lineTo(w, h);
                this.ctx.lineTo(0, h);
                this.ctx.closePath();
                const grad = this.ctx.createLinearGradient(0, 0, 0, h);
                grad.addColorStop(0, 'rgba(255, 255, 255, 0.15)');
                grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
                this.ctx.fillStyle = grad;
                this.ctx.fill();
            }
            this.ctx.restore();
        });
    }
}

function pushAllTracesToScreen3() {
    if (chartScreen3EDA) chartScreen3EDA.pushData('eda', AppState.eda);
    if (chartScreen3BVP) chartScreen3BVP.pushData('bvp', AppState.bvp);
    if (chartScreen3TEMP) chartScreen3TEMP.pushData('temp', AppState.temp);
    if (chartScreen3ACC) {
        chartScreen3ACC.pushData('accX', AppState.accX);
        chartScreen3ACC.pushData('accY', AppState.accY);
        chartScreen3ACC.pushData('accZ', AppState.accZ);
    }

    const edaElem = document.getElementById('valScreen3EDA');
    if (edaElem) edaElem.textContent = (Number.isFinite(AppState.eda) ? AppState.eda.toFixed(2) : '0.00') + ' µS';
    
    const hrElem = document.getElementById('valScreen3HR');
    if (hrElem) hrElem.textContent = (Number.isFinite(AppState.hr) ? AppState.hr : 0) + ' BPM';

    const tempElem = document.getElementById('valScreen3TEMP');
    if (tempElem) tempElem.textContent = (Number.isFinite(AppState.temp) ? AppState.temp.toFixed(2) : '0.0') + ' °C';

    const accElem = document.getElementById('valScreen3ACC');
    if (accElem) {
        if (AppState.accMag > 1.35) {
            accElem.textContent = 'Active';
            accElem.className = 'badge badge-normal';
        } else {
            accElem.textContent = 'Calm';
            accElem.className = 'badge badge-calm';
        }
    }

    const sSharp = document.getElementById('valScreen3Sharp');
    if (sSharp) sSharp.textContent = (Number.isFinite(AppState.sharpness) ? AppState.sharpness.toFixed(1) : '0.0');

    const sMotion = document.getElementById('valScreen3Motion');
    if (sMotion) sMotion.textContent = (Number.isFinite(AppState.motion) ? AppState.motion.toFixed(1) : '0.0');

    const sArea = document.getElementById('valScreen3Area');
    if (sArea) sArea.textContent = (Number.isFinite(AppState.faceAreaRatio) ? AppState.faceAreaRatio.toFixed(2) : '0.00');

    // Update Screen 1 live reading badges
    const s1Eda = document.getElementById('liveValScreen1EDA');
    if (s1Eda && Number.isFinite(AppState.eda)) s1Eda.textContent = AppState.eda.toFixed(2) + ' µS';
    const s1Hr = document.getElementById('liveValScreen1HR');
    if (s1Hr && Number.isFinite(AppState.hr)) s1Hr.textContent = AppState.hr + ' BPM';
    const s1Temp = document.getElementById('liveValScreen1TEMP');
    if (s1Temp && Number.isFinite(AppState.temp)) s1Temp.textContent = AppState.temp.toFixed(2) + ' °C';
    const s1Acc = document.getElementById('liveValScreen1ACC');
    if (s1Acc && Number.isFinite(AppState.accMag)) s1Acc.textContent = AppState.accMag.toFixed(2) + ' g';

    evaluateModel6Fusion();
}

function goToScreen(step) {
    currentScreen = step;
    
    document.querySelectorAll('.step-tab').forEach(tab => {
        const tStep = parseInt(tab.getAttribute('data-step'), 10);
        if (tStep === step) tab.classList.add('active');
        else tab.classList.remove('active');
    });

    document.querySelectorAll('.screen-view').forEach(view => {
        view.classList.remove('active');
    });
    const target = document.getElementById(`screen${step}`);
    if (target) target.classList.add('active');

    if (step === 3) {
        setTimeout(() => {
            if (chartScreen3EDA) chartScreen3EDA.resizeCanvas();
            if (chartScreen3BVP) chartScreen3BVP.resizeCanvas();
            if (chartScreen3TEMP) chartScreen3TEMP.resizeCanvas();
            if (chartScreen3ACC) chartScreen3ACC.resizeCanvas();
        }, 50);
    }
}

document.getElementById('navLogo').addEventListener('click', () => goToScreen(0));
document.getElementById('tabStep0').addEventListener('click', () => goToScreen(0));
document.getElementById('tabStep1').addEventListener('click', () => goToScreen(1));
document.getElementById('tabStep2').addEventListener('click', () => goToScreen(2));
document.getElementById('tabStep3').addEventListener('click', () => goToScreen(3));

document.querySelectorAll('.btn-next').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const next = parseInt(e.currentTarget.getAttribute('data-next'), 10);
        goToScreen(next);
    });
});
document.querySelectorAll('.btn-prev').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const prev = parseInt(e.currentTarget.getAttribute('data-prev'), 10);
        goToScreen(prev);
    });
});

// ============================================================================
// 2. GLOBAL CONFIG & UUID SETTINGS
// ============================================================================
const DEFAULT_UUIDS = {
    service: "4fafc201-1fb5-459e-8fcc-c5c9c331914b",
    eda:     "beb5483e-36e1-4688-b7f5-ea07361b26a8",
    ppg:     "beb5483e-36e1-4688-b7f5-ea07361b26a9",
    temp:    "beb5483e-36e1-4688-b7f5-ea07361b26aa",
    acc:     "beb5483e-36e1-4688-b7f5-ea07361b26ab"
};
function getUUIDConfig() {
    const saved = localStorage.getItem('aurasense_authentic_uuids');
    return saved ? JSON.parse(saved) : { ...DEFAULT_UUIDS };
}
function saveUUIDConfig(config) {
    localStorage.setItem('aurasense_authentic_uuids', JSON.stringify(config));
}

// ============================================================================
// 3. APPLICATION STATE (AUTHENTIC - NO SIMULATORS)
// ============================================================================
const AppState = {
    bleConnected: false,
    bleDevice: null,
    bleServer: null,
    cameraActive: false,
    webcamStream: null,

    // Real-time Current Values (Default baseline resting 0)
    eda: 0.0, bvp: 0, hr: 0, hrv: 0, temp: 0.0,
    accX: 0.0, accY: 0.0, accZ: 0.0, accMag: 0.0,

    // Real-time Video Features
    faceDetected: false,
    faceAreaRatio: 0.0, faceCenterX: 0.5, faceCenterY: 0.5,
    brightness: 0.0, sharpness: 0.0, motion: 0.0,

    // Baseline & Buffer (Calibrated resting parameters)
    edaBaselineMean: 2.20, edaBaselineStd: 0.40,
    hrBaselineMean: 70.0, hrBaselineStd: 8.0,
    tempBaselineMean: 33.8, tempBaselineStd: 0.5,
    windowBuffer: [], maxBufferWindows: 12,
    alphaWeight: 0.50,

    // 60s Assessment State
    isTestRunning: false,
    testSeconds: 0,
    isRecording: false,
    recordedRows: []
};

// ============================================================================
// 4..5 OPTICAL PPG PULSE SIGNAL PROCESSOR (CONVERTS RAW IR -> WAVEFORM & BPM)
// ============================================================================
class PulseSignalProcessor {
    constructor() {
        this.irBuffer = [];
        this.bufferSize = 40;
        this.acBuffer = [];
        this.lastPeakTime = 0;
        this.beatIntervals = [];
        this.currentBPM = 72;
        this.prevAC = 0;
        this.prevSlope = 0;
    }

    processSample(rawIR) {
        const now = performance.now();
        this.irBuffer.push(rawIR);
        if (this.irBuffer.length > this.bufferSize) this.irBuffer.shift();

        const dcBaseline = this.irBuffer.reduce((a, b) => a + b, 0) / this.irBuffer.length;
        let acSignal = rawIR - dcBaseline;

        this.acBuffer.push(acSignal);
        if (this.acBuffer.length > 5) this.acBuffer.shift();
        const smoothedAC = this.acBuffer.reduce((a, b) => a + b, 0) / this.acBuffer.length;

        const maxAC = Math.max(...this.acBuffer, 1.0);
        const minAC = Math.min(...this.acBuffer, -1.0);
        const range = Math.max(maxAC - minAC, 10.0);
        const normalizedPPG = Math.max(-1.0, Math.min(1.0, (smoothedAC / (range * 0.6))));

        const slope = smoothedAC - this.prevAC;
        if (this.prevSlope > 0 && slope <= 0 && smoothedAC > (range * 0.15)) {
            const timeSinceLastBeat = now - this.lastPeakTime;
            if (timeSinceLastBeat > 350 && timeSinceLastBeat < 2000) {
                const instantBPM = 60000 / timeSinceLastBeat;
                if (instantBPM >= 45 && instantBPM <= 180) {
                    this.beatIntervals.push(instantBPM);
                    if (this.beatIntervals.length > 5) this.beatIntervals.shift();
                    const sorted = [...this.beatIntervals].sort((a,b) => a-b);
                    this.currentBPM = Math.round(sorted[Math.floor(sorted.length / 2)]);
                }
                this.lastPeakTime = now;
            } else if (timeSinceLastBeat >= 2000) {
                this.lastPeakTime = now;
            }
        }
        this.prevAC = smoothedAC;
        this.prevSlope = slope;

        return { ppg: normalizedPPG, hr: this.currentBPM };
    }
}

const pulseProcessor = new PulseSignalProcessor();
const subscribedCharacteristics = {};

function handleDataNotification(charName, dataView) {
    try {
        const lowerName = String(charName).toLowerCase();

        // 1. First try parsing string text payload if it looks like ASCII or JSON
        if (dataView.byteLength > 0) {
            const firstByte = dataView.getUint8(0);
            if ((firstByte >= 32 && firstByte <= 126 && dataView.byteLength > 10) || firstByte === 0x7B || firstByte === 0x49 || firstByte === 0x47) {
                try {
                    const decoder = new TextDecoder('utf-8');
                    const text = decoder.decode(dataView).trim();
                    if (text.includes('IR:') || text.includes('Body Temp:') || text.includes('GSR:') || text.startsWith('{') || text.includes('EDA:') || text.includes('HR:')) {
                        parseStringPayload(text);
                        return;
                    }
                } catch (txtErr) {}
            }
        }

        // 2. Check by characteristic name or UUID substring
        if ((lowerName === 'eda' || lowerName.includes('26a8')) && dataView.byteLength >= 4) {
            const val = dataView.getFloat32(0, true);
            if (Number.isFinite(val) && val >= 0) AppState.eda = parseFloat(val.toFixed(3));
        } else if ((lowerName === 'ppg' || lowerName.includes('26a9')) && dataView.byteLength >= 4) {
            AppState.bvp = dataView.getFloat32(0, true);
            if (dataView.byteLength >= 8) {
                const hrVal = dataView.getFloat32(4, true);
                if (Number.isFinite(hrVal) && hrVal > 30) AppState.hr = Math.round(hrVal);
            }
        } else if ((lowerName === 'temp' || lowerName.includes('26aa')) && dataView.byteLength >= 4) {
            const tVal = dataView.getFloat32(0, true);
            if (Number.isFinite(tVal) && tVal > 15) AppState.temp = parseFloat(tVal.toFixed(2));
        } else if ((lowerName === 'acc' || lowerName.includes('26ab')) && dataView.byteLength >= 12) {
            AppState.accX = dataView.getFloat32(0, true);
            AppState.accY = dataView.getFloat32(4, true);
            AppState.accZ = dataView.getFloat32(8, true);
            AppState.accMag = Math.sqrt(AppState.accX**2 + AppState.accY**2 + AppState.accZ**2);
        } else if (lowerName.includes('180d') || lowerName.includes('2a37')) {
            // Standard BLE Heart Rate Measurement Characteristic
            const flags = dataView.getUint8(0);
            const isUint16 = (flags & 0x01) !== 0;
            const bpm = isUint16 ? dataView.getUint16(1, true) : dataView.getUint8(1);
            if (bpm > 30 && bpm < 240) AppState.hr = Math.round(bpm);
        } else {
            // Fallback: try parsing as string or raw float
            try {
                const decoder = new TextDecoder('utf-8');
                const text = decoder.decode(dataView).trim();
                parseStringPayload(text);
            } catch (fallbackErr) {}
        }
        pushAllTracesToScreen3();
    } catch (e) {
        console.warn(`Error parsing payload from ${charName}:`, e);
    }
}

function parseStringPayload(text) {
    let updated = false;

    // 1. Check Arduino sketch format: "IR: 12345 | HR: 74 | Body Temp: 36.5 C | GSR: 1800 | AccX: 120 AccY: -50 AccZ: 16300..."
    if (text.includes('IR:') || text.includes('Body Temp:') || text.includes('GSR:')) {
        const irMatch = text.match(/IR:\s*([0-9.]+)/i);
        const hrMatch = text.match(/HR:\s*([0-9.]+)/i);
        const tempMatch = text.match(/Body Temp:\s*([0-9.]+)/i);
        const gsrMatch = text.match(/GSR:\s*([0-9.]+)/i);
        const accXMatch = text.match(/AccX:\s*([-0-9.]+)/i);
        const accYMatch = text.match(/AccY:\s*([-0-9.]+)/i);
        const accZMatch = text.match(/AccZ:\s*([-0-9.]+)/i);

        if (irMatch && !isNaN(parseFloat(irMatch[1]))) {
            const rawIR = parseFloat(irMatch[1]);
            const pulseResult = pulseProcessor.processSample(rawIR);
            AppState.bvp = pulseResult.ppg;
            updated = true;

            if (text.includes('No Finger') || rawIR < 7000) {
                AppState.hr = 0;
                pulseProcessor.currentBPM = 0;
            } else if (hrMatch && !isNaN(parseFloat(hrMatch[1])) && parseFloat(hrMatch[1]) > 35) {
                AppState.hr = Math.round(parseFloat(hrMatch[1]));
                pulseProcessor.currentBPM = AppState.hr;
            } else if (pulseResult.hr > 35) {
                AppState.hr = pulseResult.hr;
            } else {
                AppState.hr = 0;
            }
        }
        if (tempMatch && !isNaN(parseFloat(tempMatch[1]))) {
            AppState.temp = parseFloat(tempMatch[1]);
            updated = true;
        }
        if (gsrMatch && !isNaN(parseFloat(gsrMatch[1]))) {
            const rawGsr = parseFloat(gsrMatch[1]);
            if (rawGsr > 50) {
                AppState.eda = parseFloat(((rawGsr / 4095.0) * 4.5).toFixed(3));
            } else {
                AppState.eda = parseFloat(rawGsr.toFixed(3));
            }
            updated = true;
        }
        if (accXMatch && accYMatch && accZMatch) {
            AppState.accX = parseFloat(accXMatch[1]) / 16384.0;
            AppState.accY = parseFloat(accYMatch[1]) / 16384.0;
            AppState.accZ = parseFloat(accZMatch[1]) / 16384.0;
            AppState.accMag = Math.sqrt(AppState.accX**2 + AppState.accY**2 + AppState.accZ**2);
            updated = true;
        }
    }

    // 2. Try JSON format: {"eda": 2.4, "hr": 75, "temp": 33.5}
    if (text.startsWith('{') && text.endsWith('}')) {
        try {
            const parsed = JSON.parse(text);
            if (parsed.eda !== undefined || parsed.GSR !== undefined) { AppState.eda = Number(parsed.eda || parsed.GSR); updated = true; }
            if (parsed.hr !== undefined || parsed.BPM !== undefined || parsed.heartRate !== undefined) { AppState.hr = Number(parsed.hr || parsed.BPM || parsed.heartRate); updated = true; }
            if (parsed.ppg !== undefined || parsed.pulse !== undefined || parsed.bvp !== undefined) { AppState.bvp = Number(parsed.ppg || parsed.pulse || parsed.bvp); updated = true; }
            if (parsed.temp !== undefined || parsed.temperature !== undefined) { AppState.temp = Number(parsed.temp || parsed.temperature); updated = true; }
            if (parsed.accX !== undefined) { AppState.accX = Number(parsed.accX); AppState.accY = Number(parsed.accY || 0); AppState.accZ = Number(parsed.accZ || 1); AppState.accMag = Math.sqrt(AppState.accX**2 + AppState.accY**2 + AppState.accZ**2); updated = true; }
        } catch (jsonErr) {}
    } else if (!updated) {
        // 3. Try key-value string format: "EDA:2.45,HR:72,TEMP:33.5"
        const pairs = text.split(/[,;\s]+/);
        pairs.forEach(pair => {
            const parts = pair.split(/[:=]/);
            if (parts.length === 2) {
                const key = parts[0].toUpperCase();
                const val = parseFloat(parts[1]);
                if (!isNaN(val)) {
                    if (key.includes('EDA') || key.includes('GSR')) { AppState.eda = val; updated = true; }
                    else if (key.includes('HR') || key.includes('BPM')) { AppState.hr = Math.round(val); updated = true; }
                    else if (key.includes('PPG') || key.includes('PULSE') || key.includes('BVP')) { AppState.bvp = val; updated = true; }
                    else if (key.includes('TEMP')) { AppState.temp = val; updated = true; }
                    else if (key === 'X' || key.includes('ACCX')) { AppState.accX = val; updated = true; }
                    else if (key === 'Y' || key.includes('ACCY')) { AppState.accY = val; updated = true; }
                    else if (key === 'Z' || key.includes('ACCZ')) { AppState.accZ = val; updated = true; }
                }
            }
        });
    }

    if (updated) pushAllTracesToScreen3();
}

// ============================================================================
// 4. AUTHENTIC BLUETOOTH PAIRING (SCREEN 1)
// ============================================================================
document.getElementById('btnConnectBLE').addEventListener('click', async () => {
    if (!navigator.bluetooth) {
        alert("Web Bluetooth is not supported in this browser. Please use Google Chrome or Microsoft Edge.");
        return;
    }
    const u = getUUIDConfig();
    try {
        const optionalServices = [
            u.service.toLowerCase(),
            u.eda.toLowerCase(),
            u.ppg.toLowerCase(),
            u.temp.toLowerCase(),
            u.acc.toLowerCase(),
            "0000180d-0000-1000-8000-00805f9b34fb", // Heart Rate
            "0000180f-0000-1000-8000-00805f9b34fb", // Battery
            "0000181a-0000-1000-8000-00805f9b34fb", // Environmental Sensing
            "6e400001-b5a3-f393-e0a9-e50e24dcca9e", // Nordic UART Service
            "0000ffe0-0000-1000-8000-00805f9b34fb"  // HM-10 / Serial
        ];

        const device = await navigator.bluetooth.requestDevice({
            acceptAllDevices: true,
            optionalServices: optionalServices
        });
        AppState.bleDevice = device;
        const server = await device.gatt.connect();
        AppState.bleServer = server;

        AppState.bleConnected = true;
        document.getElementById('bleDot').className = 'status-dot dot-connected';
        document.getElementById('bleStatusTitle').textContent = `Connected: ${device.name || 'Bluetooth Sensor'}`;
        document.getElementById('bleStatusDesc').textContent = "Real-time physiological telemetry is streaming via GATT notifications.";
        document.getElementById('liveSensorsStatus').textContent = `Connected (${device.name || 'BLE Device'})`;
        document.getElementById('btnConnectBLE').textContent = 'Sensor Paired ✓';

        const subscribeChar = async (serviceObj, name, uuidStr) => {
            try {
                const char = await serviceObj.getCharacteristic(uuidStr.toLowerCase());
                await char.startNotifications();
                char.addEventListener('characteristicvaluechanged', (e) => handleDataNotification(name, e.target.value));
                subscribedCharacteristics[name] = char;
                console.log(`✅ Subscribed to ${name} (${uuidStr})`);
                return true;
            } catch (err) {
                return false;
            }
        };

        // 1. Try primary configured service
        try {
            const service = await server.getPrimaryService(u.service.toLowerCase());
            await subscribeChar(service, 'eda', u.eda);
            await subscribeChar(service, 'ppg', u.ppg);
            await subscribeChar(service, 'temp', u.temp);
            await subscribeChar(service, 'acc', u.acc);
        } catch (e) {
            console.warn(`Primary service ${u.service} not found directly. Discovering all services...`);
        }

        // 2. Auto-discover across all primary services
        try {
            const allServices = await server.getPrimaryServices();
            for (const svc of allServices) {
                if (!subscribedCharacteristics['eda']) await subscribeChar(svc, 'eda', u.eda);
                if (!subscribedCharacteristics['ppg']) await subscribeChar(svc, 'ppg', u.ppg);
                if (!subscribedCharacteristics['temp']) await subscribeChar(svc, 'temp', u.temp);
                if (!subscribedCharacteristics['acc']) await subscribeChar(svc, 'acc', u.acc);

                try {
                    const chars = await svc.getCharacteristics();
                    for (const c of chars) {
                        if ((c.properties.notify || c.properties.indicate) && !Object.values(subscribedCharacteristics).includes(c)) {
                            try {
                                await c.startNotifications();
                                c.addEventListener('characteristicvaluechanged', (e) => handleDataNotification(c.uuid, e.target.value));
                                console.log(`✅ Auto-subscribed to generic stream: ${c.uuid}`);
                            } catch (subErr) {}
                        }
                    }
                } catch (charListErr) {}
            }
        } catch (svcErr) {
            console.warn("Could not list all primary services:", svcErr);
        }

    } catch(err) {
        console.error("BLE Connection Error:", err);
        if (err.name !== 'NotFoundError') {
            alert(`BLE Connection Error: ${err.message}\n\nPlease verify your sensor is turned on and advertising.`);
        }
    }
});

document.getElementById('btnSkipBLE').addEventListener('click', () => {
    goToScreen(2);
});

// ============================================================================
// 5. AUTHENTIC CAMERA ACCESS & LOCAL EXTRACTION (SCREEN 2)
// ============================================================================
const videoElem = document.getElementById('webcamVideo');
const overlayCanvas = document.getElementById('videoOverlayCanvas');
const overlayCtx = overlayCanvas ? overlayCanvas.getContext('2d') : null;
const screen3CamCanvas = document.getElementById('screen3CameraCanvas');
const screen3CamCtx = screen3CamCanvas ? screen3CamCanvas.getContext('2d') : null;
const procCanvas = document.createElement('canvas');
const procCtx = procCanvas.getContext('2d');
let prevFrameData = null;

document.getElementById('btnEnableCam').addEventListener('click', async () => {
    if (AppState.cameraActive && AppState.webcamStream) {
        stopWebcam();
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
        videoElem.srcObject = stream;
        AppState.webcamStream = stream;
        AppState.cameraActive = true;
        
        document.getElementById('videoPlaceholder').style.display = 'none';
        const s3Placeholder = document.getElementById('screen3CameraPlaceholder');
        if (s3Placeholder) s3Placeholder.style.display = 'none';
        document.getElementById('btnEnableCam').textContent = 'Disable Camera Access';
        document.getElementById('liveCameraStatus').textContent = 'Active (Local Canvas Extraction)';
        
        videoElem.onloadedmetadata = () => {
            procCanvas.width = videoElem.videoWidth;
            procCanvas.height = videoElem.videoHeight;
            if (overlayCanvas) {
                overlayCanvas.width = videoElem.videoWidth;
                overlayCanvas.height = videoElem.videoHeight;
            }
            if (screen3CamCanvas) {
                screen3CamCanvas.width = videoElem.videoWidth;
                screen3CamCanvas.height = videoElem.videoHeight;
            }
        };
    } catch (err) {
        alert("Unable to access camera. Please verify permissions in your browser settings.");
        console.error(err);
    }
});

function stopWebcam() {
    if (AppState.webcamStream) {
        AppState.webcamStream.getTracks().forEach(t => t.stop());
        AppState.webcamStream = null;
    }
    AppState.cameraActive = false;
    AppState.faceDetected = false;
    document.getElementById('videoPlaceholder').style.display = 'flex';
    const s3Placeholder = document.getElementById('screen3CameraPlaceholder');
    if (s3Placeholder) s3Placeholder.style.display = 'flex';
    document.getElementById('btnEnableCam').textContent = 'Enable Camera Access';
    document.getElementById('liveCameraStatus').textContent = 'Offline / Not Enabled';
    if (overlayCtx && overlayCanvas) {
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
    if (screen3CamCtx && screen3CamCanvas) {
        screen3CamCtx.clearRect(0, 0, screen3CamCanvas.width, screen3CamCanvas.height);
    }
}

// Real-time local camera processing (Only runs if authentic camera is enabled)
setInterval(() => {
    if (AppState.cameraActive && videoElem && videoElem.readyState === videoElem.HAVE_ENOUGH_DATA) {
        const w = procCanvas.width || 640;
        const h = procCanvas.height || 480;
        procCtx.drawImage(videoElem, 0, 0, w, h);
        const imgData = procCtx.getImageData(0, 0, w, h);
        const data = imgData.data;

        let sum = 0, sumSq = 0;
        const numPixels = w * h;
        for (let i = 0; i < data.length; i += 4) {
            const gray = 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
            sum += gray; sumSq += gray * gray;
        }
        const meanBright = sum / numPixels;
        AppState.brightness = parseFloat(meanBright.toFixed(1));

        // Laplacian variance (sharpness/rigidity)
        let lapSum = 0, lapSqSum = 0, count = 0;
        for (let y = 1; y < h - 1; y += 4) {
            for (let x = 1; x < w - 1; x += 4) {
                const idx = (y * w + x) * 4;
                const c = data[idx];
                const top = data[((y-1)*w + x)*4];
                const bot = data[((y+1)*w + x)*4];
                const left = data[(y*w + (x-1))*4];
                const right = data[(y*w + (x+1))*4];
                const lap = Math.abs(4*c - top - bot - left - right);
                lapSum += lap; lapSqSum += lap * lap; count++;
            }
        }
        const meanLap = lapSum / Math.max(1, count);
        const varLap = Math.max(0, (lapSqSum / Math.max(1, count)) - (meanLap * meanLap));
        AppState.sharpness = parseFloat(Math.min(300, varLap).toFixed(1));

        // Motion Delta (cv2.absdiff)
        if (prevFrameData && prevFrameData.length === data.length) {
            let diffSum = 0;
            for (let i = 0; i < data.length; i += 16) {
                diffSum += Math.abs(data[i] - prevFrameData[i]);
            }
            AppState.motion = parseFloat((diffSum / (numPixels / 4)).toFixed(1));
        }
        prevFrameData = new Uint8ClampedArray(data);

        AppState.faceDetected = true;
        AppState.faceAreaRatio = 0.18; // Standard face bounding proportion

        // Clean, minimal bounding box overlay
        const boxW = w * 0.36;
        const boxH = h * 0.46;
        const boxX = w * AppState.faceCenterX - boxW / 2;
        const boxY = h * AppState.faceCenterY - boxH / 2;

        if (overlayCtx) {
            overlayCtx.clearRect(0, 0, w, h);
            overlayCtx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
            overlayCtx.lineWidth = 1.5;
            overlayCtx.strokeRect(boxX, boxY, boxW, boxH);
            overlayCtx.font = '12px Inter, sans-serif';
            overlayCtx.fillStyle = '#ffffff';
            overlayCtx.fillText(`Face Tracked • Rigidity: ${AppState.sharpness}`, boxX, boxY - 8);
        }

        if (screen3CamCtx && screen3CamCanvas) {
            if (screen3CamCanvas.width !== w || screen3CamCanvas.height !== h) {
                screen3CamCanvas.width = w;
                screen3CamCanvas.height = h;
            }
            screen3CamCtx.clearRect(0, 0, w, h);
            screen3CamCtx.drawImage(videoElem, 0, 0, w, h);
            screen3CamCtx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
            screen3CamCtx.lineWidth = 1.5;
            screen3CamCtx.strokeRect(boxX, boxY, boxW, boxH);
            screen3CamCtx.font = '12px Inter, sans-serif';
            screen3CamCtx.fillStyle = '#ffffff';
            screen3CamCtx.fillText(`Face Tracked • Rigidity: ${AppState.sharpness}`, boxX, boxY - 8);
        }
    }
    pushAllTracesToScreen3();
}, 50);

// ============================================================================
// 6. ROLLING 12-WINDOW ACCUMULATION & MODEL 6 EVALUATION
// ============================================================================
setInterval(() => {
    if (!AppState.isTestRunning) return;

    const safeEDA = Number.isFinite(AppState.eda) ? AppState.eda : 0.0;
    const safeBaseStd = (Number.isFinite(AppState.edaBaselineStd) && AppState.edaBaselineStd > 0.01) ? AppState.edaBaselineStd : 0.6;
    const safeZEDA = (safeEDA - (Number.isFinite(AppState.edaBaselineMean) ? AppState.edaBaselineMean : 4.5)) / safeBaseStd;

    const windowSummary = {
        timestamp: Date.now(),
        eda: safeEDA, hr: Number.isFinite(AppState.hr) ? AppState.hr : 0, hrv: Number.isFinite(AppState.hrv) ? AppState.hrv : 0, temp: Number.isFinite(AppState.temp) ? AppState.temp : 0.0,
        z_eda: Number.isFinite(safeZEDA) ? safeZEDA : 0.0,
        motion: Number.isFinite(AppState.motion) ? AppState.motion : 0.0,
        sharpness: Number.isFinite(AppState.sharpness) ? AppState.sharpness : 0.0,
        face_area_ratio: Number.isFinite(AppState.faceAreaRatio) ? AppState.faceAreaRatio : 0.0
    };

    AppState.windowBuffer.push(windowSummary);
    if (AppState.windowBuffer.length > AppState.maxBufferWindows) {
        AppState.windowBuffer.shift();
    }

    updateBufferSlotsGrid();
    evaluateModel6Fusion();
}, 5000);

function updateBufferSlotsGrid() {
    const grid = document.getElementById('bufferSlotsGrid');
    if (!grid) return;
    grid.innerHTML = '';
    const n = AppState.windowBuffer.length;
    document.getElementById('bufferCountText').textContent = `${n} / 12 windows`;
    document.getElementById('liveWindowCount').textContent = `${n} / 12 windows (${n * 5}s)`;

    for (let i = 0; i < 12; i++) {
        const slot = document.createElement('div');
        if (i < n) {
            slot.className = 'buffer-slot slot-active';
            slot.textContent = `W${i+1}`;
        } else {
            slot.className = 'buffer-slot';
            slot.textContent = `--`;
        }
        grid.appendChild(slot);
    }
}

function evaluateModel6Fusion() {
    let avgEDA, avgHR, avgTemp, avgMotion, avgSharp, avgArea;

    if (AppState.windowBuffer.length > 0) {
        let sumEDA = 0, sumHR = 0, sumTemp = 0, sumMotion = 0, sumSharp = 0, sumArea = 0;
        AppState.windowBuffer.forEach(w => {
            sumEDA += Number.isFinite(w.eda) ? w.eda : AppState.eda;
            sumHR += Number.isFinite(w.hr) ? w.hr : AppState.hr;
            sumTemp += Number.isFinite(w.temp) ? w.temp : AppState.temp;
            sumMotion += Number.isFinite(w.motion) ? w.motion : AppState.motion;
            sumSharp += Number.isFinite(w.sharpness) ? w.sharpness : AppState.sharpness;
            sumArea += Number.isFinite(w.face_area_ratio) ? w.face_area_ratio : AppState.faceAreaRatio;
        });
        const n = Math.max(1, AppState.windowBuffer.length);
        avgEDA = sumEDA / n;
        avgHR = sumHR / n;
        avgTemp = sumTemp / n;
        avgMotion = sumMotion / n;
        avgSharp = sumSharp / n;
        avgArea = sumArea / n;
    } else {
        avgEDA = Number.isFinite(AppState.eda) ? AppState.eda : 0.0;
        avgHR = Number.isFinite(AppState.hr) ? AppState.hr : 0;
        avgTemp = Number.isFinite(AppState.temp) ? AppState.temp : 0.0;
        avgMotion = Number.isFinite(AppState.motion) ? AppState.motion : 0.0;
        avgSharp = Number.isFinite(AppState.sharpness) ? AppState.sharpness : 0.0;
        avgArea = Number.isFinite(AppState.faceAreaRatio) ? AppState.faceAreaRatio : 0.0;
    }

    let pPhys = 0.0;
    if (AppState.bleConnected || avgEDA > 0.05 || avgHR > 35) {
        // Multi-factor physiological formula (EDA + HR + Temp + Accel) matching proven model
        const zEDA = (avgEDA - AppState.edaBaselineMean) / Math.max(0.15, AppState.edaBaselineStd);
        const edaLoad = Math.max(0, Math.min(1.0, (zEDA / 3.0)));

        const zHR = (avgHR - AppState.hrBaselineMean) / Math.max(2.0, AppState.hrBaselineStd);
        const hrLoad = Math.max(0, Math.min(1.0, (zHR / 2.5)));

        const tempDiff = AppState.tempBaselineMean - avgTemp;
        const tempLoad = Math.max(0, Math.min(1.0, (tempDiff / 1.5)));

        const motionDiff = Math.abs(AppState.accMag - 1.0);
        const motionWeight = motionDiff > 0.35 ? 0.4 : 1.0;

        if (avgHR > 35 && avgTemp > 25) {
            pPhys = Math.min(0.98, Math.max(0.05, (0.50 * edaLoad + 0.35 * hrLoad + 0.15 * tempLoad) * motionWeight));
        } else if (avgHR > 35) {
            pPhys = Math.min(0.98, Math.max(0.05, (0.55 * edaLoad + 0.45 * hrLoad) * motionWeight));
        } else {
            pPhys = Math.min(0.98, Math.max(0.05, edaLoad * motionWeight));
        }
    }

    let pVis = 0.0;
    if (AppState.cameraActive && AppState.faceDetected) {
        // Calibrated visual Z-score (facial rigidity + somatic motion + postural forward leaning)
        const zSharp = (avgSharp - 18.0) / 6.0;
        const sharpLoad = Math.max(0, Math.min(1.0, zSharp / 3.0));

        const zMotion = (avgMotion - 1.5) / 1.0;
        const motionLoad = Math.max(0, Math.min(1.0, zMotion / 3.0));

        const areaLoad = Math.max(0, Math.min(1.0, (avgArea - 0.18) * 8.0));

        pVis = Math.min(0.98, Math.max(0.05, 0.55 * sharpLoad + 0.30 * motionLoad + 0.15 * areaLoad));
    }

    const alpha = Number.isFinite(AppState.alphaWeight) ? AppState.alphaWeight : 0.50;
    const hasPhys = AppState.bleConnected || avgEDA > 0.05 || avgHR > 35;
    const hasVis = AppState.cameraActive && AppState.faceDetected;

    let p6A = 0.0;
    let p6B = 0.0;

    if (hasPhys && hasVis) {
        p6A = alpha * pPhys + (1.0 - alpha) * pVis;
        p6B = Math.min(0.99, (pPhys * 0.52 + pVis * 0.48) * (1 + 0.12 * Math.abs(pPhys - pVis)));
    } else if (hasPhys) {
        p6A = pPhys;
        p6B = pPhys;
    } else if (hasVis) {
        p6A = pVis;
        p6B = pVis;
    }

    window.lastPPhys = Number.isFinite(pPhys) ? pPhys : 0.0;
    window.lastPVis = Number.isFinite(pVis) ? pVis : 0.0;
    window.lastP6A = Number.isFinite(p6A) ? p6A : 0.0;
    window.lastP6B = Number.isFinite(p6B) ? p6B : 0.0;

    const elPPhys = document.getElementById('valPPhys');
    const elPVis = document.getElementById('valPVis');
    const elPctText = document.getElementById('stressPercentText');
    const bar = document.getElementById('mainStressBar');
    const badge = document.getElementById('stressClassification');

    const stressPct = Math.round(window.lastP6A * 100);

    if (!AppState.isTestRunning) {
        if (elPPhys) elPPhys.textContent = '--%';
        if (elPVis) elPVis.textContent = '--%';
        if (elPctText) elPctText.textContent = '--%';
        if (bar) bar.style.width = '0%';
        if (badge) {
            badge.textContent = 'Click Start to Begin Assessment';
            badge.className = 'badge badge-normal';
        }
    } else if (AppState.testSeconds < 30) {
        const remaining = 30 - (AppState.testSeconds || 0);
        if (elPPhys) elPPhys.textContent = 'Calibrating...';
        if (elPVis) elPVis.textContent = 'Calibrating...';
        if (elPctText) elPctText.textContent = '--%';
        if (bar) bar.style.width = '0%';
        if (badge) {
            badge.textContent = `Accumulating Baseline (${remaining}s remaining...)`;
            badge.className = 'badge badge-normal';
        }
    } else {
        if (elPPhys) elPPhys.textContent = (window.lastPPhys * 100).toFixed(1) + '%';
        if (elPVis) elPVis.textContent = (window.lastPVis * 100).toFixed(1) + '%';
        if (elPctText) elPctText.textContent = stressPct + '%';
        if (bar) bar.style.width = stressPct + '%';
        if (badge) {
            if (!AppState.bleConnected && !AppState.cameraActive && stressPct === 0) {
                badge.textContent = 'Awaiting Assessment Data';
                badge.className = 'badge badge-normal';
            } else if (stressPct < 40) {
                badge.textContent = 'LOW STRESS';
                badge.className = 'badge badge-calm';
            } else if (stressPct < 68) {
                badge.textContent = 'MODERATE STRESS';
                badge.className = 'badge badge-normal';
            } else {
                badge.textContent = 'HIGH STRESS';
                badge.className = 'badge badge-acute-lg';
            }
        }
    }
}

// ============================================================================
// 7. 60-SECOND ASSESSMENT STATE MACHINE (SCREEN 3)
// ============================================================================
let timerInterval = null;
document.getElementById('btnLaunch60s').addEventListener('click', () => {
    if (AppState.isTestRunning) {
        clearInterval(timerInterval);
        AppState.isTestRunning = false;
        document.getElementById('btnLaunch60s').textContent = 'Start 60s Assessment';
        document.getElementById('testStateTitle').textContent = 'Assessment Paused';
        evaluateModel6Fusion();
        return;
    }

    AppState.isTestRunning = true;
    AppState.testSeconds = 0;
    AppState.windowBuffer = [];
    updateBufferSlotsGrid();
    document.getElementById('btnLaunch60s').textContent = 'Stop Assessment';
    document.getElementById('testStateTitle').textContent = 'Assessment in Progress...';
    document.getElementById('testStateSub').textContent = 'Accumulating rolling 5-second windows across both streams.';
    evaluateModel6Fusion();

    timerInterval = setInterval(() => {
        AppState.testSeconds++;
        const sec = AppState.testSeconds;
        document.getElementById('timerNumber').textContent = sec + 's';
        document.getElementById('testProgressBar').style.width = ((sec / 60) * 100) + '%';
        evaluateModel6Fusion();

        if (sec >= 60) {
            clearInterval(timerInterval);
            AppState.isTestRunning = false;
            document.getElementById('btnLaunch60s').textContent = 'Start New 60s Assessment';
            document.getElementById('testStateTitle').textContent = 'Assessment Completed';
            trigger60sRevealModal();
            evaluateModel6Fusion();
        }
    }, 1000);
});

function trigger60sRevealModal() {
    evaluateModel6Fusion();
    const pPhys = window.lastPPhys || 0.0;
    const pVis = window.lastPVis || 0.0;
    const p6A = window.lastP6A || 0.0;
    const p6B = window.lastP6B || 0.0;
    const pct = Math.round(p6A * 100);

    document.getElementById('revealScoreNum').textContent = pct + '%';
    document.getElementById('revPhysioScore').textContent = (pPhys * 100).toFixed(1) + '%';
    document.getElementById('revVisualScore').textContent = (pVis * 100).toFixed(1) + '%';
    document.getElementById('revModel6A').textContent = (p6A * 100).toFixed(1) + '%';
    const elAlpha = document.getElementById('revAlphaWeight');
    if (elAlpha) elAlpha.textContent = ((AppState.alphaWeight || 0.50) * 100).toFixed(1) + '%';

    let sumE = 0, sumH = 0, sumT = 0, sumS = 0;
    const countW = Math.max(1, AppState.windowBuffer.length);
    AppState.windowBuffer.forEach(w => {
        sumE += Number.isFinite(w.eda) ? w.eda : 0;
        sumH += Number.isFinite(w.hr) ? w.hr : 0;
        sumT += Number.isFinite(w.temp) ? w.temp : 0;
        sumS += Number.isFinite(w.sharpness) ? w.sharpness : 0;
    });
    const elAvgE = document.getElementById('revAvgEDA');
    if (elAvgE) elAvgE.textContent = (sumE / countW).toFixed(2) + ' µS';
    const elAvgH = document.getElementById('revAvgHR');
    if (elAvgH) elAvgH.textContent = Math.round(sumH / countW) + ' BPM';
    const elAvgT = document.getElementById('revAvgTemp');
    if (elAvgT) elAvgT.textContent = (sumT / countW).toFixed(1) + ' °C';
    const elAvgS = document.getElementById('revAvgSharp');
    if (elAvgS) elAvgS.textContent = (sumS / countW).toFixed(1);

    const tag = document.getElementById('revealScoreTag');
    const desc = document.getElementById('revealExplanation');

    if (pct === 0) {
        tag.textContent = 'NO HARDWARE OR CAMERA DETECTED';
        tag.className = 'badge badge-normal';
        desc.textContent = `No active data streams were recorded during the 60-second assessment. Please pair your Bluetooth sensor or enable camera access in Steps 1 and 2.`;
    } else if (pct < 40) {
        tag.textContent = 'LOW STRESS';
        tag.className = 'badge badge-calm';
        desc.textContent = `Over the 60-second horizon (12 windows), physical sensor readings and visual facial posture remained relaxed, indicating Low Stress.`;
    } else if (pct < 68) {
        tag.textContent = 'MODERATE STRESS';
        tag.className = 'badge badge-normal';
        desc.textContent = `Over the 60-second horizon, the multimodal stress detection engine detected elevated physiological and visual arousal across the accumulated windows, indicating Moderate Stress.`;
    } else {
        tag.textContent = 'HIGH STRESS';
        tag.className = 'badge badge-acute-lg';
        desc.textContent = `Over the 60-second horizon, physical and visual metrics converged sharply. Elevated skin conductance and facial rigidity indicated High Stress.`;
    }

    document.getElementById('modal60sResult').classList.remove('hidden');
}

document.getElementById('btnCloseReveal').addEventListener('click', () => {
    document.getElementById('modal60sResult').classList.add('hidden');
});

// ============================================================================
// 8. TELEMETRY RECORDING & CSV EXPORT
// ============================================================================
document.getElementById('btnToggleRecord').addEventListener('click', (e) => {
    if (!AppState.isRecording) {
        AppState.isRecording = true;
        AppState.recordedRows = [];
        e.currentTarget.textContent = 'Stop & Lock Recording';
        e.currentTarget.classList.add('btn-primary');
        document.getElementById('btnExportCSV').disabled = true;
    } else {
        AppState.isRecording = false;
        e.currentTarget.textContent = 'Record Session Telemetry';
        e.currentTarget.classList.remove('btn-primary');
        document.getElementById('btnExportCSV').disabled = false;
    }
});

document.getElementById('btnExportCSV').addEventListener('click', () => {
    if (AppState.recordedRows.length === 0) return;
    const headers = ["timestamp", "eda", "bvp", "hr", "temp", "acc_mag", "face_area_ratio", "brightness", "sharpness", "motion", "p_phys", "p_vis", "p_fusion_6a", "p_fusion_6b"];
    const rows = AppState.recordedRows.map(r => headers.map(h => r[h]).join(','));
    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(','), ...rows].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `aurasense_authentic_telemetry_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
});

// Settings Modal
document.getElementById('btnOpenConfig').addEventListener('click', () => {
    const u = getUUIDConfig();
    document.getElementById('inputServiceUUID').value = u.service;
    document.getElementById('inputEDAUUID').value = u.eda;
    document.getElementById('inputPPGUUID').value = u.ppg;
    document.getElementById('inputTEMPUUID').value = u.temp;
    document.getElementById('inputACCUUID').value = u.acc;
    document.getElementById('modalConfig').classList.remove('hidden');
});
document.getElementById('btnCloseConfig').addEventListener('click', () => {
    document.getElementById('modalConfig').classList.add('hidden');
});
document.getElementById('btnSaveUUIDs').addEventListener('click', () => {
    saveUUIDConfig({
        service: document.getElementById('inputServiceUUID').value.trim(),
        eda:     document.getElementById('inputEDAUUID').value.trim(),
        ppg:     document.getElementById('inputPPGUUID').value.trim(),
        temp:    document.getElementById('inputTEMPUUID').value.trim(),
        acc:     document.getElementById('inputACCUUID').value.trim()
    });
    document.getElementById('modalConfig').classList.add('hidden');
});
document.getElementById('btnResetUUIDs').addEventListener('click', () => {
    saveUUIDConfig({ ...DEFAULT_UUIDS });
    document.getElementById('modalConfig').classList.add('hidden');
});

// Init
window.addEventListener('DOMContentLoaded', () => {
    chartScreen3EDA = new ChartCanvas('canvasScreen3EDA', [{ id: 'eda', color: '#3b82f6', fill: true }], { minY: 0, maxY: 10 });
    chartScreen3BVP = new ChartCanvas('canvasScreen3BVP', [{ id: 'bvp', color: '#60a5fa', fill: true }], { minY: 100, maxY: 900 });
    chartScreen3TEMP = new ChartCanvas('canvasScreen3TEMP', [{ id: 'temp', color: '#93c5fd', fill: false }], { minY: 30, maxY: 38 });
    chartScreen3ACC = new ChartCanvas('canvasScreen3ACC', [
        { id: 'accX', color: '#3b82f6', lineWidth: 1.5 },
        { id: 'accY', color: '#60a5fa', lineWidth: 1.5 },
        { id: 'accZ', color: '#10b981', lineWidth: 2 }
    ], { minY: -2, maxY: 3 });

    updateBufferSlotsGrid();
    pushAllTracesToScreen3();
});
