const typingArea = document.getElementById('typingArea');
const clearBtn = document.getElementById('clearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');

const emotionResult = document.getElementById('emotionResult');
const confidenceResult = document.getElementById('confidenceResult');
const wpmResult = document.getElementById('wpmResult');
const avgIntervalResult = document.getElementById('avgIntervalResult');
const avgPauseResult = document.getElementById('avgPauseResult');

// Raw events
let keyEvents = [];
let sessionStartTime = null;
let lastKeyTime = null;

// Capture keydown/keyup with timestamps
typingArea.addEventListener('keydown', function (event) {
    const now = performance.now();

    if (!sessionStartTime) {
        sessionStartTime = now;
    }

    keyEvents.push({
        type: 'keydown',
        key: event.key,
        time: now
    });

    lastKeyTime = now;
});

typingArea.addEventListener('keyup', function (event) {
    const now = performance.now();

    keyEvents.push({
        type: 'keyup',
        key: event.key,
        time: now
    });

    lastKeyTime = now;
});

// Clear button
clearBtn.addEventListener('click', function () {
    typingArea.value = '';
    keyEvents = [];
    sessionStartTime = null;
    lastKeyTime = null;

    emotionResult.textContent = 'N/A';
    confidenceResult.textContent = 'N/A';
    wpmResult.textContent = 'N/A';
    avgIntervalResult.textContent = 'N/A';
    avgPauseResult.textContent = 'N/A';
});

// Analyze button: extract features and send to backend
analyzeBtn.addEventListener('click', async function () {
    if (keyEvents.length === 0) {
        alert('Please type something first.');
        return;
    }

    const text = typingArea.value;
    const features = computeFeatures(keyEvents, text);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: features })
        });

        const data = await response.json();

        emotionResult.textContent = data.emotion || 'N/A';
        confidenceResult.textContent = data.confidence !== undefined
            ? (data.confidence * 100).toFixed(2) + '%'
            : 'N/A';

        wpmResult.textContent = features.typing_speed_wpm.toFixed(2);
        avgIntervalResult.textContent = features.avg_key_interval_ms.toFixed(2);
        avgPauseResult.textContent = features.avg_pause_ms.toFixed(2);
    } catch (err) {
        console.error(err);
        alert('Error while contacting server.');
    }
});

// Feature extraction logic
function computeFeatures(events, text) {
    let keydownTimes = [];
    let keyupTimes = [];
    let intervals = [];
    let pauses = [];

    let lastEventTime = null;

    events.forEach(ev => {
        if (ev.type === 'keydown') {
            keydownTimes.push(ev.time);
        } else if (ev.type === 'keyup') {
            keyupTimes.push(ev.time);
        }

        if (lastEventTime !== null) {
            const diff = ev.time - lastEventTime;
            if (diff > 0) {
                intervals.push(diff);
            }
        }
        lastEventTime = ev.time;
    });

    for (let i = 1; i < keydownTimes.length; i++) {
        const diff = keydownTimes[i] - keydownTimes[i - 1];
        if (diff > 300) {
            pauses.push(diff);
        }
    }

    const totalTimeMs = (events[events.length - 1].time - events[0].time);
    const totalTimeMinutes = totalTimeMs / (1000 * 60);
    const numWords = text.trim().length > 0 ? text.trim().split(/\s+/).length : 0;
    const wpm = totalTimeMinutes > 0 ? numWords / totalTimeMinutes : 0;

    const avgInterval = intervals.length > 0
        ? intervals.reduce((a, b) => a + b, 0) / intervals.length
        : 0;

    const avgPause = pauses.length > 0
        ? pauses.reduce((a, b) => a + b, 0) / pauses.length
        : 0;

    return {
        typing_speed_wpm: wpm,
        avg_key_interval_ms: avgInterval,
        avg_pause_ms: avgPause,
        num_key_events: events.length
    };
}
