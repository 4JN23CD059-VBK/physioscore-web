/**
 * Script for fetching the AQA score and updating the UI dynamically.
 */
document.addEventListener('DOMContentLoaded', () => {
    const scoreValue = document.getElementById('score_value');
    const feedbackText = document.getElementById('feedback_text');
    const feedbackCard = document.getElementById('feedback_card');
    const progressBar = document.getElementById('progress_bar');
    const frameCountStatus = document.getElementById('frame_count_status');
    const videoDisplay = document.getElementById('video_display');
    const loadingOverlay = document.getElementById('loading_overlay');

    // --- NEW: WEBCAM TRANSMITTER LOGIC ---
    const hiddenCanvas = document.createElement('canvas');
    const ctx = hiddenCanvas.getContext('2d');
    hiddenCanvas.width = 640;  // Standard for MiDaS
    hiddenCanvas.height = 480;

    // 1. Get User Media (Open the webcam in the browser)
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
        .then(stream => {
            videoDisplay.srcObject = stream;
            videoDisplay.play();
            loadingOverlay.style.display = 'none';
            
            // 2. Start sending frames to the server every 150ms (~7 FPS)
            // We use 150ms to avoid overwhelming the Render Free Tier RAM
            setInterval(sendFrameToServer, 150);
        })
        .catch(err => {
            console.error("Webcam Error: ", err);
            loadingOverlay.textContent = "Please allow camera access for the Research Demo.";
        });

    function sendFrameToServer() {
        // Draw the current video frame to our hidden canvas
        ctx.drawImage(videoDisplay, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
        
        // Convert to Base64 JPEG (Quality 0.5 to save bandwidth)
        const dataUrl = hiddenCanvas.toDataURL('image/jpeg', 0.5);

        fetch('/process_webcam', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl })
        }).catch(err => console.log("Upload failed. Server likely rebooting..."));
    }

    // --- EXISTING: SCORE FETCHING LOGIC ---
    function fetchScore() {
        fetch('/score_feed')
            .then(response => response.json())
            .then(data => {
                scoreValue.textContent = data.score;
                feedbackText.textContent = data.feedback_text;
                feedbackCard.className = `score-card ${data.feedback_class}`;
                
                const progress = data.progress;
                progressBar.style.width = `${progress}%`;
                
                const FRAME_COUNT_MAX = 64; 
                const currentFrames = Math.min(FRAME_COUNT_MAX, Math.floor(FRAME_COUNT_MAX * (progress / 100)));
                frameCountStatus.textContent = currentFrames;
            });
    }

    setInterval(fetchScore, 500);
});
