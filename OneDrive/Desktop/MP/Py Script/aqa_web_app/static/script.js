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

    let isVideoLoaded = false;
    
    // Function to fetch the score from the backend
    function fetchScore() {
        fetch('/score_feed')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Update Score and Feedback
                scoreValue.textContent = data.score;
                feedbackText.textContent = data.feedback_text;

                // Update Card Styling based on feedback_class (e.g., score-excellent)
                // Note: Assumes feedback_class aligns with the CSS classes defined in the HTML <style> block
                feedbackCard.className = `score-card ${data.feedback_class}`;

                // Update Progress Bar
                const progress = data.progress;
                progressBar.style.width = `${progress}%`;
                
                // Change progress bar color based on status (better visual cue)
                progressBar.classList.remove('bg-blue-500', 'bg-green-500', 'bg-red-500');
                if (data.feedback_class === 'score-initializing') {
                    progressBar.classList.add('bg-blue-500');
                } else if (data.feedback_class === 'score-excellent' || data.feedback_class === 'score-good') {
                    progressBar.classList.add('bg-green-500');
                } else {
                    progressBar.classList.add('bg-red-500');
                }
                
                // Estimate current frame count based on progress
                const FRAME_COUNT_MAX = 64; 
                const currentFrames = Math.min(FRAME_COUNT_MAX, Math.floor(FRAME_COUNT_MAX * (progress / 100)));
                frameCountStatus.textContent = currentFrames;

            })
            .catch(error => {
                console.error('Error fetching score feed:', error);
                feedbackText.textContent = "Connection lost. Restarting server...";
                feedbackCard.className = 'score-card score-poor';
                scoreValue.textContent = "ERR";
            });
    }

    // Check if the video stream has started loading
    videoDisplay.onload = () => {
        if (!isVideoLoaded) {
            console.log("Video stream started successfully.");
            loadingOverlay.style.display = 'none';
            isVideoLoaded = true;
        }
    };
    
    // Fallback for camera not starting quickly (keep overlay visible)
    setTimeout(() => {
        if (!isVideoLoaded) {
            loadingOverlay.textContent = "Camera not responding. Check app.py VIDEO_PATH.";
        }
    }, 5000);


    // Poll the score feed every 500ms (twice a second)
    setInterval(fetchScore, 500);

    // Initial fetch to populate the UI immediately
    fetchScore();
});