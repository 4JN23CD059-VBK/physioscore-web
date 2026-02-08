from flask import Flask, render_template, Response, jsonify, request
import gc
import cv2
import numpy as np
import torch
import os
import time
import threading 
import base64
from torch.nn import BatchNorm3d, Conv3d, Linear, MaxPool3d, Module, Dropout, functional as F

# -------------------------------------------------------------------------
# --- CRITICAL CONFIGURATION & GLOBALS ---
# -------------------------------------------------------------------------

PROJECT_ROOT = './'
# 💥 VERIFY THIS PATH 💥
# Ensure your model file is named exactly this or update the path
DEMO_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'model_weights_lite.pth')
# --- CONFIGURED RESOLUTION & CAMERA PATH ---
# Using the path and resolution specified in your last request:
VIDEO_PATH = 0 
TARGET_WIDTH = 1920  
TARGET_HEIGHT = 1080 
# -------------------------------------------

FRAME_COUNT = 64
TARGET_SIZE = (115, 115)
DEPTH_MASK_MULTIPLIER = 1.8 

# --- DUAL BUFFERS for performance separation ---
RAW_FRAME_BUFFER = []      # Fast: Stores raw OpenCV frames for MiDaS input
PROCESSED_FRAME_BUFFER = [] # Slow: Stores MiDaS output (depth maps) for AQA input

# Global controls for threading and locking
THREAD_STOP_EVENT = threading.Event() 
RAW_BUFFER_LOCK = threading.Lock() 
PROCESSED_BUFFER_LOCK = threading.Lock()

# Shared Global State for score and feedback
LATEST_AQA_DATA = [{"score": "...", "feedback": "Initializing...", "class": "score-initializing", "progress": 0}] 

# Display Resolution for the web interface (kept lower for fast streaming)
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = int(DISPLAY_WIDTH * (9 / 16))

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------------------
# --- MODEL DEFINITION & LOADING ---
# -------------------------------------------------------------------------
class Deeper3DCNN(Module):
    def __init__(self, num_output_classes=1):
        super().__init__()
        self.conv1 = Conv3d(1, 32, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)); self.bn1 = BatchNorm3d(32)
        self.conv2 = Conv3d(32, 64, (3, 3, 3), stride=(1, 2, 2), padding=1); self.bn2 = BatchNorm3d(64)
        self.pool1 = MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv3 = Conv3d(64, 128, (3, 3, 3), padding=1); self.bn3 = BatchNorm3d(128)
        self.conv4 = Conv3d(128, 256, (3, 3, 3), padding=1); self.bn4 = BatchNorm3d(256)
        self.pool2 = MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.fc1 = Linear(401408, 512)
        self.dropout = Dropout(p=0.5)
        self.fc_out = Linear(512, num_output_classes)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x))); x = F.relu(self.bn4(self.conv4(x))); x = self.pool2(x)
        x = x.flatten(start_dim=1); x = F.relu(self.fc1(x)); x = self.dropout(x); x = self.fc_out(x)
        return x

model = Deeper3DCNN(num_output_classes=1).to(device)
try:
    model.load_state_dict(torch.load(DEMO_CHECKPOINT_PATH, map_location=device))
    model.eval()
    print("✅ Deeper3DCNN model weights loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load Deeper3DCNN model checkpoint: {e}")
    pass 

# NOTE: MiDaS loading is disabled here to keep memory under 512MB for Render Free Tier.
# If deploying on a system with >2GB RAM, you can re-enable the MiDaS block.

# -------------------------------------------------------------------------
# --- PROCESSING AND FEEDBACK FUNCTIONS ---
# -------------------------------------------------------------------------
def run_midas(frame: np.ndarray) -> np.ndarray:
    # Lean mode: Just convert to gray and resize to save RAM
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)

def apply_depth_mask(depth_map: np.ndarray) -> np.ndarray:
    return depth_map # Just pass it through in Lean mode

def resize_and_normalize(masked_depth_image: np.ndarray) -> np.ndarray:
    """Applies final resize and normalization (0-1 scaling)."""
    resized = cv2.resize(masked_depth_image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    processed_frame = resized.astype(np.float32)
    max_val = np.max(processed_frame)
    if max_val > 0:
         processed_frame /= max_val
    return processed_frame

def generate_corrective_feedback(predicted_score):
    """Generates user-friendly feedback based on the predicted score."""
    score_display = f"{predicted_score:.1f}"
    if predicted_score >= 90.0: 
        return {"text": f"EXCELLENT! Score {score_display}/100. Perfect rehabilitation technique!", "class": "score-excellent"}
    elif 80.0 <= predicted_score < 90.0: 
        return {"text": f"GOOD FORM! Score {score_display}/100. Maintain depth and focus on consistent tempo.", "class": "score-good"}
    elif 65.0 <= predicted_score < 80.0: 
        return {"text": f"IMPROVEMENT NEEDED. Score {score_display}/100. Focus on driving hips back and increasing depth.", "class": "score-medium"}
    else: 
        return {"text": f"MAJOR DEVIATION. Score {score_display}/100. Seek maximum depth and keep your chest up!", "class": "score-poor"}


# -------------------------------------------------------------------------
# --- THREADED AQA PROCESSING (Slow, runs in background) ---
# -------------------------------------------------------------------------

def process_frames_for_aqa():
    """Manages raw frame processing and 3DCNN inference."""
    global RAW_FRAME_BUFFER, PROCESSED_FRAME_BUFFER, LATEST_AQA_DATA
    
    while not THREAD_STOP_EVENT.is_set():
        # 1. Acquire ONE raw frame for processing
        raw_frame = None
        with RAW_BUFFER_LOCK:
            if RAW_FRAME_BUFFER:
                raw_frame = RAW_FRAME_BUFFER.pop(0) 

        if raw_frame is not None:
            # 2. RUN Pre-Processing (Optimized for Render)
            try:
                # RESEARCH COMMENT: Check model availability before processing
                if 'model' not in globals():
                    LATEST_AQA_DATA[0].update({"feedback": "Model failed to load.", "class": "score-poor"})
                    continue

                depth_map = run_midas(raw_frame)
                masked_depth_image = apply_depth_mask(depth_map)
                processed_frame = resize_and_normalize(masked_depth_image)

                with PROCESSED_BUFFER_LOCK:
                    PROCESSED_FRAME_BUFFER.append(processed_frame)

                # CLEAN RAM IMMEDIATELY after processing frame
                del depth_map, masked_depth_image
                gc.collect()
            except Exception as e:
                print(f"Processing Error: {e}")
                continue

        # 3. INFERENCE LOGIC
        with PROCESSED_BUFFER_LOCK:
            if len(PROCESSED_FRAME_BUFFER) >= FRAME_COUNT:
                clip_data = np.stack(PROCESSED_FRAME_BUFFER[:FRAME_COUNT], axis=0)
                # Sliding window: keep the last 32 frames for continuity (Ref: System B)
                PROCESSED_FRAME_BUFFER = PROCESSED_FRAME_BUFFER[32:]
                
                # Perform 3DCNN Inference
                X = torch.from_numpy(clip_data).unsqueeze(0).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    predicted_output = model(X)
                    raw_score = predicted_output.item()
                    feedback_data = generate_corrective_feedback(raw_score)
                    LATEST_AQA_DATA[0].update({
                        "score": f"{raw_score:.2f}",
                        "feedback": feedback_data['text'],
                        "class": feedback_data['class'],
                        "progress": 100
                    })
                
                # FINAL RAM CLEANUP
                del X
                gc.collect()

        time.sleep(0.01)

# NEW ROUTE: This is how the browser sends the webcam data to the server
@app.route('/process_webcam', methods=['POST'])
def process_webcam():
    data = request.get_json()
    image_data = data['image'].split(",")[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    with RAW_BUFFER_LOCK:
        if len(RAW_FRAME_BUFFER) < 10:
            RAW_FRAME_BUFFER.append(frame)
            
    return jsonify({"status": "received"})


# -------------------------------------------------------------------------
# --- FLASK APPLICATION & VIDEO GENERATOR ---
# -------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/score_feed')
def score_feed():
    """Route to serve the score, feedback text, CSS class, and progress."""
    data = LATEST_AQA_DATA[0]
    return jsonify({
        'score': data['score'], 
        'feedback_text': data['feedback'],
        'feedback_class': data['class'],
        'progress': data['progress']
    })

if __name__ == '__main__':
    # Start the AQA processing thread manually
    aqa_thread = threading.Thread(target=process_frames_for_aqa)
    aqa_thread.daemon = True 
    aqa_thread.start()
    
    print("Web App running at http://127.0.0.1:5000/")
    # Render requires host 0.0.0.0
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
