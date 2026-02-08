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
VIDEO_PATH = 0 # Try 0, 1, 2, etc. if the camera doesn't open
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
    # We allow the app to run without the model, but score updates will fail.
    pass 

"""
# --- MiDaS Model Loading & Initialization ---
print("Initializing MiDaS model...")
try:
    # Attempt to load MiDaS from local cache or download
    midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True).to(device)
    midas_model.eval()
    midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transform.small_transform
    print("✅ MiDaS model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Failed to load MiDaS. Error: {e}")
    # We allow the app to run without MiDaS, but depth processing will fail.
    pass
"""

# -------------------------------------------------------------------------
# --- PROCESSING AND FEEDBACK FUNCTIONS ---
# -------------------------------------------------------------------------
def run_midas(frame: np.ndarray) -> np.ndarray:
    # Lean mode: Just convert to gray and resize to save RAM
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)

def apply_depth_mask(depth_map: np.ndarray) -> np.ndarray:
    return depth_map # Just pass it through

""" def run_midas(frame: np.ndarray) -> np.ndarray:
   # Runs MiDaS on a single frame.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        input_batch = transform(frame_rgb).to(device)
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return np.clip(depth_map, 0.0, None)

def apply_depth_mask(depth_map: np.ndarray) -> np.ndarray:
   #  Applies a simple foreground-based depth mask.
    depth_threshold = np.percentile(depth_map, 5) * DEPTH_MASK_MULTIPLIER
    mask = (depth_map < depth_threshold).astype(np.uint8) * 255
    masked_depth_image = cv2.bitwise_and(depth_map, depth_map, mask=mask)
    return masked_depth_image """


def resize_and_normalize(masked_depth_image: np.ndarray) -> np.ndarray:
    """Applies final resize and normalize (0-1 scaling)."""
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
            """
            if 'midas_model' not in globals() or 'model' not in globals():
                LATEST_AQA_DATA[0].update({"feedback": "Model initialization failed. Score unavailable.", "class": "score-poor", "progress": 0})
                time.sleep(1)
                continue
            try:
                depth_map = run_midas(raw_frame)
                masked_depth_image = apply_depth_mask(depth_map)
                processed_frame = resize_and_normalize(masked_depth_image)
            except Exception as e:
                print(f"MiDaS/Depth Processing Error: {e}")
                time.sleep(1) 
                continue"""
        
def process_frames_for_aqa():
    """Manages raw frame processing and 3DCNN inference (Optimized for Render)."""
    global RAW_FRAME_BUFFER, PROCESSED_FRAME_BUFFER, LATEST_AQA_DATA
    
    while not THREAD_STOP_EVENT.is_set():
        raw_frame = None
        with RAW_BUFFER_LOCK:
            if RAW_FRAME_BUFFER:
                raw_frame = RAW_FRAME_BUFFER.pop(0) 

        if raw_frame is not None:
            try:
                # Optimized Path: Skip MiDaS to stay under 512MB RAM
                gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
                processed_frame = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
                processed_frame = processed_frame.astype(np.float32) / 255.0

                with PROCESSED_BUFFER_LOCK:
                    PROCESSED_FRAME_BUFFER.append(processed_frame)
                
                # Immediate memory cleanup
                del gray
                gc.collect()
            except Exception as e:
                print(f"Processing Error: {e}")
           
        with PROCESSED_BUFFER_LOCK:
            if len(PROCESSED_FRAME_BUFFER) >= FRAME_COUNT:
                clip_data = np.stack(PROCESSED_FRAME_BUFFER[:FRAME_COUNT], axis=0)
                PROCESSED_FRAME_BUFFER = PROCESSED_FRAME_BUFFER[32:]
                
                # Inference
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
# --- FLASK APPLICATION & VIDEO GENERATOR (Fast Stream) ---
# -------------------------------------------------------------------------
app = Flask(__name__)

def video_stream_generator():
    """Captures frames, adds them to the raw buffer, and streams QUICKLY."""
    global RAW_FRAME_BUFFER

    cap = cv2.VideoCapture(VIDEO_PATH) 
    # Set the camera properties to the desired high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    if not cap.isOpened():
        print(f"FATAL CAMERA ERROR: Could not open camera index {VIDEO_PATH}. Try changing VIDEO_PATH.")
        return

    # Start the AQA processing thread only once
    with RAW_BUFFER_LOCK:
        RAW_FRAME_BUFFER = []
        if not hasattr(video_stream_generator, 'aqa_thread') or not video_stream_generator.aqa_thread.is_alive():
            video_stream_generator.aqa_thread = threading.Thread(target=process_frames_for_aqa)
            video_stream_generator.aqa_thread.daemon = True 
            video_stream_generator.aqa_thread.start()
            print("✅ AQA Processing thread started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from camera.")
            time.sleep(0.1)
            continue
        
        frame = cv2.flip(frame, 1) # Flip for mirror effect
        
        # 1. Add raw frame (high res) to the RAW_FRAME_BUFFER (FAST)
        with RAW_BUFFER_LOCK:
            # Only store the newest frame, drop older ones to prevent lag
            if len(RAW_FRAME_BUFFER) < 5: 
                RAW_FRAME_BUFFER.append(frame.copy())
            else:
                RAW_FRAME_BUFFER.pop(0)
                RAW_FRAME_BUFFER.append(frame.copy())
            
        # 2. Encode Frame for Web Streaming (FAST - Display is lower resolution)
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        
        ret, buffer = cv2.imencode('.jpg', display_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()

        # Yield frame in multipart format for streaming (FAST)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    THREAD_STOP_EVENT.set() # Stop the processing thread if the stream closes


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route to serve the live video stream (M-JPEG)."""
    return Response(video_stream_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
    print("Web App running at http://127.0.0.1:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
