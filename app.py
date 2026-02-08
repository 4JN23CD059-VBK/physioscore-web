from flask import Flask, render_template, jsonify, request
import gc, cv2, torch, os, time, threading, base64
import numpy as np
from torch.nn import BatchNorm3d, Conv3d, Linear, MaxPool3d, Module, Dropout, functional as F

# --- CRITICAL MEMORY CONTROLS ---
torch.set_num_threads(1)
torch.set_grad_enabled(False)

app = Flask(__name__)
PROJECT_ROOT = './'
DEMO_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'model_weights_lite.pth')
FRAME_COUNT = 64
TARGET_SIZE = (115, 115)

# --- SHARED STATE ---
RAW_FRAME_BUFFER = []
PROCESSED_FRAME_BUFFER = []
RAW_BUFFER_LOCK = threading.Lock()
PROCESSED_BUFFER_LOCK = threading.Lock()
LATEST_AQA_DATA = [{"score": "...", "feedback": "Initializing...", "class": "score-initializing", "progress": 0}]

# --- MODEL DEFINITION ---
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
        self.dropout = Dropout(p=0.5); self.fc_out = Linear(512, num_output_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x))); x = F.relu(self.bn4(self.conv4(x))); x = self.pool2(x)
        x = x.flatten(start_dim=1); x = F.relu(self.fc1(x)); x = self.dropout(x); x = self.fc_out(x)
        return x

# Global model pointer
global_model = None

def get_model():
    global global_model
    if global_model is None:
        device = torch.device("cpu")
        global_model = Deeper3DCNN(num_output_classes=1).to(device)
        if os.path.exists(DEMO_CHECKPOINT_PATH):
            global_model.load_state_dict(torch.load(DEMO_CHECKPOINT_PATH, map_location=device))
        global_model.eval()
        gc.collect()
    return global_model

def process_frames_for_aqa():
    global PROCESSED_FRAME_BUFFER, LATEST_AQA_DATA
    while True:
        raw_frame = None
        with RAW_BUFFER_LOCK:
            if RAW_FRAME_BUFFER:
                raw_frame = RAW_FRAME_BUFFER.pop(0)

        if raw_frame is not None:
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, TARGET_SIZE)
            processed = resized.astype(np.float32) / 255.0
            with PROCESSED_BUFFER_LOCK:
                PROCESSED_FRAME_BUFFER.append(processed)
            del raw_frame, gray, resized
            gc.collect()

        with PROCESSED_BUFFER_LOCK:
            if len(PROCESSED_FRAME_BUFFER) >= FRAME_COUNT:
                clip = np.stack(PROCESSED_FRAME_BUFFER[:FRAME_COUNT], axis=0)
                PROCESSED_FRAME_BUFFER = PROCESSED_FRAME_BUFFER[32:] 
                X = torch.from_numpy(clip).unsqueeze(0).unsqueeze(0)
                
                model = get_model()
                with torch.no_grad():
                    score = model(X).item()
                    # Mapping feedback
                    fb = "EXCELLENT" if score >= 85 else "GOOD" if score >= 70 else "ADJUST FORM"
                    cl = "score-excellent" if score >= 85 else "score-good" if score >= 70 else "score-poor"
                    LATEST_AQA_DATA[0].update({"score": f"{score:.1f}", "feedback": fb, "class": cl, "progress": 100})
                del X, clip
                gc.collect()
        time.sleep(0.05)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/process_webcam', methods=['POST'])
def process_webcam():
    data = request.get_json()
    image_data = data['image'].split(",")[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    with RAW_BUFFER_LOCK:
        if len(RAW_FRAME_BUFFER) < 10: RAW_FRAME_BUFFER.append(frame)
    return jsonify({"status": "ok"})

@app.route('/score_feed')
def score_feed():
    return jsonify(LATEST_AQA_DATA[0])

if __name__ == '__main__':
    threading.Thread(target=process_frames_for_aqa, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
