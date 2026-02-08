import cv2
import numpy as np
import torch
import os
import time
from torch.nn import BatchNorm3d, Conv3d, Linear, MaxPool3d, Module, Dropout, functional as F

# -------------------------------------------------------------------------
# --- CRITICAL CONFIGURATION (MODIFIED) ---
# -------------------------------------------------------------------------

# 💥 CHANGE THIS PATH 💥
# This should point to the directory where your model file is saved on your laptop.
PROJECT_ROOT = './'  # Use '.' if model is in the same directory as the script

# Path to the trained model weights
DEMO_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'Deeper3DCNN_Checkpoint_L1_RECOVERY_BEST_MODEL.pth')

# ***************************************************************
# *** MODIFICATION START: Switching to Lenovo FHD WebCam ***
# ***************************************************************

# VIDEO_PATH is set to 1 to capture the first external camera (your Lenovo Cam).
# If the laptop's internal cam still opens, change this to 2, 3, etc.
VIDEO_PATH = 0 

# Resolution settings for the new FHD camera
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080 

# ***************************************************************
# *** MODIFICATION END ***
# ***************************************************************

FRAME_COUNT = 64
TARGET_SIZE = (115, 115)
DEPTH_MASK_MULTIPLIER = 1.8 
FRAME_BUFFER = []

# Device setup: Use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# 1. DEPTH-AWARE PREPROCESSING FUNCTIONS
# -------------------------------------------------------------------------

# --- A. MiDaS Model Loading & Initialization ---
print("Initializing MiDaS model...")
try:
    midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True).to(device)
    midas_model.eval()
    midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transform.small_transform
    print("✅ MiDaS model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Failed to load MiDaS. Error: {e}")
    exit()

# --- B. MiDaS Inference Function ---
def run_midas(frame: np.ndarray) -> np.ndarray:
    """Runs MiDaS inference and returns the raw depth map (disparity)."""
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

# --- C. Depth Masking Function ---
def apply_depth_mask(depth_map: np.ndarray) -> np.ndarray:
    """Applies a simple foreground-based depth mask."""
    # Find the threshold: 5th percentile for the foreground object
    depth_threshold = np.percentile(depth_map, 5) * DEPTH_MASK_MULTIPLIER
    mask = (depth_map < depth_threshold).astype(np.uint8) * 255
    masked_depth_image = cv2.bitwise_and(depth_map, depth_map, mask=mask)
    return masked_depth_image

# --- D. Final Normalization ---
def resize_and_normalize(masked_depth_image: np.ndarray) -> np.ndarray:
    """Applies the final resize and CRITICAL normalization (0-1 scaling)."""
    resized = cv2.resize(masked_depth_image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    processed_frame = resized.astype(np.float32)
    max_val = np.max(processed_frame)
    if max_val > 0:
         processed_frame /= max_val
    return processed_frame

# --- E. Feedback Function (Revised for ASCII compatibility) ---
def generate_corrective_feedback(predicted_score):
    # Round the score once for display consistency
    score_display = f"{predicted_score:.1f}"
    
    if predicted_score >= 90: 
        return f"EXCELLENT! Score {score_display}/100. Perfect technique!"
    elif 80 <= predicted_score < 90: 
        return f"GOOD FORM! Score {score_display}/100. Focus on tempo/depth consistency."
    elif 65 <= predicted_score < 80: 
        return f"IMPROVEMENT. Score {score_display}/100. Increase hip depth, maintain a straighter back."
    else: 
        return f"MAJOR ERRORS. Score {score_display}/100. Focus on driving hips back and increasing depth."

# -------------------------------------------------------------------------
# 2. MODEL DEFINITION & LOADING
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
    print(f"FATAL ERROR: Could not load Deeper3DCNN model checkpoint from {DEMO_CHECKPOINT_PATH}. Error: {e}")
    print("Please check the path and file name.")
    exit()

# -------------------------------------------------------------------------
# 3. LIVE PROCESSING LOOP (MODIFIED FOR RESIZABILITY)
# -------------------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH) 

if not cap.isOpened():
    print(f"FATAL ERROR: Could not open webcam (index {VIDEO_PATH}). Please try changing VIDEO_PATH to 0, 2, etc.")
    exit()
    
# Set the resolution to the high-quality FHD specs of the new Lenovo camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

print(f"Successfully opened camera index {VIDEO_PATH}.")
print(f"Attempting to stream at {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} resolution.")

print(f"🚀 Starting Live AQA Stream from Webcam...")
print("-> Stand in front of the camera and perform the squat exercise.")
print("-> Press 'q' to quit.")

current_score = "..."
current_feedback = "Initializing..."

# ***************************************************************
# *** MODIFICATION START: Setting windows to be resizable ***
# ***************************************************************

# Define a usable display size (800 width, calculating the 16:9 height)
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = int(DISPLAY_WIDTH * (9 / 16)) # 800 * 9 / 16 = 450

# 1. Create Windows in a resizable mode
cv2.namedWindow("Original Feed (Press 'q' to Quit)", cv2.WINDOW_NORMAL)
cv2.namedWindow("Depth Mask Input (Background-Invariant)", cv2.WINDOW_NORMAL)

# 2. Set an initial size (enforcing the 16:9 aspect ratio)
cv2.resizeWindow("Original Feed (Press 'q' to Quit)", DISPLAY_WIDTH, DISPLAY_HEIGHT)
cv2.resizeWindow("Depth Mask Input (Background-Invariant)", DISPLAY_WIDTH, DISPLAY_HEIGHT)
# ***************************************************************
# *** MODIFICATION END ***
# ***************************************************************


while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip frame horizontally for mirror effect (helpful for self-correction)
    frame = cv2.flip(frame, 1)
    
    # 💥 THE DEPTH-AWARE PIPELINE 💥
    depth_map = run_midas(frame)
    masked_depth_image = apply_depth_mask(depth_map)
    processed_frame = resize_and_normalize(masked_depth_image) 

    # --- Prepare for Inference ---
    FRAME_BUFFER.append(processed_frame)

    if len(FRAME_BUFFER) == FRAME_COUNT:
        # 1. Prepare Tensor Input
        clip_data = np.stack(FRAME_BUFFER, axis=0)
        X = torch.from_numpy(clip_data).unsqueeze(0).unsqueeze(0).float().to(device)

        # 2. Run 3DCNN Inference
        with torch.no_grad():
            predicted_output = model(X)
            current_score = f"{predicted_output.item():.2f}"
            current_feedback = generate_corrective_feedback(predicted_output.item())

        # 3. Clear Buffer (50% overlap for continuous analysis)
        FRAME_BUFFER = FRAME_BUFFER[int(FRAME_COUNT/2):] 
        
    # --- Live Display Logic ---
    
    # 1. Create Depth Map Visual for debugging
    # Scale depth map to 0-255 for visualization
    depth_visual = (masked_depth_image - np.min(masked_depth_image))
    if np.max(depth_visual) > 0:
        depth_visual = 255 * (depth_visual / np.max(depth_visual))
    depth_visual = depth_visual.astype(np.uint8)
    # Convert single channel to 3-channel for display
    depth_visual = cv2.cvtColor(depth_visual, cv2.COLOR_GRAY2BGR)

    # 2. Display Score and Feedback on the RGB frame
    cv2.putText(frame, f"AQA Score: {current_score} / 100", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Feedback: {current_feedback}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 3. Show Windows (Now they will be resizable)
    cv2.imshow("Original Feed (Press 'q' to Quit)", frame)
    cv2.imshow("Depth Mask Input (Background-Invariant)", depth_visual)

    # 4. Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("\nLive AQA stream closed successfully.")