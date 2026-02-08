import torch
import os
import sys

# Import your model class from your app file
try:
    from app import Deeper3DCNN 
except ImportError:
    print("❌ Error: Could not find Deeper3DCNN in app.py. Make sure app.py is in this folder!")
    sys.exit()

def shrink():
    # YOUR ACTUAL FILENAME
    input_file = 'Deeper3DCNN_Checkpoint_L1_RECOVERY_BEST_MODEL.pth'
    output_file = 'model_weights_lite.pth'

    if not os.path.exists(input_file):
        print(f"❌ Error: Could not find '{input_file}'")
        print(f"Current folder contents: {os.listdir('.')}")
        return

    print(f"📂 Reading the big model: {input_file}...")
    model = Deeper3DCNN()
    
    # Load the big weights
    state_dict = torch.load(input_file, map_location='cpu')
    
    # Handle cases where the file is a full checkpoint (has 'state_dict' key) 
    # vs just the weights
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    print("⚡ Shrinking model (Dynamic Quantization)...")
    # This magic line reduces memory usage by converting weights to 8-bit integers
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear, torch.nn.Conv3d}, 
        dtype=torch.qint8
    )
    
    print(f"💾 Saving to: {output_file}...")
    torch.save(quantized_model.state_dict(), output_file)
    
    old_size = os.path.getsize(input_file) / (1024*1024)
    new_size = os.path.getsize(output_file) / (1024*1024)
    print(f"✅ Success!")
    print(f"   Old size: {old_size:.2f} MB")
    print(f"   New size: {new_size:.2f} MB")
    print(f"   Reduction: {((old_size-new_size)/old_size)*100:.1f}%")

if __name__ == "__main__":
    shrink()