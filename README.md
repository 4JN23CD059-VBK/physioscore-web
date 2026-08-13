# PhysioScore - Automated Quality Assessment for Rehabilitation Exercises

PhysioScore is a deep learning-powered web application designed to evaluate and score physical therapy exercise performance in real time. 

## Features
* **Pose & Depth Analysis**: Integrated lightweight model pipeline (optimized from MiDaS & 3DCNN architectures).
* **Real-time Scoring**: Provides quantitative feedback on exercise accuracy for physical rehabilitation.
* **Web Deployment**: Lightweight Flask backend integrated with Gunicorn and hosted for seamless access.

## 🛠️ Project Structure
```text
aqa_web_app/
├── app.py                     # Main Flask Application
├── live_aqa_system_1.py       # Core evaluation & inference script
├── model_weights_lite.pth     # Lightweight trained model weights
├── requirements.txt           # Python dependencies
├── static/                    # Frontend styles & assets
└── templates/                 # HTML templates
