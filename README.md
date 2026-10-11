# Intelligent Video Processing and Analysis
Upload a video and analyze important frames, scene changes, motion, objects, and a short summary video.

## WSL Ubuntu
```bash
sudo apt update
sudo apt install ffmpeg python3-venv
cd /home/kiit/Intelligent_Video_Processing
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:7860

Use a short 20–60 second MP4 for the first test. YOLO may download its model on first use.
