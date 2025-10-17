# Hand Tracking Project

Real-time hand detection and tracking using MediaPipe and OpenCV.

## How It Works

The project uses MediaPipe's hand detection solution to identify 21 hand landmarks in real-time video. Each hand landmark represents a specific point on the hand (fingertips, joints, wrist) with pixel coordinates that can be used for gesture recognition and interactive applications.

## Setup

1. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Basic hand tracking:**
```bash
python HandTrackingModule.py
```

**Minimal example:**
```bash
python HandTrackingMin.py
```

**Game application:**
```bash
python NewHandTrackingGame.py
```

## API

```python
from HandTrackingModule import HandDetector

# Initialize detector
detector = HandDetector()

# Detect hands and get landmarks
image = detector.find_hands(image)
landmark_list = detector.find_position(image)

# Access specific landmarks
thumb_tip = landmark_list[4]  # Thumb tip position
index_tip = landmark_list[8]   # Index finger tip position
```

## Hand Landmarks

21 landmarks per hand (0-20):
- **Fingertips**: 4, 8, 12, 16, 20 (thumb, index, middle, ring, pinky)
- **Wrist**: 0
- **Joints**: Various MCP, PIP, DIP joints for each finger

Use constants for clarity:
```python
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
PINKY_TIP = 20
```
