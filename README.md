# Hand Tracking Project

Real-time hand detection and tracking using MediaPipe and OpenCV.

## How It Works

The project uses MediaPipe's hand detection solution to identify 21 hand landmarks in real-time video. Each hand landmark represents a specific point on the hand (fingertips, joints, wrist) with pixel coordinates that can be used for gesture recognition and interactive applications.

## Demo

![Demo](assets/demo.gif)

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

**Minimal example:**
```bash
python HandTrackingMin.py
```

**Game application (recommended):**
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

# Calculate distance between landmarks
distance, image, coords = detector.find_distance(thumb_tip, index_tip, image, draw=True)
print(f"Distance: {distance:.1f} pixels")
```

## Hand Landmarks

21 landmarks per hand (0-20):
- **Fingertips**: 4, 8, 12, 16, 20 (thumb, index, middle, ring, pinky)
- **Wrist**: 0
- **Joints**: Various finger joints:
  - **MCP** (Metacarpophalangeal): Knuckle joints where fingers connect to hand
  - **PIP** (Proximal Interphalangeal): Middle joints of each finger
  - **DIP** (Distal Interphalangeal): Joints closest to fingertips

Use constants for clarity:
```python
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
PINKY_TIP = 20
```

## Testing

The project includes comprehensive unit tests to ensure reliability and catch edge cases.

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_distance_calculation.py -v
python -m pytest tests/test_hand_detection.py -v

# Run with coverage report
python -m pytest tests/ --cov=. --cov-report=html
```

### Test Coverage

- **32 comprehensive test cases** covering all HandDetector methods
- **Distance calculation tests**: 11 test cases for `find_distance()` method
- **Hand detection tests**: 7 test cases for `find_hands()` and `find_position()` methods  
- **Edge case testing**: Invalid inputs, boundary conditions, error handling
- **Mock data approach**: Tests run without requiring a camera

### Test Organization

```
tests/
├── test_hand_detector.py      # Core HandDetector class tests
├── test_distance_calculation.py  # Distance method specific tests
└── test_hand_detection.py     # Hand detection specific tests
```

### What's Tested

- ✅ **Distance calculation** with various point formats and edge cases
- ✅ **Hand detection** with mock landmarks and invalid inputs
- ✅ **Error handling** for None inputs, invalid hand numbers, missing results
- ✅ **Edge cases** like zero distance, very large/small coordinates
- ✅ **Input validation** for image formats and parameter bounds

## CI/CD Pipeline

The project includes a comprehensive GitHub Actions CI/CD pipeline that automatically runs on every push and pull request.

### What the Pipeline Does

- **🧹 Code Quality**: Runs flake8 linting to ensure PEP 8 compliance
- **🧪 Testing**: Executes all 32 tests across Python 3.8, 3.9, 3.10, 3.11
- **📊 Coverage**: Generates test coverage reports and uploads to Codecov
- **🔒 Security**: Scans for security vulnerabilities with Bandit and Safety
- **⚡ Performance**: Validates that distance calculations are fast enough
- **📚 Documentation**: Checks docstring style and required files
- **🔧 Build**: Verifies all imports and demo scripts compile correctly

### Pipeline Status

The pipeline runs automatically and provides:
- ✅ **Green checkmarks** when everything passes
- ❌ **Red X marks** when issues are found
- 📈 **Coverage reports** showing test coverage percentage
- 🔍 **Detailed logs** for debugging failed tests

### Local Testing

You can run the same checks locally:

```bash
# Run all tests with coverage
pytest tests/ -v --cov=. --cov-report=html

# Check code quality
flake8 .

# Security scan
bandit -r .

# Check documentation style
pydocstyle HandTrackingModule.py
```
