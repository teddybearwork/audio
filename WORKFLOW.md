
---

### WORKFLOW.md

```markdown
# Development Workflow

This document outlines the development workflow for the Animal Sound Recognition System, detailing the steps taken to build, test, and deliver the project within the specified 2-week timeline.

## Timeline
- **Start Date**: August 27, 2025
- **Milestone (50% Completion)**: September 3, 2025
- **Final Delivery**: September 11, 2025 (or earlier, if possible)

## Workflow Steps

### 1. Project Setup (Day 1–2)
- **Tasks**:
  - Set up a GitHub repository (`animal-sound-recognition`) and add the client (`dockduck321`) as a collaborator.
  - Initialize project structure with directories for data, models, output, and source code.
  - Create `requirements.txt` with dependencies (TensorFlow/PyTorch, PyAudio/SoundDevice, WebSocket-client, NumPy, Librosa, etc.).
  - Configure development environment with Python 3.8+ and necessary libraries.
- **Deliverables**:
  - Repository setup with initial files (`README.md`, `config.yaml`, `requirements.txt`).
  - Local development environment ready.

### 2. Data Preparation (Day 3–4)
- **Tasks**:
  - Download 24 client-provided audio files (MP3) from Google Drive or GitHub.
  - Source background noise samples (water, leaves, cave echoes, crickets) from open datasets (ESC-50, UrbanSound8K, FreeSound).
  - Preprocess audio files:
    - Convert MP3 to WAV for consistency.
    - Generate Mel-spectrograms for model input.
    - Apply data augmentation (noise addition, pitch shift, time-stretching) to improve robustness.
  - Create a dataset with labeled classes (24 animal/bird sounds + noise categories).
- **Deliverables**:
  - Preprocessed dataset in `data/` directory.
  - Python script (`preprocess.py`) for audio preprocessing and augmentation.

### 3. Model Development (Day 5–7)
- **Tasks**:
  - Implement transfer learning using YAMNet (pre-trained audio classification model).
  - Fine-tune YAMNet on the client’s 24 audio classes plus noise categories.
  - Optimize model for high accuracy (≥80% confidence threshold) and low latency.
  - Validate model performance using a test split of the dataset.
  - Save trained model to `models/` directory.
- **Deliverables**:
  - Trained model checkpoint (`models/yamnet_finetuned.h5`).
  - Python script (`model.py`) for training and inference.

### 4. Real-Time Streaming Pipeline (Day 8–10)
- **Tasks**:
  - Develop WebSocket-based audio streaming using Python’s `websocket-client` library.
  - Implement a circular buffer to handle audio streams from the MEMS microphone.
  - Add support for laptop microphone input (using PyAudio/SoundDevice) for testing.
  - Create sliding window mechanism (1–2 second segments) for real-time inference.
  - Integrate noise filtering (band-pass, noise reduction) to minimize false positives.
  - Log detection results (sound, timestamp, confidence) to `output/detections.csv`.
- **Deliverables**:
  - Python script (`stream.py`) for WebSocket and laptop mic streaming.
  - Functional streaming pipeline with low-latency inference.

### 5. Testing and Validation (Day 11–12)
- **Tasks**:
  - Test the system with laptop microphone input to validate real-time detection.
  - Simulate WebSocket streaming with sample audio to ensure compatibility with MEMS microphone.
  - Verify model accuracy in noisy environments using synthetic test cases.
  - Debug and optimize latency to achieve ~200–500ms detection time.
- **Deliverables**:
  - Tested prototype with working real-time detection.
  - Debug logs in `logs/app.log`.

### 6. Final Integration and Delivery (Day 13–14)
- **Tasks**:
  - Integrate all components into a single CLI script (`main.py`).
  - Add command-line arguments for input source (`--input websocket` or `--input laptop`).
  - Finalize CSV logging with timestamp, sound, and confidence.
  - Write comprehensive documentation (`README.md`, `WORKFLOW.md`, `OBJECTIVE.md`).
  - Package source code and dependencies for client delivery.
  - Push final code to GitHub repository.
- **Deliverables**:
  - Complete CLI tool (`main.py`) for real-time sound recognition.
  - Full source code and documentation in GitHub repository.
  - CSV output functionality for detection logs.

## Milestone Deliverables (September 3, 2025)
- **50% Completion**:
  - Preprocessed dataset with spectrograms and augmentations.
  - Partially trained model with initial validation results.
  - Basic streaming pipeline with laptop microphone support.
  - Preliminary code pushed to GitHub repository.
  - Zip file containing source code and setup instructions.

## Final Deliverables (September 11, 2025)
- Fully functional CLI tool for real-time animal sound recognition.
- Support for WebSocket streaming with MEMS microphone and laptop microphone testing.
- Trained model with ≥80% confidence threshold.
- CSV logging of detections (sound, timestamp, confidence).
- Complete source code and documentation in GitHub repository.

## Tools and Technologies
- **Programming**: Python 3.8+
- **Machine Learning**: TensorFlow/PyTorch, YAMNet
- **Audio Processing**: Librosa, PyAudio/SoundDevice
- **Streaming**: WebSocket-client
- **Data Augmentation**: Audiomentations
- **Logging**: Python `logging` module
- **Version Control**: Git, GitHub
- **Datasets**: Client-provided audio, ESC-50, UrbanSound8K, FreeSound

## Notes
- The client requested a lightweight CLI tool, so no GUI is developed.
- No retraining functionality is included, as per client requirements.
- The system is optimized for low-latency detection and robust performance in noisy environments.
- All code is developed locally and pushed to GitHub upon completion or milestone delivery.