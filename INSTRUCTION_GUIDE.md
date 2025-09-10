# Project Objective and Client Requirements

## Objective
The objective of this project is to develop a lightweight Python-based command-line tool that recognizes specific animal, bird, and background noise sounds from a live audio stream captured via a MEMS low-noise single microphone module (3.3V) over a WebSocket connection (WiFi/LAN). The system must process audio in real-time, identify sounds with a confidence threshold of ≥80%, and log detection results (sound, timestamp, confidence) to a CSV file. The solution is designed as a simple, plug-and-play tool for a student project, requiring minimal setup and no future retraining capability.

## Client Requirements
Based on the communication with the client (Ageay srp Sriram, referred to as AG), the following requirements were outlined:

1. **Core Functionality**:
   - Develop a Python module to recognize 24 specific animal and bird sounds from live audio input.
   - Support real-time audio streaming from a MEMS low-noise single microphone module (3.3V) via WebSocket over WiFi/LAN.
   - Output recognized sounds with a confidence score of ≥80% to the console and log results to a CSV file (format: timestamp, sound, confidence).

2. **Audio Dataset**:
   - Use 24 client-provided MP3 audio files for animals and birds, accessible via Google Drive or GitHub.
   - Handle background noises (e.g., water, leaves, cave echoes, crickets) without client-provided audio by sourcing from open datasets (e.g., ESC-50, UrbanSound8K, FreeSound).

3. **System Design**:
   - Deliver a lightweight command-line interface (CLI) tool; no graphical user interface (GUI) is required.
   - Ensure low-latency detection (~200–500ms) using a circular buffer and sliding window approach.
   - Include an option to test with a laptop microphone for validation before deploying with the MEMS microphone.
   - No retraining functionality is needed for adding new sounds in the future.

4. **Performance**:
   - Achieve high accuracy with a confidence threshold of ≥80% to minimize false positives.
   - Implement noise filtering and data augmentation to handle environmental noises effectively.
   - Ensure compatibility with noisy environments by training the model with augmented and synthetic noise data.

5. **Deliverables**:
   - A functional prototype (50% complete) by September 3, 2025, including:
     - Preprocessed dataset and partially trained model.
     - Basic streaming pipeline with laptop microphone support.
     - Source code and setup instructions.
   - Final deliverable by September 11, 2025 (or earlier), including:
     - Fully functional CLI tool for real-time sound recognition.
     - Support for WebSocket streaming and laptop microphone testing.
     - Trained model with ≥80% confidence threshold.
     - CSV logging of detection results.
     - Complete source code and documentation in a GitHub repository.

6. **Constraints**:
   - Budget: Fixed price of $150 USD, as agreed with the client.
   - Timeline: 2 weeks (August 27, 2025 – September 11, 2025).
   - No contract creation until final delivery, as per developer preference.
   - Client is a student with limited budget and time, requiring a simple, working solution as part of a larger project.

7. **Additional Notes**:
   - The client emphasized timely delivery, with a milestone check on September 3, 2025, for a 50% complete prototype.
   - The system must be easy to set up and run without requiring extensive technical knowledge.
   - The developer will push code to a GitHub repository (`animal-sound-recognition`) and add the client (`dockduck321`) as a collaborator for audio file uploads and code review.

## Success Criteria
- The system accurately recognizes the 24 specified sounds with ≥80% confidence in real-time.
- The CLI tool runs smoothly on the client’s system with WebSocket streaming or laptop microphone input.
- Detection results are logged correctly to a CSV file with timestamp, sound, and confidence.
- The solution is delivered on time (by September 11, 2025) and within the agreed budget ($150 USD).
- Comprehensive documentation is provided to ensure the client can set up and run the system without issues.