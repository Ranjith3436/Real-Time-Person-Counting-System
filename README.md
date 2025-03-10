# Real-Time-Person-Counting-System
This project is an AI-based person counting system that tracks the number of people entering and exiting a space. It utilizes computer vision and deep learning techniques to detect movement and count individuals in real time. The system is useful for applications like crowd management, retail analytics, and security monitoring.

## Overview
This application uses computer vision to detect and count people entering and exiting a defined area in a video stream. It's built with Flask for the web interface and YOLOv8 for real-time object detection.

## Features
- Real-time people detection using YOLOv8
- Tracks entries and exits across a virtual boundary line
- Calculates and displays:
  - Inside count (people entering)
  - Outside count (people exiting)
  - Total count
  - Current occupancy (existing people)
- Responsive web interface with live video feed and statistics

## Technologies Used
- Python 3.x
- Flask (Web Framework)
- OpenCV (Computer Vision)
- YOLOv8 (Object Detection)
- Bootstrap (Frontend Styling)

## Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/people-counter.git
   cd people-counter
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model:
   ```
   # The application will automatically download yolov8s.pt on first run
   # or you can manually download it from the Ultralytics repository
   ```

4. Place your video file in the project directory and update the filename in `app.py`:
   ```python
   cap = cv2.VideoCapture('your-video-file.mp4')
   ```

## Usage
1. Start the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. The application will display the video feed with detection boxes and counting statistics

## Configuration
You can modify the following parameters in `app.py`:
- Line coordinates for entry/exit detection
- Detection confidence threshold
- Counting area size

## Project Structure
- `app.py`: Main application file with video processing and Flask routes
- `templates/index.html`: Frontend HTML template
- `public/styles.css`: CSS styling
- `public/logo.png`: Application logo

## Author
- Ranjith B

## License
© 2025 SHARA AI Productions. All rights reserved.
