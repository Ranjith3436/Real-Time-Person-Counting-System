from flask import Flask, render_template, Response, jsonify, send_from_directory
import sys
import cv2
import os
from ultralytics import YOLO
import numpy as np
import time
from math import atan2, cos, sin, radians
import threading
# Set process priority (higher values mean lower priority)
os.nice(10)  # 0 (normal) to 19 (lowest priority)

# Load YOLOv8 model
model = YOLO("yolov8s.pt")  # You can use 'yolov8s.pt' for better accuracy

# Global variables for counts
inside_count = 0
outside_count = 0
total_count = 0
existing_people = 0
lock = threading.Lock()  # Thread lock to ensure thread-safe updates

# Function to calculate the cross product to determine the side of the line
def point_line_side(p, p1, p2):
    # Cross product (p2 - p1) x (p - p1)
    d = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
    return d

def enter_status(In, d):
    return (In > 0 and d > In) or (In <= 0 and d < In)

# Function to calculate the perpendicular vectors and form the rectangle
def get_rectangle_coords(p1, p2, padding=100):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = atan2(dy, dx)
    perp_length = padding
    norm = np.sqrt(dy**2 + dx**2)
    dx_perp = -dy / norm * perp_length
    dy_perp = dx / norm * perp_length
    p1_top_left = (p1[0] + dx_perp, p1[1] + dy_perp)
    p1_bottom_left = (p1[0] - dx_perp, p1[1] - dy_perp)
    p2_top_left = (p2[0] + dx_perp, p2[1] + dy_perp)
    p2_bottom_left = (p2[0] - dx_perp, p2[1] - dy_perp)
    return [p1_top_left, p1_bottom_left, p2_bottom_left, p2_top_left]

app = Flask(__name__)

# Video processing function
def process_video():
    global inside_count, outside_count,total_count,existing_people
    cap = cv2.VideoCapture('the-cctv-people-demo-1_ndRSex0j.mp4')  # Replace with your video path
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0

    # User-defined coordinates for line
    line_point_1 = (350, 500)  # First point of the line (x1, y1)
    line_point_2 = (1000, 500)  # Second point of the line (x2, y2)



    # Table inside walk
    # line_point_1 = (330, 650)  # First point of the line (x1, y1)
    # line_point_2 = (800, 300)  # Second point of the line (x2, y2)

    # User-defined side for "IN" and "OUT"
    side_point = (200, 150)  # A point on the "IN" side

    # Get the rectangle coordinates
    count_rectangle_coords = get_rectangle_coords(line_point_1, line_point_2, padding=100)
    count_rectangle_coords = np.array(count_rectangle_coords, np.int32)
    count_polygon_points = count_rectangle_coords.reshape((-1, 1, 2))

    Margin_rectangle_coords = get_rectangle_coords(line_point_1, line_point_2, padding=50)
    Margin_rectangle_coords = np.array(Margin_rectangle_coords, np.int32)
    Margin_polygon_points = Margin_rectangle_coords.reshape((-1, 1, 2))

    In = point_line_side(side_point, line_point_1, line_point_2)
    In = 0.001 if In > 0 else -0.001

    # Initialize person tracker
    person_tracks = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (1440, 810))
        detections_gen = model.track(frame, conf=0.3, persist=True, classes=[0])

        # Draw the line
        cv2.line(frame, line_point_1, line_point_2, (0, 255, 0), 2)
        cv2.polylines(frame, [count_rectangle_coords], isClosed=True, color=(100, 200, 0), thickness=2)
        cv2.polylines(frame, [Margin_rectangle_coords], isClosed=True, color=(100, 200, 100), thickness=2)

        cv2.putText(frame, f"Frame_count :: {frame_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame = detections_gen[0].plot()
        for detections in detections_gen:
            if detections is not None:
                for detection, object_class, confidence in zip(detections.boxes.data.tolist(), detections.boxes.cls.tolist(), detections.boxes.conf.tolist()):
                    id = int(detection[4])
                    x1, y1, x2, y2 = map(int, detection[:4])
                    centroid_x = (x1 + x2) // 2
                    centroid_y = (y1 + y2) // 2
                    centroid = (centroid_x, centroid_y)

                    # Draw bounding box and centroid
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

                    # Check if the centroid is inside the polygon
                    inside = cv2.pointPolygonTest(count_polygon_points, (centroid_x, centroid_y), False)
                    if inside > 0:
                        count_inside = cv2.pointPolygonTest(Margin_polygon_points, (centroid_x, centroid_y), False)

                        if count_inside > 0:
                            cv2.putText(frame, f"Poly:{count_inside}", (centroid_x, centroid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            continue
                        else:
                            cv2.putText(frame, f":{count_inside}", (centroid_x, centroid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        d = point_line_side(centroid, line_point_1, line_point_2)

                        if id not in person_tracks:
                            person_tracks[id] = d
                        else:
                            d_old = person_tracks[id]
                            d_new = d
                            multiple = d_new * d_old
                            entry_status1 = enter_status (In,d_new)

                            if multiple < 0 and entry_status1 == True:
                                with lock:  # Ensure thread-safe update
                                    inside_count += 1
                                del person_tracks[id]
                                cv2.putText(frame, f"IN", (centroid_x, centroid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            elif multiple < 0 and entry_status1 == False:
                                with lock:  # Ensure thread-safe update
                                    outside_count += 1
                                del person_tracks[id]
                                cv2.putText(frame, f"OUT", (centroid_x, centroid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display the current counts
        cv2.putText(frame, f"Inside Count: {inside_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Outside Count: {outside_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        total_count = inside_count + outside_count
        
        existing_people =  0 if (inside_count - outside_count)<0 else (inside_count - outside_count)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)
    cap.release()

# Flask route to stream the video
@app.route('/video_feed')
def video_feed():
    return Response(process_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to provide count data as JSON
@app.route('/count_data')
def count_data():
    with lock:  # Ensure thread-safe access
        return jsonify({'inside_count': inside_count, 'outside_count': outside_count,'total_count':total_count, 'existing_people':existing_people})

# Custom route to serve files from the 'public' directory
@app.route('/public/<path:filename>')
def public_files(filename):
    return send_from_directory('public', filename)

# Homepage route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
