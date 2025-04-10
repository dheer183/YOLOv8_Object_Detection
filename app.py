from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLO
import mediapipe as mp
from collections import Counter
import os
import signal

app = Flask(__name__)

# Load YOLO model
model = YOLO('yolov8n.pt')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize camera
camera = cv2.VideoCapture(1)
latest_description = "Waiting for input..."

def generate_story(object_counts, gesture=None):
    parts = []

    for obj, count in object_counts.items():
        if count == 1:
            parts.append(f"one {obj}")
        else:
            parts.append(f"{count} {obj}s")

    if gesture:
        parts.append(gesture)

    if not parts:
        return "No significant activity detected."
    
    return "In the scene, " + ", and ".join(parts) + "."

def generate_frames():
    global latest_description

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame, verbose=False)[0]
        object_labels = []

        # Pose detection
        pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gesture = None

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = pose_results.pose_landmarks.landmark
            rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            if rw.y < rs.y:
                gesture = "someone is waving"

        # Draw YOLO detections
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results.names[int(box.cls[0])]
            object_labels.append(label)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        object_counts = Counter(object_labels)
        latest_description = generate_story(object_counts, gesture)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/description')
def get_description():
    return jsonify({"description": latest_description})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    camera.release()
    os.kill(os.getpid(), signal.SIGINT)
    return 'Server shutting down...'

if __name__ == "__main__":
    app.run(debug=True)
