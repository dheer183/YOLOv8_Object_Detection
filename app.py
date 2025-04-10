from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLO
import mediapipe as mp
from collections import Counter
import os
import signal
import time

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
last_update_time = 0

def interpret_pose(landmarks):
    gestures = []
    rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    le = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
    re = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
    la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    if rw.y < rs.y:
        gestures.append("raising right hand")
    if lw.y < ls.y:
        gestures.append("raising left hand")

    hip_avg_y = (rh.y + lh.y) / 2
    knee_avg_y = (rk.y + lk.y) / 2
    ankle_avg_y = (la.y + ra.y) / 2
    if abs(hip_avg_y - knee_avg_y) < 0.08 and hip_avg_y < ankle_avg_y:
        gestures.append("sitting")
    elif hip_avg_y > ankle_avg_y:
        gestures.append("standing")

    if le.x < nose.x and re.x < nose.x:
        gestures.append("looking right")
    elif le.x > nose.x and re.x > nose.x:
        gestures.append("looking left")
    else:
        gestures.append("facing forward")

    knee_movement = abs(rk.y - lk.y)
    if knee_movement > 0.15:
        gestures.append("walking")

    return gestures


def generate_story(object_counts, gestures):
    parts = []
    for obj, count in object_counts.items():
        if count == 1:
            parts.append(f"one {obj}")
        else:
            parts.append(f"{count} {obj}s")

    if gestures:
        parts.append("someone is " + ", and ".join(gestures))

    if not parts:
        return "No significant activity detected."

    return "In the scene, " + ", and ".join(parts) + "."


def generate_frames():
    global latest_description, last_update_time

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame, verbose=False)[0]
        object_labels = []
        gestures = []

        pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = pose_results.pose_landmarks.landmark
            gestures = interpret_pose(landmarks)

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results.names[int(box.cls[0])]
            object_labels.append(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        object_counts = Counter(object_labels)
        current_time = time.time()
        if current_time - last_update_time > 3:
            latest_description = generate_story(object_counts, gestures)
            last_update_time = current_time

        cv2.putText(frame, latest_description, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
    global camera
    if camera:
        camera.release()
    os.kill(os.getpid(), signal.SIGTERM)
    return 'Server shutting down...'

if __name__ == "__main__":
    app.run(debug=True)