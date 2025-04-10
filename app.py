from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import mediapipe as mp

app = Flask(__name__)

model = YOLO('yolov8n.pt')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

camera = cv2.VideoCapture(1)
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            results = model(frame, verbose=False)[0]

            # Pose detection
            pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Draw YOLO detections
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = results.names[int(box.cls[0])]
                confidence = box.conf[0]

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0),2)

            # Mediapipe annotations and simple gesture description
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                landmarks = pose_results.pose_landmarks.landmark
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                gesture = "Standing"
                if right_wrist.y < right_shoulder.y:
                    gesture = "Waving Right Hand"

                cv2.putText(frame, gesture, (30,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
