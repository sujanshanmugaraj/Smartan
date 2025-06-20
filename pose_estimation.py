import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from datetime import datetime
from utils.angles import calculate_angle
from utils.rules import check_bicep_curl, check_lateral_raise, check_squat, check_jumping_jack
from utils.filters import smooth_signal

exercise_type = "bicep_curl"  
input_path = f'sample_videos/jumping_jacks_1.mp4'
output_path = f'output_videos/multiii_output.mp4'
log_path = f'output_videos/{exercise_type}_log.csv'
SMOOTH_WINDOW = 5

os.makedirs('output_videos', exist_ok=True)
if not os.path.exists(input_path):
    print(f"[❌ ERROR] Input video not found at: {input_path}")
    exit()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"[❌ ERROR] Cannot open video: {input_path}")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

log_file = open(log_path, mode='w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(['Frame', 'Timestamp', 'Metric', 'Value', 'Feedback', 'Reps'])

rep_count = 0
stage = "down"
metric_buffer = []

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = round(frame_idx / fps, 2)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            feedback = ""
            color = (255, 255, 255)

            if exercise_type == "bicep_curl":
                angle, status, joint = check_bicep_curl(landmarks, width, height)
                metric_buffer.append(angle)
                if len(metric_buffer) >= SMOOTH_WINDOW:
                    angle = smooth_signal(metric_buffer[-SMOOTH_WINDOW:], SMOOTH_WINDOW, 2)[-1]
                feedback = f"{status} Form"
                if angle > 160:
                    if stage != "down":
                        stage = "down"
                elif angle < 60 and stage == "down":
                    stage = "up"
                    rep_count += 1
                    print(f"[✅ REP] Bicep Curl Reps: {rep_count}")
                print(f"[📏 ANGLE] Frame {frame_idx}, Elbow Angle: {angle:.2f}, Feedback: {feedback}")
                cv2.putText(image, f'Angle: {int(angle)}', (int(joint[0]), int(joint[1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                metric = angle
                metric_label = "Elbow Angle"

            elif exercise_type == "lateral_raise":
                y_diff, status, joint = check_lateral_raise(landmarks, width, height)
                metric_buffer.append(y_diff)
                if len(metric_buffer) >= SMOOTH_WINDOW:
                    y_diff = smooth_signal(metric_buffer[-SMOOTH_WINDOW:], SMOOTH_WINDOW, 2)[-1]
                feedback = f"{status} Arm Height"
                print(f"[📏 DIFF] Frame {frame_idx}, Y-Diff: {y_diff:.2f}, Feedback: {feedback}")
                cv2.putText(image, f'Y Diff: {int(y_diff)}', (int(joint[0]), int(joint[1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                metric = y_diff
                metric_label = "Shoulder-Wrist Y Diff"

            elif exercise_type == "squat":
                angle, status, joint = check_squat(landmarks, width, height)
                metric_buffer.append(angle)
                if len(metric_buffer) >= SMOOTH_WINDOW:
                    angle = smooth_signal(metric_buffer[-SMOOTH_WINDOW:], SMOOTH_WINDOW, 2)[-1]
                feedback = f"{status} Squat"
                if angle > 150:
                    if stage != "up":
                        stage = "up"
                elif angle < 100 and stage == "up":
                    stage = "down"
                    rep_count += 1
                    print(f"[✅ REP] Squat Reps: {rep_count}")
                print(f"[📏 ANGLE] Frame {frame_idx}, Knee Angle: {angle:.2f}, Feedback: {feedback}")
                cv2.putText(image, f'Knee Angle: {int(angle)}', (int(joint[0]), int(joint[1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                metric = angle
                metric_label = "Knee Angle"

            elif exercise_type == "jumping_jack":
                (hand_dist, foot_dist), status, joint = check_jumping_jack(landmarks, width, height)
                metric_buffer.append((hand_dist + foot_dist) / 2)
                if len(metric_buffer) >= SMOOTH_WINDOW:
                    avg = smooth_signal(metric_buffer[-SMOOTH_WINDOW:], SMOOTH_WINDOW, 2)[-1]
                    hand_dist = foot_dist = avg
                feedback = f"{status} Jump"
                if hand_dist > 200 and foot_dist > 200:
                    if stage == "closed":
                        stage = "open"
                        rep_count += 1
                        print(f"[✅ REP] Jumping Jack Reps: {rep_count}")
                else:
                    stage = "closed"
                print(f"[📏 DIST] Frame {frame_idx}, Hand: {hand_dist:.2f}, Foot: {foot_dist:.2f}, Feedback: {feedback}")
                cv2.putText(image, f'Hand Dist: {int(hand_dist)}', (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                cv2.putText(image, f'Foot Dist: {int(foot_dist)}', (50, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                metric = f"H:{int(hand_dist)}|F:{int(foot_dist)}"
                metric_label = "Distances"

            color = (0, 255, 0) if "Correct" in feedback or "Good" in feedback else (0, 0, 255)
            cv2.putText(image, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(image, f'Reps: {rep_count}', (width - 200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            csv_writer.writerow([frame_idx, timestamp, metric_label, metric, feedback, rep_count])

        out.write(image)
        frame_idx += 1

cap.release()
out.release()
log_file.close()
print(f"[✅ DONE] Output video saved to: {output_path}")
print(f"[📄 LOG] Feedback log saved to: {log_path}")



