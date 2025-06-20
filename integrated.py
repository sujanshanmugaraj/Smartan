import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import json
# import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from collections import defaultdict
import base64
from io import BytesIO

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)  
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def smooth_signal(signal, window_size, poly_order):
    """Simple smoothing function"""
    if len(signal) < window_size:
        return signal
    smoothed = []
    for i in range(len(signal)):
        start = max(0, i - window_size//2)
        end = min(len(signal), i + window_size//2 + 1)
        smoothed.append(np.mean(signal[start:end]))
    return smoothed

def check_bicep_curl(landmarks, width, height):
    """Check bicep curl form"""
    shoulder = [landmarks[11].x * width, landmarks[11].y * height]
    elbow = [landmarks[13].x * width, landmarks[13].y * height]
    wrist = [landmarks[15].x * width, landmarks[15].y * height]
    
    angle = calculate_angle(shoulder, elbow, wrist)
    
    if angle > 160:
        return angle, "Correct", elbow
    elif angle < 30:
        return angle, "Too Curled", elbow
    else:
        return angle, "Correct", elbow

def check_lateral_raise(landmarks, width, height):
    """Check lateral raise based on vertical distance between wrist and shoulder"""
    shoulder_y = landmarks[11].y * height  # Left shoulder
    wrist_y = landmarks[15].y * height     # Left wrist
    y_diff = abs(shoulder_y - wrist_y)
    joint = [landmarks[15].x * width, wrist_y]
    
    # Updated threshold values
    if 100 <= y_diff <= 160:
        return y_diff, "Good", joint
    elif y_diff < 100:
        return y_diff, "Good", joint
    else:  # y_diff > 160
        return y_diff, "Too low", joint

def check_squat(landmarks, width, height):
    """Check squat form"""
    hip = [landmarks[23].x * width, landmarks[23].y * height]
    knee = [landmarks[25].x * width, landmarks[25].y * height]
    ankle = [landmarks[27].x * width, landmarks[27].y * height]
    
    angle = calculate_angle(hip, knee, ankle)
    
    if 90 <= angle <= 150:
        return angle, "Correct", knee
    elif angle > 160:
        return angle, "Too High", knee
    else:
        return angle, "Good", knee
    
def check_jumping_jack(landmarks, width, height):
    """Check jumping jack form"""
    left_wrist = [landmarks[15].x * width, landmarks[15].y * height]
    right_wrist = [landmarks[16].x * width, landmarks[16].y * height]
    left_ankle = [landmarks[27].x * width, landmarks[27].y * height]
    right_ankle = [landmarks[28].x * width, landmarks[28].y * height]
    
    hand_dist = abs(left_wrist[0] - right_wrist[0])
    foot_dist = abs(left_ankle[0] - right_ankle[0])
    
    if hand_dist > 200 and foot_dist > 100:
        return (hand_dist, foot_dist), "Good", left_wrist
    else:
        return (hand_dist, foot_dist), "Closed", left_wrist

class ExerciseAnalyzer:
    def __init__(self, exercise_type, input_path, output_dir='output_videos'):
        self.exercise_type = exercise_type
        self.input_path = input_path
        self.output_dir = output_dir
        self.output_path = os.path.join(output_dir, f'{exercise_type}_output.mp4')
        self.log_path = os.path.join(output_dir, f'{exercise_type}_log.csv')
        self.summary_path = os.path.join(output_dir, f'{exercise_type}_summary.json')
        self.graphs_dir = os.path.join(output_dir, 'graphs')
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # Exercise parameters
        self.rep_count = 0
        self.stage = "down"
        self.metric_buffer = []
        self.SMOOTH_WINDOW = 5
        
    def process_video(self):
        """Main video processing function"""
        print(f"[🚀 START] Processing {self.exercise_type} video: {self.input_path}")
        
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        log_file = open(self.log_path, mode='w', newline='')
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(['Frame', 'Timestamp', 'Metric', 'Value', 'Feedback', 'Reps'])
        
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = round(frame_idx / fps, 2)
                processed_frame = self._process_frame(frame, pose, frame_idx, timestamp, csv_writer)
                
                out.write(processed_frame)
                frame_idx += 1
                
                if frame_idx % 30 == 0:  # Every second at 30fps
                    progress = (frame_idx / frame_count) * 100
                    print(f"[⏳ PROGRESS] {progress:.1f}% - Frame {frame_idx}/{frame_count}")
        
        cap.release()
        out.release()
        log_file.close()
        
        print(f"[✅ DONE] Video processed: {self.output_path}")
        print(f"[📄 LOG] Analysis log: {self.log_path}")
        
        self._analyze_results()
        self._create_visualizations()
        
        return self._get_final_results()
    
    def _process_frame(self, frame, pose, frame_idx, timestamp, csv_writer):
        """Process individual frame"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            feedback = ""
            color = (255, 255, 255)
            metric = 0
            metric_label = ""
            
            if self.exercise_type == "bicep_curl":
                angle, status, joint = check_bicep_curl(landmarks, image.shape[1], image.shape[0])
                self.metric_buffer.append(angle)
                if len(self.metric_buffer) >= self.SMOOTH_WINDOW:
                    angle = smooth_signal(self.metric_buffer[-self.SMOOTH_WINDOW:], self.SMOOTH_WINDOW, 2)[-1]
                
                feedback = f"{status} Form"
                self._update_bicep_reps(angle)
                
                cv2.putText(image, f'Angle: {int(angle)}', (int(joint[0]), int(joint[1]) - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                metric = angle
                metric_label = "Elbow Angle"
                
            elif self.exercise_type == "lateral_raise":
                y_diff, status, joint = check_lateral_raise(landmarks, image.shape[1], image.shape[0])
                self.metric_buffer.append(y_diff)
                if len(self.metric_buffer) >= self.SMOOTH_WINDOW:
                    y_diff = smooth_signal(self.metric_buffer[-self.SMOOTH_WINDOW:], self.SMOOTH_WINDOW, 2)[-1]
                
                feedback = f"{status} Arm Raise"
                self._update_lateral_raise_reps(y_diff)
                
                cv2.putText(image, f'Y Diff: {int(y_diff)}', (int(joint[0]), int(joint[1]) - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                metric = y_diff
                metric_label = "Shoulder-Wrist Y Diff"
                
            elif self.exercise_type == "squat":
                angle, status, joint = check_squat(landmarks, image.shape[1], image.shape[0])
                self.metric_buffer.append(angle)
                if len(self.metric_buffer) >= self.SMOOTH_WINDOW:
                    angle = smooth_signal(self.metric_buffer[-self.SMOOTH_WINDOW:], self.SMOOTH_WINDOW, 2)[-1]
                
                feedback = f"{status} Squat"
                self._update_squat_reps(angle)
                
                cv2.putText(image, f'Knee Angle: {int(angle)}', (int(joint[0]), int(joint[1]) - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                metric = angle
                metric_label = "Knee Angle"
                
            elif self.exercise_type == "jumping_jack":
                (hand_dist, foot_dist), status, joint = check_jumping_jack(landmarks, image.shape[1], image.shape[0])
                self.metric_buffer.append((hand_dist + foot_dist) / 2)
                if len(self.metric_buffer) >= self.SMOOTH_WINDOW:
                    avg = smooth_signal(self.metric_buffer[-self.SMOOTH_WINDOW:], self.SMOOTH_WINDOW, 2)[-1]
                    hand_dist = foot_dist = avg
                
                feedback = f"{status} Jump"
                self._update_jumping_jack_reps(hand_dist, foot_dist)
                
                cv2.putText(image, f'Hand Dist: {int(hand_dist)}', (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                cv2.putText(image, f'Foot Dist: {int(foot_dist)}', (50, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                metric = f"H:{int(hand_dist)}|F:{int(foot_dist)}"
                metric_label = "Distances"
            
            color = (0, 255, 0) if "Correct" in feedback or "Good" in feedback else (0, 0, 255)
            cv2.putText(image, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(image, f'Reps: {self.rep_count}', (image.shape[1] - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
            
            csv_writer.writerow([frame_idx, timestamp, metric_label, metric, feedback, self.rep_count])
        
        return image
    
    def _update_bicep_reps(self, angle):
        """Update bicep curl rep count"""
        if angle > 160:
            if self.stage != "down":
                self.stage = "down"
        elif angle < 60 and self.stage == "down":
            self.stage = "up"
            self.rep_count += 1
            print(f"[✅ REP] Bicep Curl Rep: {self.rep_count}")
    
    def _update_lateral_raise_reps(self, y_diff):
        """Update lateral raise rep count"""
        if y_diff > 140:
            if self.stage != "down":
                self.stage = "down"
        elif y_diff < 100 and self.stage == "down":
            self.stage = "up"
            self.rep_count += 1
            print(f"[✅ REP] Lateral Raise Rep: {self.rep_count}")
    
    def _update_squat_reps(self, angle):
        """Update squat rep count"""
        if angle > 150:
            if self.stage != "up":
                self.stage = "up"
        elif angle < 100 and self.stage == "up":
            self.stage = "down"
            self.rep_count += 1
            print(f"[✅ REP] Squat Rep: {self.rep_count}")
    
    def _update_jumping_jack_reps(self, hand_dist, foot_dist):
        """Update jumping jack rep count"""
        if hand_dist > 200 and foot_dist > 200:
            if self.stage == "closed":
                self.stage = "open"
                self.rep_count += 1
                print(f"[✅ REP] Jumping Jack Rep: {self.rep_count}")
        else:
            self.stage = "closed"
    
    def _analyze_results(self):
        """Analyze the logged data and create summary"""
        print("[📊 ANALYSIS] Analyzing exercise data...")
        
        df = pd.read_csv(self.log_path)
        
        frame_data = []
        correct_count = 0
        rep_times = []
        rep_last = 0
        
        for _, row in df.iterrows():
            frame = int(row['Frame'])
            time = float(row['Timestamp'])
            value = row['Value']
            feedback = row['Feedback']
            reps = int(row['Reps'])
            
            if "Correct" in feedback or "Good" in feedback:
                correct_count += 1
            
            if reps > rep_last:
                rep_times.append(time)
                rep_last = reps
            
            frame_data.append({
                'frame': frame,
                'time': time,
                'value': float(value) if '|' not in str(value) else None,
                'feedback': feedback,
                'reps': reps
            })
        
        total_frames = len(frame_data)
        accuracy = (correct_count / total_frames) * 100 if total_frames else 0
       
        total_reps = rep_last
        
        numeric_values = [d['value'] for d in frame_data if d['value'] is not None]
        avg_value = round(sum(numeric_values) / len(numeric_values), 2) if numeric_values else 0
        
        rep_intervals = [round(rep_times[i+1] - rep_times[i], 2) for i in range(len(rep_times)-1)]
        avg_tempo = round(sum(rep_intervals) / len(rep_intervals), 2) if rep_intervals else 0
        
        summary = {
            "exercise_type": self.exercise_type,
            "total_frames": total_frames,
            "correct_frames": correct_count,
            "accuracy_percent": round(accuracy, 2),
            "total_reps": total_reps,
            "average_metric": avg_value,
            "average_tempo_sec_per_rep": avg_tempo,
            "rep_times": rep_times,
            "rep_intervals": rep_intervals
        }
        
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"[✅ SUMMARY] Analysis complete: {self.summary_path}")
        return summary
    
    def _create_visualizations(self):
        """Create visualization plots"""
        print("[📈 GRAPHS] Creating visualizations...")
        
        df = pd.read_csv(self.log_path)
        
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        plt.figure(figsize=(12, 6))
        
        if self.exercise_type == "jumping_jack":
            hand_values = []
            foot_values = []
            for value in df['Value']:
                if '|' in str(value):
                    parts = str(value).split('|')
                    hand_values.append(float(parts[0].replace('H:', '')))
                    foot_values.append(float(parts[1].replace('F:', '')))
                else:
                    hand_values.append(0)
                    foot_values.append(0)
            
            plt.plot(df['Timestamp'], hand_values, label='Hand Distance', alpha=0.7)
            plt.plot(df['Timestamp'], foot_values, label='Foot Distance', alpha=0.7)
            plt.ylabel("Distance (pixels)")
            plt.legend()
        else:
            plt.plot(df['Timestamp'], df['Value'], color='#667eea', linewidth=2)
            plt.ylabel(df['Metric'].iloc[0])
        
        plt.xlabel("Time (seconds)")
        plt.title(f"{self.exercise_type.replace('_', ' ').title()} - Metric Over Time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, f"{self.exercise_type}_metric_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.plot(df['Timestamp'], df['Reps'], color='#28a745', linewidth=3, marker='o', markersize=4)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Repetition Count")
        plt.title("Repetition Progress")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, f"{self.exercise_type}_reps_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        correct = df[df['Feedback'].str.contains("Correct|Good", case=False)].shape[0]
        incorrect = len(df) - correct
        
        plt.figure(figsize=(8, 8))
        colors = ['#28a745', '#dc3545']
        plt.pie([correct, incorrect], labels=["Correct Form", "Incorrect Form"], 
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title("Form Accuracy Distribution")
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, f"{self.exercise_type}_accuracy_pie.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        with open(self.summary_path, 'r') as f:
            summary = json.load(f)
        
        if len(summary['rep_intervals']) > 0:
            plt.figure(figsize=(10, 6))
            rep_numbers = list(range(1, len(summary['rep_intervals']) + 1))
            plt.bar(rep_numbers, summary['rep_intervals'], color='#764ba2', alpha=0.8)
            plt.xlabel("Repetition Number")
            plt.ylabel("Tempo (seconds)")
            plt.title("Tempo per Repetition")
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(self.graphs_dir, f"{self.exercise_type}_tempo_analysis.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"[✅ GRAPHS] Visualizations saved to: {self.graphs_dir}")
    
    def _get_final_results(self):
        """Get final analysis results"""
        with open(self.summary_path, 'r') as f:
            summary = json.load(f)
        
        summary['output_video'] = self.output_path
        summary['log_file'] = self.log_path
        summary['graphs_directory'] = self.graphs_dir
        
        return summary

def analyze_exercise_video(exercise_type, video_path, output_dir='output_videos'):
    """
    Main function to analyze exercise video
    
    Args:
        exercise_type (str): Type of exercise ('bicep_curl', 'lateral_raise', 'squat', 'jumping_jack')
        video_path (str): Path to input video
        output_dir (str): Directory for output files
    
    Returns:
        dict: Analysis results and file paths
    """
    try:
        analyzer = ExerciseAnalyzer(exercise_type, video_path, output_dir)
        results = analyzer.process_video()
        return results
    except Exception as e:
        print(f"[❌ ERROR] Analysis failed: {str(e)}")
        raise


# if __name__ == "__main__":
#     
#     exercise = "bicep_curl"  # Change as needed
#     input_video = "sample_videos/bicep_curl_4.mp4"  # Your video path
    
#     if os.path.exists(input_video):
#         results = analyze_exercise_video(exercise, input_video)
#         print("\n" + "="*50)
#         print("FINAL RESULTS:")
#         print("="*50)
#         print(json.dumps(results, indent=2))
#     else:
#         print(f"[❌ ERROR] Video file not found: {input_video}")
#         print("Please update the 'input_video' path to point to your video file.")