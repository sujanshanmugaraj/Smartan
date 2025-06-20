import numpy as np
from utils.angles import calculate_angle

def check_bicep_curl(landmarks, width, height):
    shoulder = [landmarks[12].x * width, landmarks[12].y * height]  # Right shoulder
    elbow    = [landmarks[14].x * width, landmarks[14].y * height]  # Right elbow
    wrist    = [landmarks[16].x * width, landmarks[16].y * height]  # Right wrist
    angle = calculate_angle(shoulder, elbow, wrist)
    status = "Correct" if 30 < angle < 160 else "Incorrect"
    return angle, status, elbow

def check_lateral_raise(landmarks, width, height):
    shoulder = [landmarks[12].x * width, landmarks[12].y * height]
    wrist    = [landmarks[16].x * width, landmarks[16].y * height]
    y_diff = shoulder[1] - wrist[1]  # Positive when arm is raised above shoulder
    status = "Correct" if y_diff > 40 else "Incorrect"
    return y_diff, status, wrist

def check_squat(landmarks, width, height):
    hip     = [landmarks[24].x * width, landmarks[24].y * height]  # Right hip
    knee    = [landmarks[26].x * width, landmarks[26].y * height]  # Right knee
    ankle   = [landmarks[28].x * width, landmarks[28].y * height]  # Right ankle
    angle = calculate_angle(hip, knee, ankle)
    status = "Correct" if angle < 120 else "Incorrect"
    return angle, status, knee

def check_jumping_jack(landmarks, width, height):
    left_hand = [landmarks[15].x * width, landmarks[15].y * height]
    right_hand = [landmarks[16].x * width, landmarks[16].y * height]
    left_foot = [landmarks[27].x * width, landmarks[27].y * height]
    right_foot = [landmarks[28].x * width, landmarks[28].y * height]

    hand_dist = np.linalg.norm(np.array(left_hand) - np.array(right_hand))
    foot_dist = np.linalg.norm(np.array(left_foot) - np.array(right_foot))
    status = "Correct" if hand_dist > 180 and foot_dist > 180 else "Incorrect"
    return (hand_dist, foot_dist), status, right_hand
