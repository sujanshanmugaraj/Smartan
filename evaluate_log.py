import csv
import json
import matplotlib.pyplot as plt
from collections import defaultdict

log_path = 'output_videos/bicep_curl_log.csv'
summary_path = log_path.replace('.csv', '_summary.json')

frame_data = []
correct_count = 0
rep_last = 0
rep_times = []

with open(log_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
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
            'value': float(value) if '|' not in value else None,  
            'feedback': feedback,
            'reps': reps
        })

total_frames = len(frame_data)
accuracy = (correct_count / total_frames) * 100 if total_frames else 0
total_reps = rep_last
avg_value = round(
    sum(d['value'] for d in frame_data if d['value'] is not None) / total_frames, 2
) if total_frames else 0

rep_intervals = [round(rep_times[i+1] - rep_times[i], 2) for i in range(len(rep_times)-1)]
avg_tempo = round(sum(rep_intervals)/len(rep_intervals), 2) if rep_intervals else 0

summary = {
    "total_frames": total_frames,
    "correct_frames": correct_count,
    "accuracy_percent": round(accuracy, 2),
    "total_reps": total_reps,
    "average_metric": avg_value,
    "average_tempo_sec_per_rep": avg_tempo
}
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=4)

print(f"[✅ Summary] saved to: {summary_path}")
print(json.dumps(summary, indent=4))

times = [d['time'] for d in frame_data]
values = [d['value'] for d in frame_data]
feedbacks = [d['feedback'] for d in frame_data]

plt.figure(figsize=(12, 6))
plt.plot(times, values, label='Metric Value', color='blue')
plt.scatter(
    [t for t, fb in zip(times, feedbacks) if "Correct" in fb],
    [v for v, fb in zip(values, feedbacks) if "Correct" in fb],
    color='green', label='Correct'
)
plt.scatter(
    [t for t, fb in zip(times, feedbacks) if "Incorrect" in fb],
    [v for v, fb in zip(values, feedbacks) if "Incorrect" in fb],
    color='red', label='Incorrect'
)
plt.title("Form Metric Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
