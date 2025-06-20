import pandas as pd
import matplotlib.pyplot as plt
import os

exercise_type = "bicep_curl"  
log_path = f"output_videos/bicep_curl_log.csv"
save_dir = "output_videos/graphs"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(log_path)

plt.figure(figsize=(10, 5))
plt.plot(df['Timestamp'], df['Value'], label=df['Metric'][0])
plt.xlabel("Time (s)")
plt.ylabel(df['Metric'][0])
plt.title(f"{df['Metric'][0]} Over Time")
plt.grid(True)
plt.savefig(os.path.join(save_dir, f"{exercise_type}_metric_plot.png"))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df['Timestamp'], df['Reps'], label="Reps", color="orange")
plt.xlabel("Time (s)")
plt.ylabel("Repetition Count")
plt.title("Repetition Count Over Time")
plt.grid(True)
plt.savefig(os.path.join(save_dir, f"{exercise_type}_reps_plot.png"))
plt.close()

correct = df[df['Feedback'].str.contains("Correct", case=False)].shape[0]
incorrect = df[df['Feedback'].str.contains("Incorrect", case=False)].shape[0]

plt.figure(figsize=(6, 6))
plt.pie([correct, incorrect], labels=["Correct", "Incorrect"], autopct='%1.1f%%', colors=["green", "red"])
plt.title("Form Accuracy")
plt.savefig(os.path.join(save_dir, f"{exercise_type}_accuracy_pie.png"))
plt.close()

print(f"[✅ Graphs] Saved to: {save_dir}")
