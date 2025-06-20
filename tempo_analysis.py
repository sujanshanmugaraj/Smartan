import pandas as pd
import matplotlib.pyplot as plt
import os

exercise_type = "bicep_curl"
log_path = f"output_videos/bicep_curl_log.csv"
save_path = f"output_videos/{exercise_type}_rep_tempo.png"

df = pd.read_csv(log_path)

rep_times = []
prev_rep = -1
for i, row in df.iterrows():
    if row['Reps'] > prev_rep:
        rep_times.append(row['Timestamp'])
        prev_rep = row['Reps']

tempos = [round(rep_times[i+1] - rep_times[i], 2) for i in range(len(rep_times)-1)]
rep_ids = list(range(1, len(tempos)+1))

plt.figure(figsize=(8, 5))
plt.plot(rep_ids, tempos, marker='o', linestyle='-')
plt.title(f"{exercise_type.title()} - Tempo per Rep")
plt.xlabel("Repetition Number")
plt.ylabel("Tempo (sec)")
plt.grid(True)
plt.savefig(save_path)
plt.close()

print(f"[✅ Tempo Analysis] Saved to: {save_path}")
