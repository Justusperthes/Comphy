import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# Progress data (values between 0 and 1 to represent partial to full completion)
progress = [
    [0.8, 0.8, 1, 0, 0], [0.75, 0.5, 1, 1, 0], [0.8, 1, 1, 1, 1, 0, 0], 
    [1, 1, 1, 0.33333, 0, 0], [1, 0.75, 0, 0, 0], [1, 0.8, 0, 0, 0, 0, 0], 
    [0.75, 0.33, 0.25, 0, 0], [1, 1, 1, 1, 1, 1, 0], [0.9, 1, 1], [0.9, 1, 1, 0.5],
    [1, 0.4, 0.6, 0], [1, 1, 5.5/6], [4/9, 0.7], [0,0,0,0] 
]

exercises_per_week = []
for i in range(len(progress)):
    exercises_per_week.append(len(progress[i]))

# Adjust the size of the array for plotting
max_exercises = max(exercises_per_week)
progress_array = np.full((len(exercises_per_week), max_exercises), np.nan)  # Fill with NaN to distinguish empty cells

for i, week in enumerate(progress):
    progress_array[i, :len(week)] = week

# Use a continuous colormap
cmap = plt.cm.RdYlGn  # Choose a smooth gradient colormap
norm = Normalize(vmin=0, vmax=1)

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
cax = ax.imshow(progress_array, cmap=cmap, norm=norm, aspect='auto')

# Color bar to show the progress scale
cbar = fig.colorbar(cax, ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(['Incomplete', 'Partially complete', 'Complete'])

# Remove x-axis labels
ax.set_xticks([])
ax.set_yticks(np.arange(len(exercises_per_week)))
ax.set_yticklabels([f'Week {i}' for i in range(1, len(exercises_per_week) + 1)])

# Label each cell with continuous exercise number
exercise_counter = 1
for i in range(len(exercises_per_week)):
    for j in range(exercises_per_week[i]):
        if not np.isnan(progress_array[i, j]):
            ax.text(j, i, f'Ex{exercise_counter}', ha='center', va='center', 
                    color='white' if progress_array[i, j] >= 0.5 else 'black')
            exercise_counter += 1

plt.title('Computational Physics Progress')
plt.ylabel('Weeks')

plt.show()