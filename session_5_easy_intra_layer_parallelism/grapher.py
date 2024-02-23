import json
import numpy as np
import matplotlib.pyplot as plt

# Load data from CPU training
cpu_data = json.load(open("cpu_training.json", 'r'))

# Extract metrics
iter_times = cpu_data["iter_times"]
iter_losses = cpu_data["iter_losses"]

# Number of epochs and iterations per epoch
num_epochs = len(iter_times)
iterations_per_epoch = 2000

# Averaging every 200 iterations
average_interval = 20

# Create subplots for time and loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot iteration times for each epoch
for epoch in range(num_epochs):
    times = iter_times[epoch]
    avg_times = [np.mean(times[i:i + average_interval]) for i in range(0, len(times), average_interval)]
    epoch_numbers = [epoch + i / iterations_per_epoch for i in range(0, len(times), average_interval)]
    ax1.plot(epoch_numbers, avg_times, label=f'Epoch {epoch + 1}')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Average Time (seconds)')
ax1.set_title(f'Average Iteration Time Comparison (Averaged every {average_interval} iterations)')
ax1.legend()

# Plot iteration losses for each epoch
for epoch in range(num_epochs):
    losses = iter_losses[epoch]
    avg_losses = [np.mean(losses[i:i + average_interval]) for i in range(0, len(losses), average_interval)]
    epoch_numbers = [epoch + i / iterations_per_epoch for i in range(0, len(losses), average_interval)]
    ax2.plot(epoch_numbers, avg_losses, label=f'Epoch {epoch + 1}')

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Average Loss')
ax2.set_title(f'Average Iteration Loss Comparison (Averaged every {average_interval} iterations)')
ax2.legend()

# Display the plots
plt.tight_layout()
plt.show()

fig.savefig('plots.jpg', dpi=200)

