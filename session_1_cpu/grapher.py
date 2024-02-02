import json
import matplotlib.pyplot as plt

# Load data from CPU and GPU training
cpu_data = json.load(open("cpu_training.json", 'r'))
# gpu_data = json.load(open("gpu_training.json", 'r'))

# Extract metrics
cpu_train_loss = cpu_data["train_loss"]
cpu_train_acc = cpu_data["train_acc"]
# gpu_train_loss = gpu_data["train_loss"]
# gpu_train_acc = gpu_data["train_acc"]

# Number of epochs
num_epochs = len(cpu_train_loss)

# Create x-axis values (epochs)
epochs = range(1, num_epochs + 1)

# Create subplots for loss and accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot training loss
ax1.plot(epochs, cpu_train_loss, label='CPU', marker='o')
# ax1.plot(epochs, gpu_train_loss, label='GPU', marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss Comparison')
ax1.legend()

# Plot training accuracy
ax2.plot(epochs, cpu_train_acc, label='CPU', marker='o')
# ax2.plot(epochs, gpu_train_acc, label='GPU', marker='o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Training Accuracy (%)')
ax2.set_title('Training Accuracy Comparison')
ax2.legend()

# Display the plots
plt.tight_layout()
plt.show()

fig.savefig('plots.jpg', dpi=200)
