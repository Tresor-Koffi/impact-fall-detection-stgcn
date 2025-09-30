import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from GCN.data_processing import FallDataLoader
from GCN.graph import Graph
from GCN.stgcngru_bilstm import Stgcn_gru_biLstm
import matplotlib.pyplot as plt
import tensorflow as tf

random_seed = 42  # for reproducibility

# Function to print GPU memory usage
def print_gpu_memory_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for i, gpu in enumerate(gpus):
            # Construct the device name in the correct format (e.g., "GPU:0")
            device_name = f"GPU:{i}"
            try:
                # Get memory info for the GPU
                mem_info = tf.config.experimental.get_memory_info(device_name)
                print(f"GPU: {device_name}")
                print(f"  Current VRAM usage: {mem_info['current'] / 1024**2:.2f} MB")
                print(f"  Peak VRAM usage: {mem_info['peak'] / 1024**2:.2f} MB")
            except Exception as e:
                print(f"Error getting memory info for {device_name}: {e}")
    else:
        print("No GPU devices found.")

# Example usage
print("Initial VRAM Usage:")
print_gpu_memory_usage()
# Create the parser
parser = argparse.ArgumentParser(description="List of arguments")

# Add the arguments
parser.add_argument("--ex", dest="ex", type=str, default="Kimore_ex5", help="the type of fall.", required=True)
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for optimizer.")
parser.add_argument("--epochs", type=int, default=150, help="number of epochs to train.")
parser.add_argument("--batch_size", type=int, default=10, help="training batch size.")
parser.add_argument("--val_split", type=float, default=0.2, help="validation split ratio.")

# Execute the parse_args() method
args = parser.parse_args()

# Import the data
data_loader = FallDataLoader(args.ex)
graph = Graph(len(data_loader.body_part))

# Check that class counts are sufficient for stratified splitting
# Check that class counts are sufficient for stratified splitting
class_counts = np.bincount(data_loader.scaled_y[:, -1].astype(int))
print("Class Distribution:", class_counts)

if all(count >= 2 for count in class_counts):
    # Perform stratified split
    X_train, val_x, Y_train, val_y = train_test_split(
        data_loader.scaled_x,
        data_loader.scaled_y,
        test_size=args.val_split,
        random_state=random_seed,
        shuffle=True,
        stratify=data_loader.scaled_y[:, -1]
    )
else:
    print("Class counts are insufficient for stratified split. Performing random split.")
    # Perform random split without stratification
    X_train, val_x, Y_train, val_y = train_test_split(
        data_loader.scaled_x,
        data_loader.scaled_y,
        test_size=args.val_split,
        random_state=random_seed,
        shuffle=True
    )

# Initialize the model
stgcngru_bilstm_model = Stgcn_gru_biLstm(
    num_joints=data_loader.num_joints,
    num_channels=data_loader.num_channel,
    num_timesteps=data_loader.num_timestep,
    lr=args.lr
)

# Print initial VRAM usage
print("Initial VRAM Usage:")
print_gpu_memory_usage()

# Train the model
history = stgcngru_bilstm_model.train(
    X_train,
    Y_train,
    val_x,
    val_y,
    epochs=args.epochs,
    batch_size=args.batch_size
)

# Print VRAM usage after training
print("VRAM Usage After Training:")
print_gpu_memory_usage()

# Evaluate the model
val_probs = stgcngru_bilstm_model.predict(val_x)
val_pred = (val_probs > 0.5).astype(int)
val_y_binary = (val_y > 0.5).astype(int)

# Flatten the arrays for the entire dataset
val_y_binary_flat = val_y_binary.flatten()
val_pred_flat = val_pred.flatten()

# Calculate validation accuracy using sklearn
val_accuracy = accuracy_score(val_y_binary_flat, val_pred_flat)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Print VRAM usage after prediction
print("VRAM Usage After Prediction:")
print_gpu_memory_usage()

# Plot training and validation loss and accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'orange', label='Training Loss')
plt.plot(history.history['val_loss'], 'blue', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'orange', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()