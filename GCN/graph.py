import numpy as np
import tensorflow as tf

class Graph:
    def __init__(self, num_node=33):
        self.num_node = num_node
        self.num_channel = num_node
        self.AD, self.AD2, self.bias_mat_1, self.bias_mat_2 = self.normalize_adjacency()

    def normalize_adjacency(self):
        self_link = [(i, i) for i in range(self.num_node)] 
        neighbor_1base = [
                
            (0, 1), (1, 2), (2, 3),  
            (3, 4), (4, 5), (5, 6),  
            (6, 7), (7, 8),  
            (8, 9), (9, 10),  
            (10, 11), (11, 12), 
            (12, 13), (13, 14), 
            (14, 15), (15, 16), 
            (16, 17), (17, 18),
            (18, 19), (19, 20),
            (20, 21), (21, 22), 
            (22, 23), (23, 24), 
            (24, 25),(25, 26), 
            (26, 27),(27, 28),
            (28, 29),(29, 30),
            (30, 31),(31,32)          
                        
        ]        
        
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        A = np.zeros((self.num_node, self.num_node))  # adjacency matrix
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        A2 = np.zeros((self.num_node, self.num_node))  # second-order adjacency matrix
        for root in range(A.shape[1]):
            for neighbour in range(A.shape[0]):
                if A[root, neighbour] == 1:
                    for neighbour_of_neigbour in range(A.shape[0]):
                        if A[neighbour, neighbour_of_neigbour] == 1:
                            A2[root, neighbour_of_neigbour] = 1
        
        # Bias matrices
        bias_mat_1 = np.where(A != 0, 1.0, -1e9)
        bias_mat_2 = np.where(A2 != 0, 1.0, -1e9)
              
        # Convert to float32 and TensorFlow tensors
        AD = A.astype("float32")
        AD2 = A2.astype("float32")
        bias_mat_1 = bias_mat_1.astype("float32")
        bias_mat_2 = bias_mat_2.astype("float32")
        AD = tf.convert_to_tensor(AD)
        AD2 = tf.convert_to_tensor(AD2)
        bias_mat_1 = tf.convert_to_tensor(bias_mat_1)
        bias_mat_2 = tf.convert_to_tensor(bias_mat_2)
        
        # Print statements for debugging          
        print("Shape of bias_mat_1:", bias_mat_1.shape)
        print("Shape of bias_mat_2:", bias_mat_2.shape)
        
        return AD, AD2, bias_mat_1, bias_mat_2

    def normalize(self, adjacency):
        rowsum = np.array(adjacency.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        print("Shape of r_inv:", r_inv.shape)
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = np.diag(r_inv)
        normalize_adj = r_mat_inv.dot(adjacency)
        normalize_adj = normalize_adj.astype("float32")
        normalize_adj = tf.convert_to_tensor(normalize_adj)
        print("Shape of normalize_adj:", normalize_adj.shape)
        print("normalize_adj:", normalize_adj)
        return normalize_adj























# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.metrics import accuracy_score
# from GCN.data_processing import FallDataLoader
# from GCN.graph import Graph
# from GCN.stgcn import STGCN

# random_seed = 42  # for reproducibility

# # Create the parser
# parser = argparse.ArgumentParser(description="List of arguments")

# # Add the arguments
# parser.add_argument("--ex", dest="ex", type=str, default="Kimore_ex5", help="the name of the exercise.", required=True)
# parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for optimizer.")
# parser.add_argument("--epochs", type=int, default=150, help="number of epochs to train.")
# parser.add_argument("--batch_size", type=int, default=10, help="training batch size.")
# parser.add_argument("--test_size", type=float, default=0.2, help="percentage of data to set aside for the test set.")

# # Execute the parse_args() method
# args = parser.parse_args()

# # Import the data
# data_loader = FallDataLoader(args.ex)
# graph = Graph(len(data_loader.body_part))

# # Split the data into train-validation and test sets
# train_val_x, test_x, train_val_y, test_y = train_test_split(
#     data_loader.scaled_x,
#     data_loader.scaled_y,
#     test_size=args.test_size,
#     random_state=random_seed,
#     shuffle=True,
#     stratify=data_loader.scaled_y[:, -1]
# )

# # Initialize the model
# stgcn_model = STGCN(num_joints=data_loader.num_joints,
#                     num_channels=data_loader.num_channel,
#                     num_timesteps=data_loader.num_timestep,
#                     lr=args.lr)  

# # Define the number of folds for cross-validation
# k_folds = 5
# kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

# # Store validation accuracies for each fold
# validation_accuracies = []

# for fold, (train_index, val_index) in enumerate(kf.split(train_val_x)):
#     print(f"Fold {fold + 1}/{k_folds}")

#     X_train, X_val = train_val_x[train_index], train_val_x[val_index]
#     Y_train, Y_val = train_val_y[train_index], train_val_y[val_index]

#     # Train the model
#     history = stgcn_model.train(X_train, Y_train, X_val, Y_val, epochs=args.epochs, batch_size=args.batch_size)

#     # Evaluate the model on the validation set
#     val_probs = stgcn_model.predict(X_val)
#     val_pred = (val_probs > 0.5).astype(int)
#     val_y_binary = (Y_val > 0.5).astype(int)

#     # Flatten the arrays for the entire dataset
#     val_y_binary_flat = val_y_binary.flatten()
#     val_pred_flat = val_pred.flatten()

#     # Repeat the predicted values for each element in the sequence
#     val_pred_flat_repeated = np.repeat(val_pred_flat, val_y_binary.shape[1])

#     # Calculate validation accuracy manually
#     correct_predictions = np.sum(val_y_binary_flat == val_pred_flat_repeated)
#     total_predictions = len(val_y_binary_flat)
#     val_accuracy = correct_predictions / total_predictions

#     validation_accuracies.append(val_accuracy)
#     print(f"Validation Accuracy (Fold {fold + 1}): {val_accuracy:.4f}")

# # Calculate average validation accuracy across all folds
# avg_val_accuracy = np.mean(validation_accuracies)
# print(f"Average Validation Accuracy: {avg_val_accuracy:.4f}")

# # Evaluate on the test set
# test_probs = stgcn_model.predict(test_x)
# test_pred = (test_probs > 0.5).astype(int)
# test_y_binary = (test_y > 0.5).astype(int)

# # Check the shapes of test_y_binary and test_pred
# print("Shape of test_y_binary:", test_y_binary.shape)
# print("Shape of test_pred:", test_pred.shape)


# # Flatten the arrays for the entire dataset
# test_y_binary_flat = test_y_binary.flatten()
# test_pred_flat = test_pred.flatten()

# # Repeat the predicted values for each element in the sequence
# test_pred_flat_repeated = np.repeat(test_pred_flat, test_y_binary.shape[1])

# # Calculate test accuracy manually
# correct_predictions_test = np.sum(test_y_binary_flat == test_pred_flat_repeated)
# total_predictions_test = len(test_y_binary_flat)
# test_accuracy = correct_predictions_test / total_predictions_test

# print(f"Test Accuracy: {test_accuracy:.4f}")

# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# # Evaluate on the test set
# test_probs = stgcn_model.predict(test_x)
# test_pred = (test_probs > 0.5).astype(int)
# test_y_binary = (test_y > 0.5).astype(int)

# print("Shape of test_y_binary:", test_y_binary.shape)
# print("Shape of test_pred:", test_pred.shape)


# ## Flatten the arrays to have the same shape
# test_y_binary_flat = test_y_binary.flatten()
# test_pred_flat = test_pred.flatten()
# print("Length of test_y_binary_flat:", len(test_y_binary_flat))
# print("Length of test_pred_flat:", len(test_pred_flat))


# # Reshape test_pred to match the shape of test_y_binary
# test_pred_reshaped = test_pred.flatten()

# # Calculate confusion matrix
# conf_matrix = confusion_matrix(test_y_binary_flat, test_pred_reshaped)
# print("Confusion Matrix:")
# print(conf_matrix)

# # plot confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# # Calculate precision, recall, and F1 score
# precision = precision_score(test_y_binary_flat, test_pred_reshaped)
# recall = recall_score(test_y_binary_flat, test_pred_reshaped)
# f1 = f1_score(test_y_binary_flat, test_pred_reshaped)

# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")

# # Plot confusion matrix

# import seaborn as sns

# sns.heatmap(conf_matrix, annot=True, fmt='g')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# # # Calculate average validation accuracy across folds
# # average_accuracy = np.mean(fold_accuracies)
# # print(f"Average Validation Accuracy across {num_folds} folds: {average_accuracy:.4f}")

# # # Plot training and validation accuracy for each fold
# # plt.figure(figsize=(12, 4 * num_folds))

# # for i, history in enumerate(fold_histories, 1):
# #     plt.subplot(num_folds, 1, i)
# #     plt.plot(history.history['loss'], 'orange', label=f'Training Loss - Fold {i}')
# #     plt.title(f'Training Loss - Fold {i}')
# #     plt.xlabel('Epochs')
# #     plt.ylabel('Loss')
# #     plt.legend()

# # plt.tight_layout()
# # plt.show()
