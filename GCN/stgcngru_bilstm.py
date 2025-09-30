import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Input, LSTM, concatenate, ConvLSTM2D, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from .graph import Graph  # Assuming 'graph.py' is in the same directory as this script
import os

class Stgcn_gru_biLstm:

    def __init__(self, num_joints=33, num_channels=3, num_timesteps=100, lr=0.001):
        # Initialization
        self.num_joints = num_joints
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.lr = lr
        self.graph = Graph(num_node=33)
        self.model = self.build_model()

    def sgcn(self, Input):
        """Temporal convolution and Graph Convolution"""
        k1 = tf.keras.layers.Conv2D(64, (9, 1), padding="same", activation="sigmoid")(Input)
        k = concatenate([Input, k1], axis=-1)

        # First hop localization
        x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, activation="relu")(k)
        expand_x1 = tf.expand_dims(x1, axis=3)
        f_1 = ConvLSTM2D(filters=33, kernel_size=(1, 1), input_shape=(None, None, 33, 1, 3), return_sequences=True)(expand_x1)
        f_1 = f_1[:, :, :, 0, :]
        logits = f_1
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.graph.bias_mat_1)  # Adjacency matrix
        gcn_x1 = tf.keras.layers.Lambda(lambda x: tf.einsum("ntvw,ntwc->ntvc", x[0], x[1]))([coefs, x1])

        # Second hop localization
        y1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, activation="relu")(k)
        expand_y1 = tf.expand_dims(y1, axis=3)
        f_2 = ConvLSTM2D(filters=33, kernel_size=(1, 1), input_shape=(None, None, 33, 3), return_sequences=True)(expand_y1)
        f_2 = f_2[:, :, :, 0, :]
        logits = f_2
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.graph.bias_mat_2)
        gcn_y1 = tf.keras.layers.Lambda(lambda x: tf.einsum("ntvw,ntwc->ntvc", x[0], x[1]))([coefs, y1])

        gcn_1 = concatenate([gcn_x1, gcn_y1], axis=-1)

        # Temporal convolution
        z1 = tf.keras.layers.Conv2D(16, (9, 1), padding="same", activation="relu")(gcn_1)
        z1 = Dropout(0.25)(z1)
        z2 = tf.keras.layers.Conv2D(16, (15, 1), padding="same", activation="relu")(z1)
        z2 = Dropout(0.25)(z2)
        z3 = tf.keras.layers.Conv2D(16, (20, 1), padding="same", activation="relu")(z2)
        z3 = Dropout(0.25)(z3)
        z = concatenate([z1, z2, z3], axis=-1)
        return z
    
    def bilstm_gru(self, x):
        # Fixed reshape to work with dynamic shapes
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        feature_dim = x.shape[2] * x.shape[3]  # 33 * 48 = 1584
        
        x = tf.keras.layers.Reshape(target_shape=(-1, feature_dim))(x)
        
        rec = tf.keras.layers.Bidirectional(LSTM(80, return_sequences=True))(x)
        rec = Dropout(0.25)(rec)
        rec = GRU(40, return_sequences=True)(rec)
        rec = Dropout(0.25)(rec)
        rec = tf.keras.layers.Bidirectional(LSTM(40, return_sequences=True))(rec)
        rec = Dropout(0.25)(rec)
        rec = GRU(80)(rec)  # This outputs (batch_size, 80)
        rec = Dropout(0.25)(rec)
        
        # Single output for sequence classification
        out = Dense(1, activation="sigmoid")(rec)
        return out    
    
    def build_model(self):
        # Build the complete model
        seq_input = Input(shape=(None, self.num_joints, self.num_channels), batch_size=None)
        x = self.sgcn(seq_input)
        y = x + self.sgcn(x)
        z = y + self.sgcn(y)
        out = self.bilstm_gru(z)
        model = Model(seq_input, out)
        model.summary()
        return model
    
    def train(self, train_x, train_y, val_x, val_y, epochs, batch_size):
        """
        Fixed training method that uses the instance model
        """
        print("=== TRAINING CONFIGURATION ===")
        print(f"Train X shape: {train_x.shape}")
        print(f"Train Y shape: {train_y.shape}")
        print(f"Val X shape: {val_x.shape}")
        print(f"Val Y shape: {val_y.shape}")
        
        # Process labels for sequence classification
        # Take the last timestep or majority vote
        if len(train_y.shape) > 1 and train_y.shape[1] > 1:
            # If labels are per timestep, take the last timestep
            train_y_processed = train_y[:, -1:]  # Shape: (batch_size, 1) - FIXED: was -1:1
            val_y_processed = val_y[:, -1:]      # Shape: (batch_size, 1) - FIXED: was -1:1
            print(f"Processed Train Y shape: {train_y_processed.shape}")
            print(f"Processed Val Y shape: {val_y_processed.shape}")
        else:
            train_y_processed = train_y
            val_y_processed = val_y
        
        # Compile the existing model (DON'T create a new one!)
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),  # Better for binary classification
            optimizer=Adam(learning_rate=self.lr),
            metrics=['accuracy']
        )

        # Enhanced callbacks
        callbacks = [
            ModelCheckpoint(
                "best_stgcn_gru_bilstmmodel.hdf5", 
                monitor='val_loss', 
                save_best_only=True,
                mode='min',
                save_freq='epoch',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("=== STARTING TRAINING ===")
        
        # Train the EXISTING model
        history = self.model.fit(
            train_x,
            train_y_processed,
            validation_data=(val_x, val_y_processed),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("=== TRAINING COMPLETED ===")
        return history, train_y_processed, val_y_processed

    def predict(self, data):
        return self.model.predict(data)

    def save_model_to_json(self, json_file_path):
        # Save the model architecture to JSON file
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        model_json = self.model.to_json()
        with open(json_file_path, "w") as json_file:
            json_file.write(model_json)
        print(f"Model architecture saved to {json_file_path}")

    def save_model_weights(self, weight_file_path):
        # Save model weights
        os.makedirs(os.path.dirname(weight_file_path), exist_ok=True)
        self.model.save_weights(weight_file_path)
        print(f"Model weights saved to {weight_file_path}")

    def load_model_weights(self, weight_file_path):
        # Load model weights
        self.model.load_weights(weight_file_path)
        print(f"Model weights loaded from {weight_file_path}")

    def save_model(self, model_path):
        """Save the complete model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"Complete model saved to {model_path}")


# Example usage with corrected instantiation
if __name__ == "__main__":
    stgcn_gru_bilstm_model = Stgcn_gru_biLstm(num_joints=33, num_channels=3, num_timesteps=100)

    # Save the model architecture to the specified JSON file
    json_file_path = "C:/Users/ytkoffi/Desktop/sketon/STGCNGRU_BILSTM/stgcn_gru_bilstm_model.json"
    stgcn_gru_bilstm_model.save_model_to_json(json_file_path)























#  ##########################################################

#                 #STGCNGRU_BiLSTM

# ##########################################################
# from keras.layers import Lambda

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, Input, Conv2D, Concatenate, TimeDistributed, Flatten, Reshape
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import GRU
# from sklearn.preprocessing import StandardScaler
# from .graph import Graph
# import os
# from keras.layers import Reshape

# class Stgcn_gru_biLstm:

#     def __init__(self, num_joints=33, num_channels=3, num_timesteps=100, lr=0.001):
#         # Initialization
#         self.num_joints = num_joints
#         self.num_channels = num_channels
#         self.num_timesteps = num_timesteps
#         self.lr = lr
#         self.graph = Graph(num_node=33)
#         self.model = self.build_model()

#     def Bi_Lstm(self, x):
#         print("================X================")
#         print(x.shape)
#         x = tf.keras.layers.Reshape(target_shape=(1, x.shape[2] * x.shape[3]))(x)
#         rec = tf.keras.layers.Bidirectional(LSTM(80, return_sequences=True))(x)

#         # rec = Bidirectional(LSTM(80, return_sequences=True), merge_mode='sum')(x)
#         rec = Dropout(0.25)(rec) 
#         print("================rec================")
#         print(rec.shape)
#         rec1 = tf.keras.layers.Bidirectional(LSTM(40, return_sequences=True), merge_mode='sum')(rec)
#         rec1 = Dropout(0.25)(rec1) 
#         print("================rec1================")
#         print(rec1.shape)
#         rec2 = Bidirectional(LSTM(40, return_sequences=True), merge_mode='sum')(rec1)
#         rec2 = Dropout(0.25)(rec2) 
#         print("================rec2================")
#         print(rec2.shape)
#         rec3 = Bidirectional(LSTM(80), merge_mode='sum')(rec2)
#         rec3 = Dropout(0.25)(rec3) 
#         print("=========================rec3========================")
#         print(rec3.shape)
#         out = Dense(1, activation='linear')(rec3)
#         return out

#     def sgcn_gru(self, Input):
#         k1 = Conv2D(64, (9,1), padding='same', activation='relu')(Input)
#         k = Concatenate()([Input, k1])

#         x1 = Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu')(k)
#         x1 = Conv2D(filters=33, kernel_size=(1,1), strides=1, activation='relu')(x1)
#         print("Shape of x1 before reshape:", x1.shape)  # Add this line
        
#         # # x_dim = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0] * tf.shape(x)[1], -1)))(x1)
#         # x_dim = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[2], tf.shape(x)[3])))(x1)
#         # x_dim = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], self.num_timesteps, -1)))(x1)
#         x_dim = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[2], tf.shape(x)[3])))(x1)
        


#         print("Shape of x_dim after reshape:", x_dim.shape)
        

#         f_1 = GRU(33, return_sequences=True)(x_dim)
#         print("Shape of f_1 after GRU:", f_1.shape)
#         # Reshape the output to match expected dimensions
#         f_1_reshaped = Reshape((-1, 33, 33))(f_1)
#         print("Shape of f_1_reshaped:", f_1_reshaped.shape)
#         f_1 = tf.expand_dims(f_1, axis=3)
#         logits = f_1
#         coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.graph.bias_mat_1)

#         gcn_x1 = Lambda(lambda x: tf.einsum("ntvw,ntwc->ntvc", x[0], x[1]))([coefs, x1])
#         gcn_x1_reshaped = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[2] * tf.shape(x)[3])))(gcn_x1)

#         y1 = Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu')(k)
#         y1 = Conv2D(filters=33, kernel_size=(1,1), strides=1, activation='relu')(y1)
#         y_dim = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[0] * tf.shape(x)[3])))(y1)
        
#         # y_dim = Reshape((-1, 33))(y_dim)
#         # y_dim_reshaped = Reshape((-1, 1089))(y_dim)
#         print("Shape of y_dim before GRU:", y_dim.shape)
              
#         f_2 = GRU(33, return_sequences=True)(y_dim)
        
#         f_2 = tf.expand_dims(f_2, axis=3)
#         logits = f_2
#         coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.graph.bias_mat_2)

#         gcn_y1 = Lambda(lambda x: tf.einsum("ntvw,ntwc->ntvc", x[0], x[1]))([coefs, y1])
#         gcn_y1_reshaped = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[2] * tf.shape(x)[3])))(gcn_y1)

#         gcn_1 = Concatenate()([gcn_x1_reshaped, gcn_y1_reshaped])

#         z1 = Conv2D(16, (9,1), padding='same', activation='relu')(gcn_1)
#         z1 = Conv2D(16, (9,1), padding='same', activation='relu')(y_dim)
#         z1 = Dropout(0.25)(z1)
#         z2 = Conv2D(16, (15,1), padding='same', activation='relu')(z1)
#         z2 = Dropout(0.25)(z2)
#         z3 = Conv2D(16, (20,1), padding='same', activation='relu')(z2)
#         z3 = Dropout(0.25)(z3)
#         z = Concatenate()([z1, z2, z3])

#         return z


#     def build_model(self):
#         seq_input = Input(shape=(None, self.num_joints, self.num_channels), batch_size=None)
#         z = self.sgcn_gru(seq_input) 
#         out = self.Bi_Lstm(z) 
#         model = Model(seq_input, out)
#         return model

#     def save_model_to_json(self, json_file_path):
#         os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
#         model_json = self.model.to_json()
#         with open(json_file_path, "w") as json_file:
#             json_file.write(model_json)
#         print(f"Model architecture saved to {json_file_path}")

#     def save_model_weights(self, weights_file_path):
#         os.makedirs(os.path.dirname(weights_file_path), exist_ok=True)
#         self.model.save_weights(weights_file_path)
#         print(f"Model weights saved to {weights_file_path}")

#     def compile_model(self):
#         optimizer = Adam(lr=self.lr)
#         self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
#         print("Model compiled successfully")

#     def load_model(self, json_file_path, weights_file_path):
#         json_file = open(json_file_path, 'r')
#         loaded_model_json = json_file.read()
#         json_file.close()
#         self.model = tf.keras.models.model_from_json(loaded_model_json)
#         self.model.load_weights(weights_file_path)
#         print("Model loaded successfully")


#     def train(self, train_x, train_y, val_x, val_y, epochs, batch_size):
#         model = self.build_model()
#         model.compile(
#             loss=tf.keras.losses.Huber(delta=0.1),
#             optimizer=Adam(learning_rate=self.lr),
#             steps_per_execution=50,
#             metrics=['accuracy']
#         )

#         checkpoint = ModelCheckpoint("best_Bi_Lstm.hdf5",
#                                      monitor='val_loss',
#                                      save_best_only=True,
#                                      mode='auto',
#                                      save_freq='epoch')

#         print("Shape of train_y:", train_y.shape)
#         print("Shape of model predictions before training:", self.model.predict(train_x).shape)

#         history = model.fit(train_x,
#                             train_y,
#                             validation_data=(val_x, val_y),
#                             epochs=epochs,
#                             batch_size=batch_size,
#                             callbacks=[checkpoint])
#         return history

#     def predict(self, data):
#         return self.model.predict(data)

# Bi_Lstmmodel = Stgcn_gru_biLstm(num_joints=33, num_channels=3, num_timesteps=100)
# json_file_path = "C:/Users/ytkoffi/Desktop/sketon/STGCNGRU_BILSTM/Bi_Lstm.json"
# Bi_Lstmmodel.save_model_to_json(json_file_path)
