import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Input, Conv2D, BatchNormalization, Activation, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard


def two_d_cnn_detector(input_shape, num_filters, kernel_sizes, fc_units):
    """
    Defines the 2D-CNN-based signal detector for OTFS systems in a structured manner with individual outputs.

    Args:
        input_shape: Tuple, shape of the input tensor (N, M, C), where C=2 for real and imaginary channels.
        num_filters: List of integers, number of filters in each convolutional layer.
        kernel_sizes: List of tuples, kernel size for each convolutional layer.
        fc_units: Integer, number of units in the fully connected output layer.

    Returns:
        model: Compiled Keras model.
    """
    outputs = []
    loss = []
    matriks = []
    # Input layer for the 3D tensor
    inputs = keras.Input(shape=input_shape)
    # Initialize intermediate layers storage
    layers = []

    # Convolutional layers with BatchNormalization and ReLU activation
    for idx, (filters, kernel_size) in enumerate(zip(num_filters[:-1], kernel_sizes[:-1])):
        if idx == 0:
            x = Conv2D(filters=filters, kernel_size=kernel_size, padding="same")(inputs)
        else:
            x = Conv2D(filters=filters, kernel_size=kernel_size, padding="same")(layers[-1])
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        layers.append(x)

    # Final Conv2D layer for output with sigmoid activation
    x = Conv2D(filters=num_filters[-1], kernel_size=kernel_sizes[-1], padding="same", activation="relu")(layers[-1])
    x = Dense(1024, kernel_regularizer=keras.regularizers.l2(l=0.1), activation='relu')(x)
    x = Dense(2, kernel_regularizer=keras.regularizers.l2(l=0.2), activation='sigmoid')(x)
    outputs.append(x)

    # Assign losses and metrics
    for idx, output in enumerate(outputs):
        loss.append('mean_squared_error'); matriks.append('accuracy')
        #loss['Output_' + str(idx + 1)] = 'conv2d_15'
        #matriks['Output_' + str(idx + 1)] = ['accuracy']

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs, name='2D_CNN_Detector_Pattern')
    model.compile(SGD(lr=0.01, nesterov=True), loss=loss, metrics=matriks)

    return model



# Define the CNN model architecture
def MIMO_OTFS_Detector(input_shape=(134, 2)):
    model = models.Sequential([
        layers.Conv2D(16, (7, 7), activation='relu', padding='same', input_shape=(8, 8, 4)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(16, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(4, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(4, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(2, (3, 3), activation='sigmoid',padding='same'),
    ])
    return model

# Convert complex matrix to real-valued tensors for processing
def to_real_tensor(matrix):
    real_part = np.real(matrix)
    imag_part = np.imag(matrix)
    return np.stack((real_part, imag_part), axis=-1)

