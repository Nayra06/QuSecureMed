# cnn_features.py

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_feature_extractor(input_shape=(256, 256, 1)):
    """
    Creates a simple CNN model to extract a feature map from the host image.
    The DWT-SVD watermarking would then be applied to this robust feature map.
    
    Args:
        input_shape (tuple): The expected shape of the input host image (H, W, C).
        
    Returns:
        tf.keras.Model: A model that outputs the feature map.
    """
    model = models.Sequential()
    
    # Input Layer (e.g., 256x256 grayscale image)
    model.add(layers.Input(shape=input_shape))
    
    # Layer 1: Learn 32 channels of features
    model.add(layers.Conv2D(
        32, (5, 5), activation='relu', padding='same', name='Feature_Conv_1'
    ))
    
    # Layer 2: Deeper features
    model.add(layers.Conv2D(
        16, (3, 3), activation='relu', padding='same', name='Feature_Conv_2'
    ))
    
    # Max Pooling: Reduces dimensionality and adds shift invariance
    model.add(layers.MaxPooling2D((2, 2), name='MaxPool'))
    
    # The output is the feature map, e.g., 128x128x16
    
    # We create a functional model to extract the output of the final convolutional layer
    # This feature map (128x128x16) is the "host" data for DWT-SVD embedding.
    feature_extractor = models.Model(
        inputs=model.input, 
        outputs=model.get_layer('Feature_Conv_2').output
    )
    
    return feature_extractor

# Example usage:
# extractor = create_feature_extractor()
# extractor.summary()

# To use it:
# dummy_image = np.random.rand(1, 256, 256, 1).astype(np.float32)
# feature_map = extractor.predict(dummy_image)
# print(f"Extracted Feature Map Shape: {feature_map.shape}")
