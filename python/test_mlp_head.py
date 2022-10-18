from aifc import Error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflowjs as tfjs


# Parameters in Here
image_size = 224
n_channels = 3
patch_size = 32
n_patches = (image_size // patch_size) ** 2
assert (image_size % patch_size == 0), 'image_size must be divisible by patch_size'
projection_dim = 8
transformer_layers = 2
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim,
] 
num_classes = 2
mlp_head_units = [2048, 1024]

def mlp_head():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(n_patches, projection_dim), name='MLP_Head_Input'),
        layers.LayerNormalization(epsilon=1e-6),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(mlp_head_units[0], activation=tf.nn.gelu),
        layers.Dropout(0.5),
        layers.Dense(num_classes)
        ], name='MLP_head')
    
    return model

mlp_head = mlp_head()
mlp_head.summary()
print(mlp_head(tf.ones((10, 49, 8))))

tfjs.converters.save_keras_model(mlp_head, 'savedModels/MLP_head')
