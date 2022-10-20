import tensorflow as tf
from tensorflow.keras import layers, Sequential, Functional
import math

'''
input images
patchify images --> py and js function needed
flatten patches --> functional api
linearize patches --> functional api
transformer encoder --> functional api
MLP output --> functional api
'''

# need custom function in python and js
def patchify_images(batched_tensor_images, n_patches):

    return patches 


# create and convert with sequential api
def embed_patches(patches): # takes in flattened patches of, and returns patch embeddings
    return patch_embeddings

# takes in patch of images, flattens, and sends to linear layer
patch_embedding_block = Sequential([
        layers.Input(shape=(patches_size)),
        layers.Flatten(),
        layers.Dense(units=hidden_dimension)
    ])

# create and convert with functional api
def transformer_encoder(patch_embeddings):
    normalized1 = normalizationLayer(patch_embeddings)
    q, k, v = qvkGenerator(normalized1)
    attention = MSA(q, k, v) 
    residual_connection1 = attention + normalized1
    normalized2 = normalizationLayer(residual_connection1)
    mlp_inference = MLP(normalized2)
    residual_connection2 = mlp_inference + residual_connection1
    return residual_connection2

