from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
from vit import create_vit

image_size = 224
ViT = create_vit(
                image_size = image_size,
                n_channels = 3,
                patch_size = 32,
                projection_dim = 8,
                transformer_layers = 2,
                num_heads = 8,
                num_classes = 2,
                mlp_head_units = [2048, 1024]
            )

# convert image in data folder to tensor suitable for the model
file_name = 'data/image.jpeg'
img = tf.reshape(tf.convert_to_tensor(np.asarray(Image.open(file_name).resize((image_size, image_size), Image.Resampling.BILINEAR))), [1, 224, 224, 3])

# run inference on the model
output = ViT(img)

ViT.summary()






