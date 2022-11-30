from fastapi import FastAPI
import os
import tensorflow as tf
import numpy as np

app = FastAPI()

base_data_path = '../../data/images'
base_segmentation_path = '../../data/segmentation'
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=[224, 224, 3])
])


@app.get("/files/{path}")
async def classification(path: str):
    new_path = os.path.join(base_segmentation_path, path)
    f = open(new_path, 'x')
    return {new_path}
