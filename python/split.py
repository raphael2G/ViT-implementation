import tensorflowjs as tfjs
from tensorflow import keras

def split_keras_model(model, index):

    layer_input_1 = keras.layers.Input(model.layers[0].input_shape[1:])
    layer_input_2 = keras.layers.Input(model.layers[index].input_shape[1:])
    x = layer_input_1
    y = layer_input_2

    for layer in model.layers[0:index]:
        x = layer(x)

    for layer in model.layers[index:]:
        y = layer(y)

    model1 = keras.Model(inputs=layer_input_1, outputs=x)
    model2 = keras.Model(inputs=layer_input_2, outputs=y)
    
    print('-- -- -- -- -- FULL MODEL -- -- -- -- --')
    model.summary()

    print('-- -- -- - FEATURE EXTRACTION - -- -- --')
    model1.summary()

    print('-- -- -- -- - DENSE LAYERS - -- -- -- --')
    model2.summary()

    return (model1, model2)