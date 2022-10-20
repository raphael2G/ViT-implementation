import tensorflow as tf


class test_model(tf.keras.Model):

    def __init__(self, input_shape, dim1, dim2, dim3, output_dim):
        super(test_model, self).__init__()
        self.block1 = test_block(input_shape, dim1, dim2)
        self.block2 = test_block(dim2, dim3, output_dim)

    def call(self, inputs):
        x = self.block1(inputs)
        return self.block2(x)

class test_block(tf.keras.layers.Layer):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(test_block, self).__init__()

        self.linear1 = tf.keras.layers.Dense(units=hidden_dim)
        self.linear2 = tf.keras.layers.Dense(units=output_dim)

    def call(self, inputs):
        x = self.linear1(inputs)
        return self.linear2(x)

input_shape = 2
x = tf.ones((10, input_shape))
net = test_model(input_shape, 5, 10, 15, 3)
print(net(x))
net.summary()

# net.save_weights('savedModels/test/dummyModel')