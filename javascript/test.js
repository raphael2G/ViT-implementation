function test_model(input_shape) {

    const input = tf.input({shape: input_shape});

    const block1 = test_block(input_shape, 10, 20);
    const block2 = test_block(20, 10, 5);

    const output = block2.apply(block1.apply(input));
    const model = tf.model({inputs: input, outputs: output});
    return model;
}

function test_block(input_dim, hidden_dim, output_dim) {
    const model = tf.sequential();

    model.add(tf.layers.dense({units: hidden_dim, inputShape: input_dim}));
    model.add(tf.layers.dense({units: output_dim}));

    return model;
}   

const input_shape = 2;
const dummy_data = tf.ones([10, input_shape])

const model = test_model(input_shape, 5, 10, 15, 3)
console.log(model.predict(dummy_data).arraySync())
model.summary()


