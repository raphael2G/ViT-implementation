// define unsupported layers as custom js layers
class gelu extends tf.layers.Layer {
    static className = 'gelu';
 
    constructor(config) {
      super(config);
    }

    call(features) {
      // return 0.5 * features * 
      // (1.0 + Math.tanh(0.7978845608028654 * 
      //   (features + 0.044715 * 
      //   Math.pow(features, 3))))
      return features
    }
 }
 tf.serialization.registerClass(gelu);


//  const data_augmentation = await tf.loadLayersModel('../savedModels/ViT/tfjs/data_augmentation/model.json');
const subModel = await tf.loadLayersModel('../savedModels/ViT/tfjs/subModel/model.json');
subModel.summary()

const dummy_input = tf.ones([10, 49, 8])
const output = subModel.predict(dummy_input)
output.print()







