// define unsupported layers as custom js layers

// Gelu Activation Function
class gelu extends tf.layers.Layer {
    static className = 'gelu';
 
    constructor(config) {
      super(config);
    }

    call(features) {
      // need to implement gelu from the paper 
      return features
    }
 }
 tf.serialization.registerClass(gelu);


// Patch Encoder (adding position vector to patch embeddings)
class PatchesEncoder extends tf.layers.Layer {
  static className = 'PatchesEncoder';

  constructor(config) {
    super(config);
    
  }

  call(patches) {
    const projection = tf.layers.dense({units: projection_dim});
    const position_embedding = tf.layers.embedding({inputDim: n_patches, outputDim: projection_dim});

    let positions = tf.range(0, n_patches, 1);


    let embedded_position = position_embedding.apply(positions);
    const encoded_patches = projection.apply(patches);

    embedded_positions = tf.reshape(embedded_position, [1, 49, 1]);


    const repeatLayer = tf.layers.repeatVector({n: 10, inputShape: [49, 1]});
    const formatedPositions = repeatLayer.apply(repeatLayer);

    console.log(formatedPositions.shape);
    console.log(encoded_patches.shape);

    const addLayer = tf.layers.add();
    return addLayer.apply([embedded_position, encoded_patches]);
  }

  getClassName() { return 'PatchesEncoder'; }
}

// 
const patchEncoder = new PatchesEncoder();
const subModel = await tf.loadLayersModel('../savedModels/ViT/tfjs/subModel/model.json');


function getPrediction(flattenedPatches){
  const encodedPatches = patchEncoder.apply(flattenedPatches);
  const output = subModel.apply(encodedPatches);

  encodedPatches.print();
  output.print();
  
  return output
}

const dummy_data = tf.ones([10, 49, 8]);

let output = getPrediction(dummy_data);
console.log('output done well')












