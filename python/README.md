##

The current approach is to code as much of the ViT transformer using the tf.keras functional API. This method allows majority of the model's architecture + weights to be converted to  tfjs conversion. While the model in its entirety can be coded in this way, many components of the models are not supported in tfjs at the time of writing. To deal with this, these components will need to be programmed in python using the functional API with other components that are supported in tfjs (not always possibile), or these will need to be coded natively in javascript, and subsequently transfer the weights. 

## Unsupported Components
1. Mutlti-headed Self Attention Layer
2. Gelu Activation Function