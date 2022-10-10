# ViT-tfjs
Vision Transformer architecture implemented in tensorflow js

# ViT Explained
A vision transformer (ViT) is an image classification architecture that has proved to achieve SOTA results.
Initialy created for use in NLP, a variety of steps and modifications to the self-attention NLP transformer were made for creating inferences on images. 
<br></br>
The steps to creating a ViT model are outlined below.

### Model Overview
"We split an image into fixed-size patches, linearly embed each of them,
add position embeddings, and feed the resulting sequence of vectors to a standard Transformer
encoder. In order to perform classification, we use the standard approach of adding an extra learnable
'classification token' to the sequence"

### Parameters
image_size - size of the input images
n_of_patches - number of patches that the image will be split into + 1 cls token
hidden_dimension - sets the dimension of the patch embeddings that enter into transformer
mlp_ratio - ratio between the size of the hidden layer and output layer in the MLP block
n_heads - number of heads in the Multi-headed Self Attention Block


# Formatting Images
A ViT handles sequences of data, not images. In order to make an image work with a transformer, it must be converted into a sequence. This is achieved by first splitting the image into patches, and then flattening each patch through a linear layer. These linearized image patches will then be modified by a created position vector, and then -- in addition to associated cls token -- will be passsed through the transformer as a sequence, for every sequence. 

### Step 1: Patching Images
Splits the image into n_of_patches. These are of size (image_size //  patch_size). 

### Step 2: Linearizing Patches
This can be achieved through the implementation of a trainable linear projection. This first flattens the patches from R^2 to R^1, then passes through a 1 layer MLP with input dimension of (HxWxC/P^2) and output of size hidden_dimension. The output of this layer is reffered to as the patch embeddings. 

### Step 3: Creating Positional Vector
To restore the spacial continuity for the images, every linearized patch is added to an associated position vector before input. Without this, the inputs permutation is irrelevant. Various implementations (learned and functional) have proven to be possible. This implementation will use a learnable positional vector, however, support for various ViT architectures will be avaible in the future. 

### Step 4: Creating Learned CLS Token
In addition to each linearized patch, an initial CLS token is passed through the model. 


# Transformer Encoder
After the images are patched, patches are linearized, and patch embeddings annealed with a position vector, the the sequence containing the cls token and patch embeddings is passed into the transformer encoder. Often, the ViT model architectures consist of a series of multiple transformer encoders. Inside each transformer encoder, there are two blocks. A Multi-headed Self Attention (MSA) block, and an MLP block. Before data enters each block, a layer normalization takes place, and after data exits each block, a residual connection takes place. 

Transformer Encoder Layers
1. Embedded Patches (1) 
2. Normalization Layer 
3. Multiheaded Self Attention (MSA) (2)
4. Residual Connection (3) = Sum of MSA output (2) and Embedded Patches (1) 
5. Normalization Layer 
6. MLP (4)
7. Residual Conection - Add MLP output (4) to Residual Connection (2)


## Multiheaded Self Attention Block
Self Attention is finding the similarity between inputs of a sequence. This is done by first generating Queries, Keys, and Values (q, k, v), determining their attention scores, and then passing it through a final linear layer. In order to view the image globally and locally, multiple heads (n_heads) are used. This adds an extra concatenation step before passing it through the last linear layer. 

### Step 1: Generating Queries, Keys, and Values
To generate (q, k, v), the embedded patch is sent through 3 distinct lienarization layers. For each head, these layers are unique. These layers will create q of dimension Dk, k of dimension Dk, and v of dimension Dv. 
<br>
Ex: For embedded patch P, q = (Wq)(P), k = (Wk)(P), v = (Wv)(P), where Wq, Wk, and Wv are distinct learnable, linear layers. 

### Step 2: Determining Attention
In order to determine the attention score between q, k, and v, Scaled Dot Product Attention is used. 
<br>
1. Dot Product between q and k
2. Scale Dot Product by (sqrt of dimension of k)
3. Apply softmax to output
4. Multiply by v

### Step 3: Concatenation
It is important to remember that there are h heads computing in paralell. In order to make meaningful inferences from this computation, the outputs from each head are concatenated into a single vector of size (Dk)

### Step 4: Linear Layer
The final concantenated vector is then passed through a trainable linear layer that outputs dimension Dv


## MLP Block
The MLP block consists of 2 layers. The size of the hidden layer is mlp_ratio times the output layer of dimension hidden_dimension.


## Normalization Layer
Normalizes the input vector to an output of mean 0 and standard deviation 1.

## Residual Connection
Connects the output of the MSA with its input and sends it to the MLP block. 
Connects the output of the MLP with its input and serves as the output for the transformer encoder. 

# Definitions

### CLS Token
A classification token was implemented in the original ViT paper (other implementations have removed it). It is a learnable parameter that stores information regarding other tokens (flattened patches). It is the first token in the sequence. Takes the shape of other flattened patches (1, hidden dimensions).

### Position Vector
A position vector is combined with each patch post embedding. The purpose of this is to restore spatial continuity before being passed through the encoder. 

# Resources
A lot of resources went into making this. I've tried to list all the ones that helped me here. 
1. https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
2. http://peterbloem.nl/blog/transformers
3. http://nlp.seas.harvard.edu/annotated-transformer/#prelims
4. 