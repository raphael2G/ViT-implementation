### ViT Explained
A vision transformer (ViT) is an image classification architecture that has proved to achieve SOTA results.
Initialy created for use in NLP, a variety of steps and modifications to the self-attention NLP transformer were made for creating inferences on images. 
<br></br>
The steps to creating a ViT model are outlined below.

# Model Overview
"We split an image into fixed-size patches, linearly embed each of them,
add position embeddings, and feed the resulting sequence of vectors to a standard Transformer
encoder. In order to perform classification, we use the standard approach of adding an extra learnable
'classification token' to the sequence"

# Parameters
Image Size - 
#_of_patches - assert(Image Size % number of patches == 0)
patch_size - (image height * image width / number of patches) + 1 cls token
hidden_dimensions - size of linearized patches that go into transformer. theoritically arbitrary


### Formatting Images
A ViT handles sequences of data, not images. In order to make an image work with a transformer, it must be converted into a sequence. This is achieved by first splitting the image into patches, and then flattening each patch through a linear layer. These linearized image patches will then be modified by a created position vector, and then -- in addition to associated cls token -- will be passsed through the transformer as a sequence, for every sequence. 

# Step 1: Patching Images

# Step 2: Linearizing Patches
This can be achieved through the implementation of a trainable linear projection. This first flattens the patches from R^2 to R^1, then passes through a 1 layer MLP with input dimension of (HxWxC/P^2) and output of size hidden_dimensions. The output of this layer is reffered to as the patch embeddings. 

# Step 3: Creating Positional Vector
Patching the image destroys its spatial continuity. To account for this, every linearized patch is passed with an associated position vector as input. Without this, the inputs permutation is irrelevant. Various implementations (learned and functional) have proven to be possible. A single positional vector is annealed to each sequence. See model diagram for visual explanation. 

# Step 4: Creating Learned CLS Token
In addition to each linearized patch, an initial CLS token is passed through the model. 


### Transformer Encoder
Each encoder must first create the query, keys, and values, then send those inputs through two blocks. 

## Generating Queries, Keys, and Values
To generate (q, k, v), the embedded patch is sent through 3 distinct lienarization layers. 
Ex: for embedded patch P, q = (Wq)(P), k = (Wk)(P), v = (Wv)(P)
Wq, Wk, and Wv are distinct learnable, linear layers. 

1. Mutliheaded Self Attention (MSA)
2. MLP

Before each input, a layer normalization takes place, and after each output, a residual connection takes place. 

Each transformer encoder layers follow like this: 
1. Embedded Patches (1) 
            ↓↓↓↓
2. Normalization Layer 
            ↓↓↓↓
3. Multiheaded Self Attention (MSA)
            ↓↓↓↓
4. Residual Connection (2) - Sum MSA output and Embedded Patches (1) 
            ↓↓↓↓
5. Normalization Layer 
            ↓↓↓↓
6. MLP 
            ↓↓↓↓
7. Residual Conection - Add MLP output to Residual Connection (2)


### Multiheaded Self Attention
Self Attention is finding the similarity between inputs of a sequence. Multiheaded Self Attention does this with with h instances, concatenates the results, and puts this through a final linear projection.  
# Step 1: 

## MLP 

# Step 1: 

## Normalization Layer
Normalizes the passed vector to have a mean of 0 and standard deviation of 1

## Residual Connection
Connects the output of the MSA with its input. 
Connects the output of the MLP with its input.

### Definitions

# CLS Token
A classification token was implemented in the original ViT paper (other implementations have removed it). It is a learnable parameter that stores information regarding other tokens (flattened patches). It is the first token in the sequence. Takes the shape of other flattened patches (1, hidden dimensions).

# Position Vector
