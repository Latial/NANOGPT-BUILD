from dataclasses import dataclass
import torch;
import torch.nn as nn
from torch.nn import functional as F

class Block(nn.Module):
    #initalization
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttetion(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) #happends with every single

    #clean residual pathway

    #Layer Normalization directly estimates the normalization statistics from the summed inputs to the neurons within a hidden layer so the normalization does not introduce any new 
    # dependencies between training cases. It works well for RNNs and improves both the training time and the generalization performance of several existing RNN models. 
    # More recently, it has been used with Transformer models.
    
    #this is where all the tokens comunicate
    def forward(self,x):
        #attn(attetion is a agrogation fucntion, pulling function, awaited some funnction, reduce operation )
        x = x + self.attn(self.ln_1(x)) #prenormalazation version where x goes through layer normalazation then attetion then back out to layer normalazation 2(multia perception(feed forward network ffn))
        #after that it goest trough residual stream again
        #mlp is the map where the tokens dont communicate at all with each other
        x  = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size : int = 256
    vocab_size : int = 65
    n_layer : int  = 6
    n_head : int = 6
    n_embd : int = 384

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            #nnEmbedding - fancy wrapper module around a single array of numbers(block of numbers)
            #Embedding - refers to representing high-dimensional data (like text, images, or categorical data) in a lower-dimensional vector space, typically as numerical vectors
            wte = nn.Embedding(config.vocab_size, config.n_emdb), #weight token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), #weight position embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # list of hidden layers, where each layer is a module (in this case, instances of Block), and it is managed by PyTorch's nn.ModuleList.
            ln_f = nn.LayerNorm(config.n_embd), #final Layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False) #final classifier layer model head that projects from embeding dimensions to vocab_size
