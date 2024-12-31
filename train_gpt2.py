import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

#MLP Block - we have two linear projections that are sandwiched between gelu non-linearity. Gelu is like a relu except there is no flat tail with exactly 0.
#We use the exact version over approximate version. BUT GTP-2 USES APPROXIMATE VERSION. We use Gelu over relu bcs if the tail is exactly 0 the changes that follow will get 0 gradient.
#There is no change or development of network if it's flat but the gelu always contributes local gradient so there will always be a change.

class CausalSelfAttention(nn.Module):
    #multi head attention. All heads function in parallel and their outputs are being  concatenated, and it becomes output of the multi head attention
    def __init__(self, config):
        super().__init__(config)
        assert config.n_embd % config.n_heads == 0
        #key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        #not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1, config.block_size, config.block_size))
    def forward(self, x):
        B, T, C = x.size()#batch size, sequence length, embedding dimensionality (n_embd)
        #calculate query, key, value for all heads in batch and move head forward to be the batch
        #nh is "number of heads", hs is "head size", and C (number of channels) = nh*hs
        #e.g. in GPT-2 (124M), n_heads = 12 hs = 64 so nh*hs=C=768 channels in the transformer
        #Tokens lined up in a sequence (there are 1020) and then each token in this stage of attention admits 3 vectors(query,key,value)
        #First the querys and the keys have to multiply each other to get the attention amount(how interesting they find each other) so they have to interact multiplicative
        #Calculating the qkv while splitting it
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_head, dim=2)
        #Gymanstics where we are making the number of heads into a batch dimension so that in the operation that follows pytorch treats B, nh as batches and apply all the operation to them in parallel
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh,T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh,T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh,T, hs)
        #Operation that apply : the query and the keys interact to give us our attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #Autoregressive masks that asure that the tokens only attend to the tokens before them and never to tokens in the future
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #Softmax normalizes the attention
        att = F.softmax(att, dim=-1)
        #att matrix multiply with the values is basically to do awaited sum of the values of the tokens that we found interesting
        #(The operation att @ v performs a weighted sum of the values, where the weights are determined by the attention scores in the att matrix. This operation aggregates the information from the tokens in the values matrix (v), giving more importance to the tokens deemed "interesting" or relevant (as determined by the attention scores in att).)
        y = att @ v #(B, nh, T, T) x (B, nh, T, hs) -> (B, nh,T, hs)
        #re-assemble all of it again(all the head inputs side by side)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        #output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

class Block(nn.Module):
    #initalization
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) #happends with every single

    #clean residual pathway

    #Layer Normalization directly estimates the normalization statistics from the summed inputs to the neurons within a hidden layer so the normalization does not introduce any new 
    # dependencies between training cases. It works well for RNNs and improves both the training time and the generalization performance of several existing RNN models. 
    # More recently, it has been used with Transformer models.
    
    #this is where all the tokens communicate
    def forward(self,x):
        #attn(attention is an aggregation function, pulling function, awaited some function, reduce operation )
        x = x + self.attn(self.ln_1(x)) #prenormalazation version where x goes through layer normalization then attention then back out to layer normalization 2(multia perception(feed forward network ffn))
        #after that it goes through residual stream again
        #mlp is the map where the tokens don't communicate at all with each other
        x  = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size : int = 1024 #max sequence length
    vocab_size : int = 50357 #number of tokens
    n_layer : int  = 12 #number of layers
    n_head : int = 12 #number of heads
    n_embd : int = 384 #embedding of dimension

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
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False) #final classifier layer model head that projects from embedding dimensions to vocab_size
