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
        super().__init__()
        assert config.n_embd % config.n_head == 0
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
            wte = nn.Embedding(config.vocab_size, config.n_embd), #weight token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), #weight position embedding
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # list of hidden layers, where each layer is a module (in this case, instances of Block), and it is managed by PyTorch's nn.ModuleList.
            ln_f = nn.LayerNorm(config.n_embd), #final Layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False) #final classifier layer model head that projects from embedding dimensions to vocab_size

    #Thi is the forward pass of the network
    def forward(self, idx):
        #idx - token indices
        # idx is of shape(B,T)
        B, T = idx.size
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        #forward the token and position embeddings
        #we use range and iterating from 0 to T and creating pos(position) indices and we are making sure they are the same device as idx because we are not going to train on only CPU we want to be training on GPU
        pos = torch.arange(T, dtype=torch.long, device=idx.device) #shape (T)
        pos_emb = self.transformer.wpe(pos) #position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) #token embeddings of shape (B, n_embd)
        x = tok_emb + pos_emb
        #forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # Apply the final layer normalization to the residual stream (x).
        # Layer normalization stabilizes the distribution of features by normalizing
        # the input across its feature dimension, ensuring consistent scale and mean.
        # This prepares the output for the next stage (e.g., the projection head) by
        # improving numerical stability and model performance.
        x = self.transformer.ln_f(x)
        # Calculate the logits for the next token in the sequence.
        # If the input has a shape (B, T) — where B is the batch size and T is the sequence length —
        # the logits tensor will have a shape (B, T, vocab_size), representing the predicted scores
        # for each possible token in the vocabulary at each position in the sequence.
        # Logits are raw, unnormalized scores and can be converted into probabilities
        # by applying a softmax function.
        #logits. If the input was B by T indices, then every single B by T we will calculate the logits for what token comes next in the sequence.
        #vocab_size is the number of possible tokens so this is a tenser that we are going to obtain. Logits are just a softmax away from becoming probabilist
        logits = self.lm_head(x) # B, T, vocab_size
        return logits


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt : %s" %model_type)
        #n_layers, n_heads and n_embd are determined from model_type
        config_args = {
            'gpt2' :        dict(n_layer=12, n_head=12, n_embd=768), #124M params
            'gpt2-medium' : dict(n_layer=24, n_head=16, n_embd=1024), #350M params
            'gpt2-large' :  dict(n_layer=36, n_head=20, n_embd=1280), #774M params
            'gpt2-xl' :     dict(n_layer=48, n_head=25, n_embd=1600) #1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 #always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 #always 1024 for GPT model checkpoints
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] #discard this mask

        #init a huggingface/transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        #copy while ensuring all the parameters are aligned and match in names and shapes
        #we are also ignoring a few buffers ex attn.bias and this comes from tenserflow repo and some weights are transposed so we hardcoded the weights that needs to be transposed
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval() #This is a good practice when you are not going to train it but use the model. But here maybe it's not doing anything right now as our model doesn't contain modules or layers that have different behavior in training or evaluation time
model.to('cuda') #Moving the model to cuda. Moving all the tensors to GPU. Sshed here to a cloud box and there are a bunch of GPU on the box. And here we are moving to a whole separated computer sitting on a GPU and GPU is connected to CPU. It is well catered to parallel processing tasks.

#prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # We get 8 tokens
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #We get 5 rows of 8 tokens (5,8)
x = tokens.to('cuda')