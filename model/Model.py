import torch
import torch.nn as nn
import torch.nn.functional as F



# batch_size = 64 # how many independent sequences will we process in parallel?
# block_size = 256 # what is the maximum context length for predictions?
max_iter = 5000
time_interval = 500
lr = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
# max_iters = 5000
# eval_interval = 100
# learning_rate = 1e-3
# if torch.has_mps:
#     device = torch.device('mps')
# elif torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
device = torch.device('cpu')
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2


with open("input.txt" , 'r', encoding='utf-8') as f:
    text = f.read()



# %%
chars = sorted(list(set(text)))
vocab_size = len(chars)

class LayerNorm1d: # (used to be BatchNorm1d)
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]



class Head(nn.Module):
   def  __init__(self, head_size):
       super().__init__()
       self.key = nn.Linear(n_embd, head_size, bias=False)
       self.query = nn.Linear(n_embd, head_size, bias=False)
       self.value = nn.Linear(n_embd, head_size, bias=False)
       self.register_buffer('tril', torch.tril(torch.ones (block_size, block_size)))
       
       self.dropout = nn.Dropout(dropout)
    
   def forward(self, x):
       B, T, C = x.shape
       k = self.key(x) # (B, T, H)
       q = self.query(x) # (B, T, H)

       wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)
       wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
       wei = F.softmax(wei, dim=-1) # (B, T, T)
       
       
       v = self.value(x) # (B, T, H)
       out = wei @ v # (B, T, H)
       
       return out
   


class MultiHeadAttention(nn.Module):
    """Multi heads in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out
       
       
class FeedForward(nn.Module):
    """ A Simple Linear Layer followed by non Linearity"""
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffw = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln1(x))
        return x

        

class BigramLanguageModel(nn.Module):
    def __init__(self) :
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding (vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_head = MultiHeadAttention(4, n_embd//4)
        # self.ffwd = FeedForward(n_embd)
        # self.block = nn.Sequential(
        #     Block(n_embd=n_embd, n_head=4),
        #     Block(n_embd=n_embd, n_head=4),
        #     Block(n_embd=n_embd, n_head=4),
        #     nn.LayerNorm(n_embd)
        # )
        self.block = nn.Sequential(*[Block(n_embd=n_embd, n_head=n_head) for _ in range(n_layer)])
        self.lm_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None) :
    # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_emb + pos_emb # (B, T, C)
        x = self.block(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, Vocab_size)
        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate (self, idx, max_new_tokens) :
        # idx is (B, T) array of indices in the current context
        for _ in range (max_new_tokens) :
        # get the predictions
            idx_cond = idx[:, -block_size:] # (B, T)  
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax (logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch. cat((idx, idx_next), dim=1) # (B, T+1)
        return idx