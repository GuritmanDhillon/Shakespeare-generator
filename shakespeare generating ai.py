#we always start with a dataset to train on. We download the tiny shakespere dataset
import urllib.request
import torch
from math import e
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
urllib.request.urlretrieve(url, 'input.txt')

#read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

#create a maping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #encoder: take in a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: take a list of integers, output a string

#lets now encode the entire dataset and store it into a torch.tensor
data = torch.tensor(encode(text), dtype=torch.long)

#now lets split up our data into train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#List of Hyper Parameters
batch_size = 32 # how many independent sequences will we process in parallel
block_size = 8 # what is the maximum content length for prediction
max_iters = 5000 # how many iterations to train for
eval_interval = 500 # how often to check the validation loss
learning_rate = 3e-4 # what learning rate to use
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # how many iterations to evaluate the model
n_embd = 128 # the dimensionality of the character embedding vectors
n_head = 4 # the number of heads in the multiheadattention models
n_layer = 4 # the number of sublayers in the transformer
dropout = 0.2 # the dropout probability
train_data[:block_size+1]
#-------------------------------------------|


def get_batch(split):
  #generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data)-block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  #x and y are both (batch_size, block_size) tensors
  return x, y


@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(split)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out


xb, yb = get_batch('train')

class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei.masked_fill_(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # preform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B,T,C) -> (B,T,C*n_heads)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(n_embd, 4 * n_embd),
         nn.ReLU(),
         nn.Linear(4 * n_embd, n_embd),
         nn.Dropout(dropout)
      )

    def forward(self, x):
      return self.net(x)

class Block(nn.Module):
    """a transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        #n_embd is the embedding dimension, n_heads is the number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
       x = x + self.sa(self.ln1(x))
       x = x + self.ffwd(self.ln2(x))
       return x

class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):

    # idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx) # (B,T,C)
    pos_emb = self.token_embedding_table(idx) # (B,T,C)
    x = tok_emb + pos_emb # (B,T,C)
    x = self.blocks(x) # (B,T,C)
    x = self.ln_f(x) # (B,T,C)
    logits = self.lm_head(x) # (B,T,vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        #crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        #get the predictions
        logits, loss = self(idx_cond)
        #focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        #apply softmax to get probabilities
        probs = F.softmax(logits, dim = -1) # (B, C)
        #sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        #append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

model = BigramLanguageModel()
m = model.to(device)


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    #sample of batch data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))
# remove the device=device argument in the print() above, if the code doesnt work