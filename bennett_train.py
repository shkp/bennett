# Import required libraries
import torch  # Main PyTorch library for deep learning
import torch.nn as nn  # Neural network modules from PyTorch
from torch.nn import functional as F  # Functional interface for neural network operations
import os  # Operating system interface for file operations
from ModelConfig import ModelConfig as cfg  # Custom configuration class for model parameters

# Set up directories and files for model checkpoints and logging
checkpoint_dir = 'checkpoints-austen-128'  # Directory to store model checkpoints
os.makedirs(checkpoint_dir, exist_ok=True)  # Create checkpoint directory if it doesn't exist
log_file = 'meta/austen-128-run.txt'  # File to log training progress

# Set up device and random seed
device = cfg.get_device()  # Get computing device (CPU/GPU) from config
print(f"Using device: {device}")  # Display which device will be used
torch.manual_seed(1337)  # Set random seed for reproducibility

# Load and preprocess the training data
with open('austen-5k.txt', 'r', encoding='utf-8') as f:  # Open input text file with UTF-8 encoding
    text = f.read()  # Read entire file content

# Create vocabulary and encoding/decoding mappings
chars = sorted(list(set(text)))  # Get sorted list of unique characters
vocab_size = len(chars)  # Count unique characters for vocabulary size
stoi = { ch:i for i,ch in enumerate(chars) }  # Map characters to indices
itos = { i:ch for i,ch in enumerate(chars) }  # Map indices to characters
encode = lambda s: [stoi[c] for c in s]  # Function to convert string to indices
decode = lambda l: ''.join([itos[i] for i in l])  # Function to convert indices to string

# Create train and validation splits
data = torch.tensor(encode(text), dtype=torch.long)  # Convert text to tensor of indices
n = int(0.9*len(data))  # Calculate 90% split point
train_data = data[:n]  # First 90% for training
val_data = data[n:]  # Last 10% for validation

def get_batch(split):
    """Get a random batch of data with corresponding targets"""
    data = train_data if split == 'train' else val_data  # Select appropriate dataset
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))  # Generate random starting indices
    x = torch.stack([data[i:i+cfg.block_size] for i in ix])  # Create input sequences
    y = torch.stack([data[i+1:i+cfg.block_size+1] for i in ix])  # Create target sequences
    x, y = x.to(device), y.to(device)  # Move data to appropriate device
    return x, y

@torch.no_grad()  # Disable gradient calculation for efficiency
def estimate_loss():
    """Estimate model's loss on train and validation sets"""
    out = {}  # Dictionary to store results
    model.eval()  # Set model to evaluation mode
    for split in ['train', 'val']:  # Iterate over both splits
        losses = torch.zeros(cfg.eval_iters)  # Initialize tensor for losses
        for k in range(cfg.eval_iters):  # Iterate multiple times for better estimate
            X, Y = get_batch(split)  # Get batch of data
            logits, loss = model(X, Y)  # Forward pass
            losses[k] = loss.item()  # Store loss value
        out[split] = losses.mean()  # Calculate mean loss
    model.train()  # Set model back to training mode
    return out

class Head(nn.Module):
    """Single attention head component"""
    def __init__(self, head_size):
        """Initialize attention head with specified size"""
        super().__init__()  # Initialize parent class
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)  # Key transformation
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)  # Query transformation
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)  # Value transformation
        self.register_buffer('tril', torch.tril(torch.ones(cfg.block_size, cfg.block_size)))  # Causal mask
        self.dropout = nn.Dropout(cfg.dropout)  # Dropout for regularization

    def forward(self, x):
        """Forward pass for attention head"""
        B,T,C = x.shape  # Get batch size, sequence length, embedding dim
        k = self.key(x)  # Generate keys   
        q = self.query(x)  # Generate queries
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # Compute attention scores with scaling
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Apply causal mask
        wei = F.softmax(wei, dim=-1)  # Apply softmax to get attention weights
        wei = self.dropout(wei)  # Apply dropout
        v = self.value(x)  # Generate values
        out = wei @ v  # Compute weighted sum of values
        return out

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, num_heads, head_size):
        """Initialize multiple attention heads"""
        super().__init__()  # Initialize parent class
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # Create list of attention heads
        self.proj = nn.Linear(head_size * num_heads, cfg.n_embd)  # Projection layer to combine heads
        self.dropout = nn.Dropout(cfg.dropout)  # Dropout for regularization

    def forward(self, x):
        """Forward pass for multi-head attention"""
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate outputs from all heads
        out = self.dropout(self.proj(out))  # Project and apply dropout
        return out

class FeedFoward(nn.Module):
    """Feed-forward neural network component"""
    def __init__(self, n_embd):
        """Initialize feed-forward network"""
        super().__init__()  # Initialize parent class
        self.net = nn.Sequential(  # Define network layers
            nn.Linear(n_embd, 4 * n_embd),  # First linear transformation
            nn.ReLU(),  # ReLU activation
            nn.Linear(4 * n_embd, n_embd),  # Second linear transformation
            nn.Dropout(cfg.dropout),  # Dropout for regularization
        )

    def forward(self, x):
        """Forward pass for feed-forward network"""
        return self.net(x)

class Block(nn.Module):
    """Transformer block combining attention and feed-forward layers"""
    def __init__(self, n_embd, n_head):
        """Initialize transformer block"""
        super().__init__()  # Initialize parent class
        head_size = n_embd // n_head  # Calculate size per head
        self.sa = MultiHeadAttention(n_head, head_size)  # Multi-head attention layer
        self.ffwd = FeedFoward(n_embd)  # Feed-forward layer
        self.ln1 = nn.LayerNorm(n_embd)  # First layer normalization
        self.ln2 = nn.LayerNorm(n_embd)  # Second layer normalization

    def forward(self, x):
        """Forward pass for transformer block"""
        x = x + self.sa(self.ln1(x))  # Attention layer with residual connection
        x = x + self.ffwd(self.ln2(x))  # Feed-forward layer with residual connection
        return x

class GPTLanguageModel(nn.Module):
    """Main GPT language model"""
    def __init__(self):
        """Initialize GPT model"""
        super().__init__()  # Initialize parent class
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)  # Token embeddings
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)  # Position embeddings
        self.blocks = nn.Sequential(*[Block(cfg.n_embd, n_head=cfg.n_head) for _ in range(cfg.n_layer)])  # Transformer blocks
        self.ln_f = nn.LayerNorm(cfg.n_embd)  # Final layer normalization
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)  # Output projection
        self.apply(self._init_weights)  # Initialize weights

    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):  # Initialize linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):  # Initialize embedding layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """Forward pass for GPT model"""
        B, T = idx.shape  # Get batch size and sequence length
        tok_emb = self.token_embedding_table(idx)  # Get token embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # Get position embeddings
        x = tok_emb + pos_emb  # Combine embeddings
        x = self.blocks(x)  # Pass through transformer blocks
        x = self.ln_f(x)  # Apply final layer normalization
        logits = self.lm_head(x)  # Generate logits

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape  # Get dimensions
            logits = logits.view(B*T, C)  # Reshape logits
            targets = targets.view(B*T)  # Reshape targets
            loss = F.cross_entropy(logits, targets)  # Calculate loss

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens"""
        for _ in range(max_new_tokens):  # Generate specified number of tokens
            idx_cond = idx[:, -cfg.block_size:]  # Get conditioning context
            logits, loss = self(idx_cond)  # Forward pass
            logits = logits[:, -1, :]  # Get last token logits
            probs = F.softmax(logits, dim=-1)  # Convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample next token
            idx = torch.cat((idx, idx_next), dim=1)  # Append to sequence
        return idx

def save_checkpoint(model, optimizer, iter, loss, filename):
    """Save model checkpoint"""
    checkpoint = {
        'iter': iter,  # Current iteration
        'model_state_dict': model.state_dict(),  # Model state
        'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
        'loss': loss,  # Current loss
        'vocab': {  # Vocabulary mappings
            'stoi': stoi,
            'itos': itos
        }
    }
    torch.save(checkpoint, filename)  # Save to file

def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint"""
    checkpoint = torch.load(filename)  # Load checkpoint file
    model.load_state_dict(checkpoint['model_state_dict'])  # Restore model state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Restore optimizer state
    return checkpoint['iter'], checkpoint['loss'], checkpoint['vocab']  # Return saved values

if __name__ == "__main__":
    # Initialize model and optimizer
    model = GPTLanguageModel()  # Create model instance
    m = model.to(device)  # Move model to device
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')  # Print model size
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)  # Create optimizer

    # Load checkpoint if exists
    start_iter = 0  # Initialize starting iteration
    if os.path.exists(f'{checkpoint_dir}/latest.pt'):  # Check for checkpoint
        print("Loading checkpoint...")
        start_iter, loss, loaded_vocab = load_checkpoint(model, optimizer, f'{checkpoint_dir}/latest.pt')
        print(f"Resuming from iteration {start_iter}")

    # Training loop
    checkpoint_interval = 100  # Save frequency
    for iter in range(start_iter, cfg.max_iters):  # Main training loop
        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:  # Evaluation block
            losses = estimate_loss()  # Calculate losses
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Generate sample text
            print('-' * 80)
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
            print('-' * 80)

        # Save checkpoint
        if iter % checkpoint_interval == 0 or iter == cfg.max_iters - 1:
            losses = estimate_loss()  # Calculate losses
            save_checkpoint(model, optimizer, iter, losses['val'], 
                       f'{checkpoint_dir}/checkpoint_{iter}.pt')  # Save numbered checkpoint
            save_checkpoint(model, optimizer, iter, losses['val'], 
                       f'{checkpoint_dir}/latest.pt')  # Save latest checkpoint
            print(f"Saved checkpoint at iteration {iter}")
            with open(log_file, 'a') as log:  # Log progress
                log.write(f"Step: {iter}, Train Loss: {losses['train']:.4f}, Validation Loss: {losses['val']:.4f}\n")

        # Training step
        xb, yb = get_batch('train')  # Get training batch
        logits, loss = model(xb, yb)  # Forward pass
        optimizer.zero_grad(set_to_none=True)  # Zero gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

    # Generate final sample
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Create initial context
    print(decode(m.generate(context, max_new_tokens=cfg.default_max_tokens)[0].tolist()))  # Generate and print sample