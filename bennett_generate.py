# Import required libraries
import torch  # Main PyTorch library for deep learning
import torch.nn as nn  # Neural network modules from PyTorch
from torch.nn import functional as F  # Functional interface for neural network operationsben
from ModelConfig import ModelConfig as cfg  # Custom configuration class for model parameters

# Set up device configuration
device = cfg.get_device()  # Get computing device (CPU/GPU) from config
print(f"Using device: {device}")  # Display which device will be used

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        """Initialize an attention head"""
        super().__init__()  # Initialize parent class
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)  # Key transformation layer
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)  # Query transformation layer
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)  # Value transformation layer
        self.register_buffer('tril', torch.tril(torch.ones(cfg.block_size, cfg.block_size)))  # Create causal mask
        self.dropout = nn.Dropout(cfg.dropout)  # Dropout layer for regularization

    def forward(self, x):
        """Forward pass of attention head"""
        B,T,C = x.shape  # Get batch size, sequence length, and embedding dimensions
        k = self.key(x)  # Generate key vectors   
        q = self.query(x)  # Generate query vectors
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # Compute attention scores with scaling
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Apply causal mask
        wei = F.softmax(wei, dim=-1)  # Apply softmax to get attention weights
        wei = self.dropout(wei)  # Apply dropout
        v = self.value(x)  # Generate value vectors
        out = wei @ v  # Compute weighted sum of values
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        """Initialize multi-head attention"""
        super().__init__()  # Initialize parent class
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # Create multiple attention heads
        self.proj = nn.Linear(head_size * num_heads, cfg.n_embd)  # Projection layer to combine head outputs
        self.dropout = nn.Dropout(cfg.dropout)  # Dropout layer for regularization

    def forward(self, x):
        """Forward pass of multi-head attention"""
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate outputs from all heads
        out = self.dropout(self.proj(out))  # Project and apply dropout
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        """Initialize feed-forward network"""
        super().__init__()  # Initialize parent class
        self.net = nn.Sequential(  # Define sequential network
            nn.Linear(n_embd, 4 * n_embd),  # First linear transformation
            nn.ReLU(),  # ReLU activation
            nn.Linear(4 * n_embd, n_embd),  # Second linear transformation
            nn.Dropout(cfg.dropout),  # Dropout for regularization
        )

    def forward(self, x):
        """Forward pass of feed-forward network"""
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        """Initialize transformer block"""
        super().__init__()  # Initialize parent class
        head_size = n_embd // n_head  # Calculate size per attention head
        self.sa = MultiHeadAttention(n_head, head_size)  # Multi-head attention layer
        self.ffwd = FeedFoward(n_embd)  # Feed-forward layer
        self.ln1 = nn.LayerNorm(n_embd)  # First layer normalization
        self.ln2 = nn.LayerNorm(n_embd)  # Second layer normalization

    def forward(self, x):
        """Forward pass of transformer block"""
        x = x + self.sa(self.ln1(x))  # Attention with residual connection
        x = x + self.ffwd(self.ln2(x))  # Feed-forward with residual connection
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        """Initialize GPT language model"""
        super().__init__()  # Initialize parent class
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)  # Token embeddings
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)  # Position embeddings
        self.blocks = nn.Sequential(*[Block(cfg.n_embd, n_head=cfg.n_head) for _ in range(cfg.n_layer)])  # Transformer blocks
        self.ln_f = nn.LayerNorm(cfg.n_embd)  # Final layer normalization
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)  # Output projection

    def forward(self, idx, targets=None):
        """Forward pass of GPT model"""
        B, T = idx.shape  # Get batch size and sequence length
        tok_emb = self.token_embedding_table(idx)  # Get token embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # Get position embeddings
        x = tok_emb + pos_emb  # Combine embeddings
        x = self.blocks(x)  # Pass through transformer blocks
        x = self.ln_f(x)  # Apply final layer normalization
        logits = self.lm_head(x)  # Generate logits

        if targets is None:  # If no targets provided (generation mode)
            loss = None
        else:  # If targets provided (training mode)
            B, T, C = logits.shape  # Get dimensions
            logits = logits.view(B*T, C)  # Reshape logits
            targets = targets.view(B*T)  # Reshape targets
            loss = F.cross_entropy(logits, targets)  # Calculate loss

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens from context"""
        for _ in range(max_new_tokens):  # Generate specified number of tokens
            idx_cond = idx[:, -cfg.block_size:]  # Get conditioning context
            logits, _ = self(idx_cond)  # Get predictions
            logits = logits[:, -1, :]  # Get logits for next token
            probs = F.softmax(logits, dim=-1)  # Convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample next token
            idx = torch.cat((idx, idx_next), dim=1)  # Append to sequence
        return idx

def load_model(checkpoint_path):
    """Load the saved model and vocabulary"""
    checkpoint = torch.load(checkpoint_path)  # Load checkpoint file
    vocab = checkpoint['vocab']  # Get vocabulary from checkpoint
    stoi, itos = vocab['stoi'], vocab['itos']  # Get character to/from index mappings
    
    # Create and load model
    model = GPTLanguageModel(len(stoi))  # Initialize model with vocabulary size
    model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
    model = model.to(device)  # Move model to device
    model.eval()  # Set to evaluation mode
    
    return model, stoi, itos

def generate_text(model, stoi, itos, prompt="", max_tokens=None, temperature=None):
    """Generate text from a prompt"""
    model.eval()  # Set model to evaluation mode
    
    # Use default values if not specified
    max_tokens = max_tokens or cfg.default_max_tokens  # Number of tokens to generate
    temperature = temperature or cfg.default_temperature  # Temperature for sampling
    
    # Process the prompt
    if prompt:  # If prompt provided
        context = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)  # Encode prompt
    else:  # If no prompt
        context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start with empty context
    
    # Generate text
    with torch.no_grad():  # Disable gradient calculation
        generated_tokens = model.generate(context, max_new_tokens=max_tokens)[0].tolist()  # Generate tokens
        generated_text = ''.join([itos[i] for i in generated_tokens])  # Convert tokens to text
    
    return generated_text

if __name__ == "__main__":
    # Interactive text generation loop
    
    # Load model checkpoint
    while True:
        checkpoint_path = input("Enter checkpoint path (e.g., checkpoints/latest.pt): ")  # Get checkpoint path
        try:
            model, stoi, itos = load_model(checkpoint_path)  # Try to load model
            break
        except FileNotFoundError:
            print(f"Checkpoint file '{checkpoint_path}' not found. Please try again.")  # Handle missing file
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")  # Handle other errors
            print("Please try again.")

    # Main generation loop
    while True:
        # Get generation parameters
        prompt = input("\nEnter your prompt (or 'quit' to exit): ")  # Get user prompt
        if prompt.lower() == 'quit':  # Check for quit command
            break
            
        try:
            # Get generation parameters with defaults
            max_tokens = int(input(f"Enter number of tokens to generate (default {cfg.default_max_tokens}): ") or cfg.default_max_tokens)
            temperature = float(input(f"Enter temperature (0.1-2.0, default {cfg.default_temperature}): ") or cfg.default_temperature)
        except ValueError:
            print("Invalid input. Using default values.")  # Handle invalid input
            max_tokens = cfg.default_max_tokens
            temperature = cfg.default_temperature
            
        # Generate and display text
        print("\nGenerating text...\n")
        print("-" * 80)
        generated_text = generate_text(model, stoi, itos, prompt=prompt, 
                                    max_tokens=max_tokens, temperature=temperature)
        print(generated_text)
        print("-" * 80)
        
        # Handle saving output
        save = input("\nDo you want to save this output? (yes/no): ").lower()  # Ask about saving
        if save.startswith('y'):
            filename = input("Enter filename to save to: ")  # Get filename
            try:
                with open(filename, 'w', encoding='utf-8') as f:  # Save to file
                    f.write(generated_text)
                print(f"Output saved to {filename}")
            except Exception as e:
                print(f"Error saving file: {str(e)}")  # Handle save errors

        # Check if user wants to continue
        continue_generating = input("\nGenerate another text? (yes/no): ").lower()
        if not continue_generating.startswith('y'):
            break

    print("\nThank you for using the text generator!")  # Exit message