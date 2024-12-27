import torch
import torch.nn as nn
from torch.nn import functional as F
from ModelConfig import ModelConfig as cfg

device = cfg.get_device()
print(f"Using device: {device}")


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg.n_embd, n_head=cfg.n_head) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def load_model(checkpoint_path):
    """Load the saved model and vocabulary"""
    checkpoint = torch.load(checkpoint_path)
    vocab = checkpoint['vocab']
    stoi, itos = vocab['stoi'], vocab['itos']
    
    # Create and load model
    model = GPTLanguageModel(len(stoi))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, stoi, itos

def generate_text(model, stoi, itos, prompt="", max_tokens=None, temperature=None):
    """Generate text from a prompt"""
    model.eval()
    
    # Use default values from config if not specified
    max_tokens = max_tokens or cfg.default_max_tokens
    temperature = temperature or cfg.default_temperature
    
    # Encode prompt if provided, otherwise start with empty context
    if prompt:
        context = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate text
    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
        generated_text = ''.join([itos[i] for i in generated_tokens])
    
    return generated_text

if __name__ == "__main__":
    # Ask for checkpoint path
    while True:
        checkpoint_path = input("Enter checkpoint path (e.g., checkpoints/latest.pt): ")
        try:
            model, stoi, itos = load_model(checkpoint_path)
            break
        except FileNotFoundError:
            print(f"Checkpoint file '{checkpoint_path}' not found. Please try again.")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            print("Please try again.")

    while True:
        # Get generation parameters from user
        prompt = input("\nEnter your prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
            
        try:
            max_tokens = int(input(f"Enter number of tokens to generate (default {cfg.default_max_tokens}): ") or cfg.default_max_tokens)
            temperature = float(input(f"Enter temperature (0.1-2.0, default {cfg.default_temperature}): ") or cfg.default_temperature)
        except ValueError:
            print("Invalid input. Using default values.")
            max_tokens = cfg.default_max_tokens
            temperature = cfg.default_temperature
            
        print("\nGenerating text...\n")
        print("-" * 80)
        generated_text = generate_text(model, stoi, itos, prompt=prompt, 
                                    max_tokens=max_tokens, temperature=temperature)
        print(generated_text)
        print("-" * 80)
        
        # Ask if user wants to save the output
        save = input("\nDo you want to save this output? (yes/no): ").lower()
        if save.startswith('y'):
            filename = input("Enter filename to save to: ")
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(generated_text)
                print(f"Output saved to {filename}")
            except Exception as e:
                print(f"Error saving file: {str(e)}")

        # Ask if user wants to continue
        continue_generating = input("\nGenerate another text? (yes/no): ").lower()
        if not continue_generating.startswith('y'):
            break

    print("\nThank you for using the text generator!")