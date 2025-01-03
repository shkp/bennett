import torch


class ModelConfig:
    # Model architecture - Altering this will mean you need to retrain the whole model
    n_embd = 128
    n_head = 6
    n_layer = 6
    block_size = 256
    dropout = 0.2

    # Training parameters - can be altered mid-run
    batch_size = 64
    max_iters = 5000
    eval_interval = 50
    learning_rate = 3e-4
    eval_iters = 100

    # Generation parameters
    default_max_tokens = 1000
    default_temperature = 1.0

    # Device configuration
    @staticmethod
    def get_device():
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'