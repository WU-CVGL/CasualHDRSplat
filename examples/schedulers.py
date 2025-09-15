import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Literal


def get_exponential_decay_scheduler(optimizer: Optimizer,
                                    lr_init: float,
                                    lr_final: Optional[float] = None,
                                    max_steps: int = 30000,
                                    lr_pre_warmup: float = 1e-8,
                                    warmup_steps: int = 0,
                                    ramp: Literal["linear",
                                                  "cosine"] = "cosine"):
    if lr_final is None:
        lr_final = lr_init

    def func(step):
        if step < warmup_steps:
            if ramp == "cosine":
                lr = lr_pre_warmup + (lr_init - lr_pre_warmup) * np.sin(
                    0.5 * np.pi * np.clip(step / warmup_steps, 0, 1))
            else:
                lr = (lr_pre_warmup +
                      (lr_init - lr_pre_warmup) * step / warmup_steps)
        else:
            t = np.clip((step - warmup_steps) / (max_steps - warmup_steps), 0,
                        1)
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return lr / lr_init  # divided by lr_init because the multiplier is with the initial learning rate

    scheduler = LambdaLR(optimizer, lr_lambda=func)
    return scheduler


if __name__ == "__main__":
    # Create a simple model
    model = torch.nn.Linear(10, 1)

    # Initialize the optimizer
    optimizer = Adam(model.parameters(), lr=0.1)

    # Create the learning rate scheduler
    scheduler = get_exponential_decay_scheduler(optimizer,
                                                lr_init=0.1,
                                                lr_final=0.01,
                                                max_steps=100,
                                                warmup_steps=10)

    # Simulate the training process and store learning rates
    learning_rates = []

    for step in range(100):
        # Perform a scheduler step
        scheduler.step()

        # Store the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # Sample training step (dummy forward and backward pass)
        optimizer.zero_grad()
        dummy_input = torch.randn(10)
        dummy_output = model(dummy_input)
        dummy_loss = dummy_output.sum()
        dummy_loss.backward()
        optimizer.step()

    # Plot the learning rate curve
    plt.plot(range(1, 101), learning_rates)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.savefig('scheduler_test.png')
