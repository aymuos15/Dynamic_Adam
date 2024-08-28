import torch
import torch.nn as nn
import torch.optim as optim

from model import Foo
from update import update_optimizer_state

if __name__ == "__main__":
    # Setup model and optimizer
    model = Foo()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Simulate some training
    x = torch.randn(10, 20)
    y = torch.randint(0, 100, (10,))
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    # Modify model to increase output size
    print("Current output size:", model.output_size)
    old_fc_out = model.fc_out
    model.increment_output_size(copy_idx=25)
    print("Updated output size:", model.output_size)

    # Update optimizer state
    update_optimizer_state(optimizer, old_fc_out, model.fc_out, copy_idx=25, old_output_size=100)

    # Continue training with the updated model
    optimizer.zero_grad()
    y = torch.randint(0, 101, (10,))  # New target with increased number of classes
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    print("Done")