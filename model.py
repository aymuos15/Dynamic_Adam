import torch
import torch.nn as nn

class Foo(nn.Module):
    def __init__(self, input_size=20, output_size=100):
        super(Foo, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc0 = nn.Linear(self.input_size, 30)
        self.fc1 = nn.Linear(30, 40)
        self.fc_out = nn.Linear(40, self.output_size)

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return x

    def increment_output_size(self, copy_idx: int):
        old_output_size = self.output_size
        old_fc_out = self.fc_out

        self.output_size += 1
        self.fc_out = nn.Linear(40, self.output_size)
        
        with torch.no_grad():
            self.fc_out.weight.data[:old_output_size] = old_fc_out.weight.data
            self.fc_out.weight.data[-1] = old_fc_out.weight.data[copy_idx].clone()
            self.fc_out.bias.data[:old_output_size] = old_fc_out.bias.data
            self.fc_out.bias.data[-1] = old_fc_out.bias.data[copy_idx].clone()
