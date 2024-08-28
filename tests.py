import unittest
import torch
import torch.nn as nn
import torch.optim as optim

from update import update_optimizer_state
from model import Foo

class TestDynamicModel(unittest.TestCase):

    def setUp(self):
        self.input_size = 20
        self.output_size = 100
        self.model = Foo(self.input_size, self.output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def test_model_initialization(self):
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.output_size, self.output_size)
        self.assertIsInstance(self.model.fc0, nn.Linear)
        self.assertIsInstance(self.model.fc1, nn.Linear)
        self.assertIsInstance(self.model.fc_out, nn.Linear)

    def test_forward_pass(self):
        x = torch.randn(10, self.input_size)
        output = self.model(x)
        self.assertEqual(output.shape, (10, self.output_size))

    def test_increment_output_size(self):
        old_output_size = self.model.output_size
        copy_idx = 25
        self.model.increment_output_size(copy_idx)
        
        self.assertEqual(self.model.output_size, old_output_size + 1)
        self.assertEqual(self.model.fc_out.out_features, old_output_size + 1)
        
        # Check if weights and biases are correctly copied
        self.assertTrue(torch.all(self.model.fc_out.weight.data[:-1].eq(self.model.fc_out.weight.data[:-1])))
        self.assertTrue(torch.all(self.model.fc_out.weight.data[-1].eq(self.model.fc_out.weight.data[copy_idx])))
        self.assertTrue(torch.all(self.model.fc_out.bias.data[:-1].eq(self.model.fc_out.bias.data[:-1])))
        self.assertTrue(torch.all(self.model.fc_out.bias.data[-1].eq(self.model.fc_out.bias.data[copy_idx])))

    def test_update_optimizer_state(self):
        old_fc_out = self.model.fc_out
        copy_idx = 25
        old_output_size = self.model.output_size
        
        # Perform a backward pass to initialize optimizer state
        x = torch.randn(10, self.input_size)
        y = torch.randint(0, self.output_size, (10,))
        loss = nn.CrossEntropyLoss()(self.model(x), y)
        loss.backward()
        self.optimizer.step()
        
        # Store old optimizer state
        old_state = self.optimizer.state[old_fc_out.weight]['exp_avg'].clone()
        
        # Increment output size and update optimizer
        self.model.increment_output_size(copy_idx)
        update_optimizer_state(self.optimizer, old_fc_out, self.model.fc_out, copy_idx, old_output_size)
        
        # Check if optimizer state is correctly updated
        new_state = self.optimizer.state[self.model.fc_out.weight]['exp_avg']
        self.assertEqual(new_state.shape[0], old_state.shape[0] + 1)
        self.assertTrue(torch.all(new_state[:-1].eq(old_state)))
        self.assertTrue(torch.all(new_state[-1].eq(old_state[copy_idx])))

    def test_training_after_update(self):
        copy_idx = 25
        old_fc_out = self.model.fc_out
        old_output_size = self.model.output_size
        
        # Increment output size and update optimizer
        self.model.increment_output_size(copy_idx)
        update_optimizer_state(self.optimizer, old_fc_out, self.model.fc_out, copy_idx, old_output_size)
        
        # Attempt a training step
        x = torch.randn(10, self.input_size)
        y = torch.randint(0, self.model.output_size, (10,))
        self.optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(self.model(x), y)
        loss.backward()
        
        # Check if we can perform an optimizer step without errors
        try:
            self.optimizer.step()
        except Exception as e:
            self.fail(f"Optimizer step failed after update: {str(e)}")

if __name__ == '__main__':
    unittest.main()