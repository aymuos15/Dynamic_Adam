import torch

def update_optimizer_state(optimizer, old_fc_out, new_fc_out, copy_idx, old_output_size):
    # Identify the indices corresponding to the old `fc_out` parameters
    fc_out_weight_idx = next(i for i, p in enumerate(optimizer.param_groups[0]["params"]) if p is old_fc_out.weight)
    fc_out_bias_idx = next(i for i, p in enumerate(optimizer.param_groups[0]["params"]) if p is old_fc_out.bias)

    # Replace the old parameters with the new ones
    optimizer.param_groups[0]["params"][fc_out_weight_idx] = new_fc_out.weight
    optimizer.param_groups[0]["params"][fc_out_bias_idx] = new_fc_out.bias

    # Initialize new optimizer states
    new_state_weight = {
        'exp_avg': torch.zeros_like(new_fc_out.weight.data),
        'exp_avg_sq': torch.zeros_like(new_fc_out.weight.data),
        'step': torch.tensor(0, dtype=torch.int64)
    }

    new_state_bias = {
        'exp_avg': torch.zeros_like(new_fc_out.bias.data),
        'exp_avg_sq': torch.zeros_like(new_fc_out.bias.data),
        'step': torch.tensor(0, dtype=torch.int64)
    }

    if old_fc_out.weight in optimizer.state:
        old_state_weight = optimizer.state.pop(old_fc_out.weight)
        new_state_weight['exp_avg'][:old_output_size] = old_state_weight['exp_avg']
        new_state_weight['exp_avg_sq'][:old_output_size] = old_state_weight['exp_avg_sq']
        new_state_weight['exp_avg'][-1] = old_state_weight['exp_avg'][copy_idx].clone()
        new_state_weight['exp_avg_sq'][-1] = old_state_weight['exp_avg_sq'][copy_idx].clone()
        if 'step' in old_state_weight:
            new_state_weight['step'] = old_state_weight['step']

    if old_fc_out.bias in optimizer.state:
        old_state_bias = optimizer.state.pop(old_fc_out.bias)
        new_state_bias['exp_avg'][:old_output_size] = old_state_bias['exp_avg']
        new_state_bias['exp_avg_sq'][:old_output_size] = old_state_bias['exp_avg_sq']
        new_state_bias['exp_avg'][-1] = old_state_bias['exp_avg'][copy_idx].clone()
        new_state_bias['exp_avg_sq'][-1] = old_state_bias['exp_avg_sq'][copy_idx].clone()
        if 'step' in old_state_bias:
            new_state_bias['step'] = old_state_bias['step']

    # Reassign the new states to the optimizer
    optimizer.state[new_fc_out.weight] = new_state_weight
    optimizer.state[new_fc_out.bias] = new_state_bias

    # Force the optimizer to re-reference the new params
    optimizer.param_groups = optimizer.param_groups