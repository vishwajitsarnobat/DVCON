import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class TwoFourSparsityPruner(prune.BasePruningMethod):
    """
    Creates a mask that retains the two largest magnitude weights in every group of 4.
    """
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        original_shape = t.shape
        total = t.numel()
        # Pad to multiple of 4
        pad = (4 - total % 4) % 4
        t_flat = t.flatten()
        if pad:
            t_flat = torch.cat([t_flat, torch.zeros(pad, device=t.device)])
        t_reshaped = t_flat.view(-1, 4)          # each row: 4 consecutive weights
        magnitude = t_reshaped.abs()
        # Keep the two largest magnitudes (largest=True)
        _, indices_largest = torch.topk(magnitude, k=2, dim=1, largest=True)
        mask = torch.zeros_like(t_reshaped)
        mask.scatter_(1, indices_largest, 1)     # set 1 where largest two are
        return mask.view(-1)[:total].view(original_shape)

def apply_2_4_sparsity(model):
    pruned_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            TwoFourSparsityPruner.apply(module, name='weight')
            pruned_layers += 1
    print(f"Successfully applied 2:4 sparsity mask to {pruned_layers} layers!")
    return model