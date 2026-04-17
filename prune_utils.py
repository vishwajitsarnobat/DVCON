import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class TwoFourSparsityPruner(prune.BasePruningMethod):
    """
    Creates a mask, for group of 4, 1 for 2 largest elements and 0 for smallest elements.
    """
    PRUNING_TYPE='unstructured'
    
    def compute_mask(self,t,default_mask):
        original_shape=t.shape
        t_reshaped=t.view(-1,4) # 2D matrix, with 4 elements in each row
        magnitude=t_reshaped.abs()
        # indices of 2 smallest elements in each row(dim=1)
        _, indices=torch.topk(magnitude,k=2,dim=1,largest=False)
        mask=torch.ones_like(t_reshaped)
        mask.scatter_(1,indices,0) # put 0 where intended, _ for inplace
        return mask.view(original_shape)

def apply_2_4_sparsity(model):
    # actually applies 2:4 sparsity to Conv2d and Linear layers
    pruned_layers=0
    for name,module in model.named_modules():
        # We only want to prune the computational heavyweights (Convs and Linears)
        if isinstance(module,nn.Conv2d) or isinstance(module,nn.Linear):
            TwoFourSparsityPruner.apply(module,name='weight')
            pruned_layers+=1
    print(f"Successfully applied 2:4 sparsity mask to {pruned_layers} layers!")
    return model