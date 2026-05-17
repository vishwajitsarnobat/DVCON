import torch
import torch.nn as nn
import brevitas.nn as qnn

def replace_with_quantized_layers(module, name=''):
    """
    Recursively replaces Conv2d -> QuantConv2d and Linear -> QuantLinear.
    Preserves trained weights and uses a custom pre-hook with a FROZEN mask
    to ensure 2:4 sparsity stays locked in place during QAT.
    """
    replaced_count = 0
    
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        
        # 1. Capture the EXACT frozen mask from Phase 2
        is_pruned = hasattr(child, 'weight_mask')
        saved_mask = child.weight_mask.clone() if is_pruned else None
        
        quant_layer = None
        
        # 2. Build the Brevitas layers
        if isinstance(child, nn.Conv2d):
            bit_width = 16 if 'cv2' in full_name else 8
            quant_layer = qnn.QuantConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                weight_bit_width=bit_width,
                return_quant_tensor=False
            )
        elif isinstance(child, nn.Linear):
            quant_layer = qnn.QuantLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                weight_bit_width=8,
                return_quant_tensor=False
            )
        else:
            replaced_count += replace_with_quantized_layers(child, full_name)
            continue

        # 3. Transfer weights and attach the custom Hook
        if quant_layer is not None:
            # Rescue the healed weights (PyTorch stores pruned weights in 'weight_orig')
            if is_pruned:
                quant_layer.weight.data = child.weight_orig.data.clone()
            else:
                quant_layer.weight.data = child.weight.data.clone()
                
            if child.bias is not None:
                quant_layer.bias.data = child.bias.data.clone()

            if is_pruned and saved_mask is not None:
                # Save the frozen mask as a persistent buffer (moves to GPU automatically)
                quant_layer.register_buffer('sparsity_mask', saved_mask)
                
                # Apply it once immediately
                with torch.no_grad():
                    quant_layer.weight.data *= quant_layer.sparsity_mask
                    
                # The Hook: Runs before every forward pass. 
                # Multiplies the weights by the frozen mask to crush any gradient updates to the zeros.
                def enforce_sparsity_hook(m, inputs):
                    with torch.no_grad():
                        m.weight.data *= m.sparsity_mask
                        
                quant_layer.register_forward_pre_hook(enforce_sparsity_hook)
                
            # Swap the layer in the model
            setattr(module, child_name, quant_layer)
            replaced_count += 1
            
    return replaced_count