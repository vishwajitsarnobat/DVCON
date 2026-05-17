import torch
import torch.nn as nn
import brevitas.nn as qnn
from prune_utils import TwoFourSparsityPruner

def replace_with_quantized_layers(module, name=''):
    """
    Recursively searches through the YOLOv8 architecture.
    Replaces standard Conv2d layers with Brevitas QuantConv2d layers,
    preserves the trained weights, and re-applies the 2:4 sparsity hook.
    """
    replaced_count = 0
    
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        
        if isinstance(child, nn.Conv2d):
            # --- PRECISION ROUTING ---
            # In YOLOv8, the Detect head splits into two paths:
            # 'cv2' layers handle the Bounding Box coordinates (needs high precision)
            # 'cv3' layers handle the Classification scores
            bit_width = 16 if 'cv2' in full_name else 8
            
            # 1. Create the Brevitas Quantized Layer
            quant_conv = qnn.QuantConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                weight_bit_width=bit_width,
                # CRITICAL: YOLO routing breaks if it doesn't get standard PyTorch tensors back
                return_quant_tensor=False 
            )
            
            # 2. Rescue the healed weights from Phase 2
            quant_conv.weight.data = child.weight.data.clone()
            if child.bias is not None:
                quant_conv.bias.data = child.bias.data.clone()
                
            # 3. Re-apply the 2:4 Structured Sparsity hook to the new layer
            TwoFourSparsityPruner.apply(quant_conv, name='weight')
            
            # 4. Swap the layers inside the model
            setattr(module, child_name, quant_conv)
            replaced_count += 1
            
        else:
            # If it's a Sequential or C2f block, dive deeper
            replaced_count += replace_with_quantized_layers(child, full_name)
            
    return replaced_count