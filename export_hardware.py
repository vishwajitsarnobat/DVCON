import torch
import numpy as np
import brevitas.nn as qnn

from model import TaskAwareYOLO
from quantize_utils import replace_with_quantized_layers

def export_to_c_header(model, filename="vega_yolo_weights.h"):
    print(f"Exporting model to {filename}...")
    
    with open(filename, "w") as f:
        f.write("/* AUTO-GENERATED YOLOv8 WEIGHTS FOR VEGA RISC-V */\n")
        f.write("#ifndef VEGA_YOLO_WEIGHTS_H\n#define VEGA_YOLO_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")

        layer_idx = 0
        total_original_bytes = 0
        total_compressed_bytes = 0

        for name, module in model.named_modules():
            # We only want to export the Brevitas quantized layers
            if isinstance(module, qnn.QuantConv2d):
                
                # 1. Extract the raw integer weights from Brevitas
                # .int() returns the actual INT8/INT16 tensor, not the float decimals
                int_weight_tensor = module.quant_weight().int(float_datatype=False)
                flat_weights = int_weight_tensor.flatten().cpu().numpy()
                
                bit_width = int(module.quant_weight_bit_width().item())
                c_type = "int8_t" if bit_width <= 8 else "int16_t"
                
                compressed_weights = []
                bitmasks = []

                # 2. Compress using 2:4 Structured Sparsity
                for i in range(0, len(flat_weights), 4):
                    block = flat_weights[i:i+4]
                    mask = 0
                    non_zeros = []

                    # Find the non-zero elements and build the bitmask
                    for j in range(len(block)):
                        if block[j] != 0:
                            mask |= (1 << j)  # Set the j-th bit to 1
                            non_zeros.append(int(block[j]))
                    
                    # Failsafe: If the model naturally learned a zero on top of the pruned zeros, 
                    # we pad it to ensure strict 2:4 hardware alignment
                    while len(non_zeros) < 2:
                        non_zeros.append(0)
                        
                    # We only take the first 2 (in case of weird rounding edge cases)
                    compressed_weights.extend(non_zeros[:2])
                    bitmasks.append(mask)

                # 3. Write the Compressed Weights to the C-Header
                f.write(f"// Layer {layer_idx}: {name} | Precision: {bit_width}-bit\n")
                f.write(f"const {c_type} layer_{layer_idx}_weights[{len(compressed_weights)}] = {{")
                f.write(", ".join(map(str, compressed_weights)))
                f.write("};\n")

                # 4. Write the 4-bit Index Masks (stored efficiently in uint8_t)
                f.write(f"const uint8_t layer_{layer_idx}_masks[{len(bitmasks)}] = {{")
                f.write(", ".join(map(str, bitmasks)))
                f.write("};\n\n")

                # Calculate memory savings
                total_original_bytes += len(flat_weights) * (bit_width // 8)
                total_compressed_bytes += (len(compressed_weights) * (bit_width // 8)) + (len(bitmasks))
                
                layer_idx += 1

        f.write("#endif // VEGA_YOLO_WEIGHTS_H\n")
        
    print("\n✅ Hardware Export Complete!")
    print(f"Original Memory:   {total_original_bytes / 1024:.2f} KB")
    print(f"Compressed Memory: {total_compressed_bytes / 1024:.2f} KB")
    print(f"Total BRAM Saved:  {100 - (total_compressed_bytes/total_original_bytes)*100:.1f}%")

def main():
    device = torch.device('cpu') # Exporting is easier on CPU
    
    print("1. Initializing architecture...")
    model = TaskAwareYOLO(embedding_path='task_embeddings.pt').to(device)
    
    print("2. Rebuilding Brevitas Layers...")
    replace_with_quantized_layers(model)
    
    print("3. Loading final QAT weights...")
    model.load_state_dict(torch.load('yolov8n_sparse_qat.pt', map_location=device), strict=False)
    
    export_to_c_header(model)

if __name__ == "__main__":
    main()