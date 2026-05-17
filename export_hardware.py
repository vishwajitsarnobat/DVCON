import torch
import numpy as np
import brevitas.nn as qnn
from model import TaskAwareYOLO
from quantize_utils import replace_with_quantized_layers

def export_to_c_header(model, filename="vega_yolo_weights.h"):
    print("\n4. Exporting model to C-Header...")
    with open(filename, "w") as f:
        f.write("/* AUTO-GENERATED YOLOv8 WEIGHTS FOR VEGA RISC-V */\n")
        f.write("#ifndef VEGA_YOLO_WEIGHTS_H\n#define VEGA_YOLO_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")

        layer_idx = 0
        total_original_bytes = 0
        total_compressed_bytes = 0

        # Export Conv2d and Linear layers
        for name, module in model.named_modules():
            if isinstance(module, (qnn.QuantConv2d, qnn.QuantLinear)):
                # Get quantized integer weights
                int_weight_tensor = module.quant_weight().int(float_datatype=False)
                flat_weights = int_weight_tensor.flatten().cpu().numpy()
                bit_width = int(module.quant_weight_bit_width().item())
                c_type = "int8_t" if bit_width <= 8 else "int16_t"

                # 2:4 sparsity compression
                compressed_weights = []
                bitmasks = []
                for i in range(0, len(flat_weights), 4):
                    block = flat_weights[i:i+4]
                    # Count non-zeros (should be ≤2 because of 2:4 sparsity)
                    nz_indices = [j for j in range(len(block)) if block[j] != 0]
                    # Ensure at most 2 non-zeros
                    assert len(nz_indices) <= 2, f"Block {i//4} has {len(nz_indices)} non-zeros!"
                    mask = sum(1 << j for j in nz_indices)
                    nz_values = [int(block[j]) for j in nz_indices]
                    while len(nz_values) < 2:
                        nz_values.append(0)
                    compressed_weights.extend(nz_values)
                    bitmasks.append(mask)

                # Write weights
                f.write(f"// Layer {layer_idx}: {name} | Precision: {bit_width}-bit\n")
                f.write(f"const {c_type} layer_{layer_idx}_weights[{len(compressed_weights)}] = {{")
                f.write(", ".join(map(str, compressed_weights)))
                f.write("};\n")
                f.write(f"const uint8_t layer_{layer_idx}_masks[{len(bitmasks)}] = {{")
                f.write(", ".join(map(str, bitmasks)))
                f.write("};\n")

                # Export bias if present
                if hasattr(module, 'bias') and module.bias is not None:
                    bias = module.bias.data.cpu().numpy()
                    f.write(f"const {c_type} layer_{layer_idx}_bias[{len(bias)}] = {{")
                    f.write(", ".join(map(str, bias.astype(int))))
                    f.write("};\n")

                # Export quantization scales (for requantization)
                # For brevitas, scale can be obtained from module.quant_weight().scale
                # We'll export it as float (hardware can convert to fixed-point)
                weight_scale = module.quant_weight().scale.cpu().item()
                f.write(f"const float layer_{layer_idx}_weight_scale = {weight_scale:.8f};\n\n")

                total_original_bytes += len(flat_weights) * (bit_width // 8)
                total_compressed_bytes += (len(compressed_weights) * (bit_width // 8)) + len(bitmasks)
                layer_idx += 1

        f.write("#endif // VEGA_YOLO_WEIGHTS_H\n")

    print("\n Hardware Export Complete!")
    print(f"Original Memory:   {total_original_bytes / 1024:.2f} KB")
    print(f"Compressed Memory: {total_compressed_bytes / 1024:.2f} KB")
    print(f"Total BRAM Saved:  {100 - (total_compressed_bytes/total_original_bytes)*100:.1f}%")

def main():
    device = torch.device('cpu')
    print("1. Initializing architecture...")
    model = TaskAwareYOLO(embedding_path='task_embeddings.pt').to(device)

    # Fuse BatchNorm (if any remains)
    print("1.5 Fusing BatchNorm layers for Hardware...")
    model.core_model.fuse()

    print("2. Replacing with Brevitas quantized layers...")
    replace_with_quantized_layers(model)

    print("3. Loading final QAT weights...")
    model.load_state_dict(torch.load('yolov8n_sparse_qat.pt', map_location=device), strict=False)

    # Calibration forward with a realistic dummy (not all zeros)
    print("3.5 Running calibration forward pass...")
    model.eval()
    with torch.no_grad():
        dummy_img = torch.randn(1, 3, 640, 640) * 0.5 + 0.5   # random pixels in [0,1]
        dummy_task = torch.tensor([0])
        _ = model(dummy_img, dummy_task)

    export_to_c_header(model)

if __name__ == "__main__":
    main()