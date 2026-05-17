import torch
from torch.utils.data import DataLoader

from model import TaskAwareYOLO
from quantize_utils import replace_with_quantized_layers
from dataloader import COCOTasksDataset, collate_fn
from train_sparse_model import train_sparse_model

def main():
    # Make sure you have installed brevitas: `uv pip install brevitas`
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize the base architecture
    print("\n1. Initializing architecture...")
    model = TaskAwareYOLO(embedding_path='task_embeddings.pt').to(device)
    
    # 2. Load the HEALED weights from Phase 2
    # Note: strict=False is often needed if the PyTorch version saved pruning buffers differently
    print("\n2. Loading Healed Sparse Weights...")
    model.load_state_dict(torch.load('yolov8n_sparse_healed.pt'), strict=False)
    
    # 3. Apply Brevitas Quantization Surgery
    print("\n3. Converting to Brevitas Quantized Layers (INT8/INT16)...")
    layers_swapped = replace_with_quantized_layers(model)
    print(f"Successfully quantized {layers_swapped} layers!")
    
    # Push the newly created quantized layers to the GPU
    model = model.to(device)
    
    # 4. Load Dataset
    print("\n4. Loading Dataset...")
    train_dataset = COCOTasksDataset(
        annotation_dir='./dataset/annotations',
        image_dir='./dataset/train2014',
        img_size=640
    )

    # During Phase 2, PyTorch just had to calculate standard math. 
    # But in Phase 3, Brevitas does what is called "Fake Quantization." 
    # To simulate the hardware's integer math on your GPU, 
    # Brevitas injects hundreds of hidden rounding, clamping, and scaling operations into the computational graph.
    # So less batch_size
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # 5. Execute Quantization-Aware Training (QAT)
    # We only need ~3 epochs because the model already knows the features, 
    # it just needs to adapt to the integer rounding math.
    print("\n5. Starting QAT Phase...")
    qat_model = train_sparse_model(model, train_loader, epochs=1, device=device)
    
    # 6. Save the final hardware-ready model!
    torch.save(qat_model.state_dict(), 'yolov8n_sparse_qat.pt')
    print("\nSaved Final Hardware-Ready Model to 'yolov8n_sparse_qat.pt'")

if __name__ == "__main__":
    main()