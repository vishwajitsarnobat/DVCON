import torch
from torch.utils.data import DataLoader

from model import TaskAwareYOLO
from prune_utils import apply_2_4_sparsity
from dataloader import COCOTasksDataset, collate_fn
from train_engine import train_engine

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize Model
    print("\n1. Initializing model...")
    model = TaskAwareYOLO(embedding_path='task_embeddings.pt').to(device)
    
    # 2. Apply Sparsity
    print("\n2. Applying 2:4 Structured Sparsity...")
    model = apply_2_4_sparsity(model)
    
    # 3. Load Dataset
    print("\n3. Loading Dataset...")
    train_dataset = COCOTasksDataset(
        annotation_dir='./dataset/annotations', 
        image_dir='./dataset/train2014',
        img_size=640
    )
    
    # Use collate_fn to properly format the bounding boxes for YOLO
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    # 4. Train the model
    print("\n4. Starting training phase...")
    trained_model = train_engine(model, train_loader, epochs=1, device=device)
    
    # 5. Save the weights
    torch.save(trained_model.state_dict(), 'yolov8n_sparse_healed.pt')
    print("\nSaved healed model to 'yolov8n_sparse_healed.pt'")

if __name__ == "__main__":
    main()