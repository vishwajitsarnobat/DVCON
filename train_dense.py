import torch
from torch.utils.data import DataLoader
from model import TaskAwareYOLO
from dataloader import COCOTasksDataset, collate_fn
from train_engine import train_engine 

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n1. Initializing Dense TaskAwareYOLO...")
    model = TaskAwareYOLO(embedding_path='task_embeddings.pt').to(device)

    print("\n2. Loading Dataset...")
    train_dataset = COCOTasksDataset(
        annotation_dir='./dataset/annotations', 
        image_dir='./dataset/train2014',
        img_size=640
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, 
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    print("\n3. Starting Dense Training Phase...")
    trained_model = train_engine(model, train_loader, epochs=10, device=device)

    torch.save(trained_model.state_dict(), 'yolov8n_dense_trained.pt')
    print("\nSaved functionally accurate model to 'yolov8n_dense_trained.pt'")

if __name__ == "__main__":
    main()