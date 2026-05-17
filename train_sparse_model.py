import torch
from types import SimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss

def train_sparse_model(model, train_loader, epochs=5, device="cuda"):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Convert hyperparameter dict to an Object
    # Ensure args exists
    if not hasattr(model.core_model, 'args') or model.core_model.args is None:
        model.core_model.args = {}

    # If it's a dictionary, give it the default YOLOv8 loss gains and convert to SimpleNamespace
    # These are the factors that are multiplied to the respective losses
    if isinstance(model.core_model.args, dict):
        # box loss, classification loss and Distribution Focal Loss weight
        hyp_defaults = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5} # Standard YOLOv8 loss weights
        hyp_defaults.update(model.core_model.args)
        # SimpleNamespace converts dictionary keys into object attributes.
        model.core_model.args = SimpleNamespace(**hyp_defaults)

    # Now the loss function will be happy!
    loss_fn = v8DetectionLoss(model.core_model)

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        epoch_loss = 0.0
        
        for batch_idx, (images, targets, task_ids) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            task_ids = task_ids.to(device) 
            
            # targets is shape (batch_size, 6) -> [batch_idx, class, x, y, w, h]
            batch_dict = {
                "batch_idx": targets[:, 0],   
                "cls": targets[:, 1],         
                "bboxes": targets[:, 2:6]     
            }
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(images, task_ids)
            
            # Compute loss
            loss, loss_items = loss_fn(predictions, batch_dict) 

            # Force the loss into a single scalar
            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.sum()
            
            # Backward pass 
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                # loss_items contains the separated [box_loss, cls_loss, dfl_loss]
                box_loss, cls_loss, dfl_loss = loss_items.tolist()
                
                print(f"Batch {batch_idx}/{len(train_loader)} | Total: {loss.item():.4f} "
                      f"(Box: {box_loss:.4f}, Cls: {cls_loss:.4f}, DFL: {dfl_loss:.4f})")
                
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(train_loader):.4f}")

    print("\nTraining complete. The 2:4 sparse model has healed.")
    return model