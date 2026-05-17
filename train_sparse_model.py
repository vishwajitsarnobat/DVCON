import torch
from types import SimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
# Use the new recommended GradScaler API
from torch.amp import GradScaler, autocast

def train_sparse_model(model, train_loader, epochs=5, device="cuda"):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    device_str = device if isinstance(device, str) else device.type
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(device_str)

    # Convert model.core_model.args to SimpleNamespace if it's a dict ---
    if hasattr(model.core_model, 'args'):
        if isinstance(model.core_model.args, dict):
            # Convert dict to SimpleNamespace
            model.core_model.args = SimpleNamespace(**model.core_model.args)
    else:
        model.core_model.args = SimpleNamespace()

    # Set default YOLOv8 loss weights (only if not already present)
    hyp_defaults = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}
    for k, v in hyp_defaults.items():
        if not hasattr(model.core_model.args, k):
            setattr(model.core_model.args, k, v)

    loss_fn = v8DetectionLoss(model.core_model)
    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        epoch_loss = 0.0

        for batch_idx, (images, targets, task_ids) in enumerate(train_loader):
            images = images.to(device_str)
            targets = targets.to(device_str)
            task_ids = task_ids.to(device_str)

            batch_dict = {
                "batch_idx": targets[:, 0],
                "cls": targets[:, 1],
                "bboxes": targets[:, 2:6]
            }

            optimizer.zero_grad()

            # Mixed precision training
            with autocast(device_str):
                predictions = model(images, task_ids)
                loss, loss_items = loss_fn(predictions, batch_dict)
                # loss is already a scalar, but keep safe
                if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                    loss = loss.sum()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                box_loss, cls_loss, dfl_loss = loss_items.tolist()
                print(f"Batch {batch_idx}/{len(train_loader)} | Total: {loss.item():.4f} "
                      f"(Box: {box_loss:.4f}, Cls: {cls_loss:.4f}, DFL: {dfl_loss:.4f})")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

        scheduler.step()

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), 'best_checkpoint.pt')
            print("🌟 New best model saved to 'best_checkpoint.pt'")

    print("\nTraining complete. Loading best checkpoint...")
    model.load_state_dict(torch.load('best_checkpoint.pt'))
    print("Loaded best weights for final export.")
    return model