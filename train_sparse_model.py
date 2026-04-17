from ultralytics.utils.loss import v8DetectionLoss

def train_sparse_model(model,train_loader,epcohs=5,device="cuda"):
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)