import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

def evaluate(model, loader, device):
    model.eval()
    total_loss, total_mae, n = 0.0, 0.0, 0
    for batch in loader:
        # Move to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        y_true = batch["rating"].float()

        # Forward
        y_pred = model(batch).squeeze(-1)

        # Metrics
        loss = F.mse_loss(y_pred, y_true, reduction="sum")
        mae  = F.l1_loss(y_pred, y_true, reduction="sum")

        total_loss += loss.item()
        total_mae  += mae.item()
        n += y_true.numel()

    mse = total_loss / max(n, 1)
    mae = total_mae / max(n, 1)
    rmse = math.sqrt(mse)
    return {"mse": mse, "rmse": rmse, "mae": mae}