import torch
import torch.nn as nn
from ncps.torch import CfC
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import time
import numpy as np

from ncps.torch.cfc_cell import CfCCell

class CfCModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.rnn_cell = CfCCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x, timespans):
        # x: [batch, seq_len, features]
        # timespans: [batch, seq_len]
        batch_size, seq_len, _ = x.shape
        h_state = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            inputs = x[:, t, :]
            ts = timespans[:, t].view(-1, 1) # Force [batch, 1] to avoid ncps squeeze bug
            h_out, h_state = self.rnn_cell.forward(inputs, h_state, ts)

        return self.fc(h_out)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        # Standard LSTM
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, timespans=None):
        # x: [batch, seq_len, features]
        # LSTM ignores timespans
        out, _ = self.rnn(x)
        # out is [batch, seq_len, hidden_size]
        last_out = out[:, -1, :]
        return self.fc(last_out)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, dt, y in dataloader:
        x, dt, y = x.to(device), dt.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x, dt).squeeze()
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, dt, y in dataloader:
            x, dt, y = x.to(device), dt.to(device), y.to(device)
            preds = model(x, dt).squeeze()
            loss = criterion(preds, y)
            total_loss += loss.item() * x.size(0)

            # Apply sigmoid for classification metrics
            probs = torch.sigmoid(preds)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Metrics
    preds_binary = (all_preds > 0.5).astype(float)
    acc = accuracy_score(all_targets, preds_binary)

    # Handle edge case where all targets are 0 or 1
    if len(np.unique(all_targets)) > 1:
        auc = roc_auc_score(all_targets, all_preds)
        f1 = f1_score(all_targets, preds_binary)
    else:
        auc = 0.5
        f1 = 0.0

    return avg_loss, acc, auc, f1

def train_model(model, dataloaders, epochs=10, lr=0.001, device='cpu'):
    model = model.to(device)
    # Using BCEWithLogitsLoss for binary classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_f1': []}

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
        val_loss, acc, auc, f1 = evaluate(model, dataloaders['val'], criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(auc)
        history['val_f1'].append(f1)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {auc:.4f} - Val F1: {f1:.4f}")

    return history
