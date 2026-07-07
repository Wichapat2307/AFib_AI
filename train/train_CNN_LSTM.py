"""
train_cnn_lstm.py -- Train AFibCNNLSTM on LTAF, test on MITB
"""
from dataset import get_dataloaders
from model import AFibCNNLSTM
from train.trainer import train

EPOCHS     = 30
BATCH_SIZE = 64
LR         = 1e-3
PATIENCE   = 10
SAVE_PATH  = "models/cnn_lstm_best.pth"

train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

model = AFibCNNLSTM()
train(model, train_loader, val_loader, SAVE_PATH, model_name="CNN+LSTM", epochs=EPOCHS, lr=LR, patience=PATIENCE)