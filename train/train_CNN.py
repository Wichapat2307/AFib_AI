"""
train_cnn.py — Train AFibCNN on LTAF, test on MITB
"""
from dataset import get_dataloaders
from model import AFibCNN
from train.trainer import train

EPOCHS     = 30
BATCH_SIZE = 64
LR         = 1e-4
PATIENCE   = 10
SAVE_PATH  = "models/cnn_best.pth"

train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

model = AFibCNN()
train(model, train_loader, val_loader, SAVE_PATH, model_name="CNN", epochs=EPOCHS, lr=LR, patience=PATIENCE)