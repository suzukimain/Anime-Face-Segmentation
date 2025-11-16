import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import numpy as np
import cv2 as cv
from PIL import Image
import os

from network import UNet
from dataset import UNetDataset
from util import seg2img

####################################################################
# Constants and hyper parameters
DATA_PATH  = './dataset/output'
SEG_PATH = './dataset/mask'
MODEL_PATH = './model'
MODEL_NAME = 'UNet'
CHECKPOINT_PATH = './model/checkpoint.pth'

INPUT_LEN = 512

LEARNING_RATE = 0.0001
EPOCH = 50
START_EPOCH = 0

TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8
TEST_BATCH_SIZE = 1

R_TRAIN = 0.88; R_VAL = 0.07
# Define transformer
transformer = transforms.Compose([
            transforms.ToTensor(),])
# Load dataset
total_dataset = UNetDataset(img_path=DATA_PATH,seg_path=SEG_PATH, transform=transformer)
# Split train, validation, test
# Compute split sizes so that any rounding remainder goes to the training set
len_total = len(total_dataset)
R_TEST = 1.0 - R_TRAIN - R_VAL
len_val = int(len_total * R_VAL)
len_test = int(len_total * R_TEST)
len_train = len_total - len_val - len_test
train_dataset, validation_dataset, test_dataset = random_split(total_dataset, [len_train, len_val, len_test])
# Build loaders
# Do not drop the last (possibly smaller) batch so that all images are used per epoch
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(validation_dataset, batch_size=VAL_BATCH_SIZE, shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=False)
# Build Model :: In: 3x512x512 -> Out: 6x512x512
model = UNet()
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.98)
criterion = nn.BCEWithLogitsLoss()

# Load checkpoint if exists
if os.path.exists(CHECKPOINT_PATH):
    print(f'Loading checkpoint from {CHECKPOINT_PATH}...')
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    START_EPOCH = checkpoint['epoch'] + 1
    print(f'Resuming training from epoch {START_EPOCH}')
else:
    print('No checkpoint found. Starting training from scratch.')
    # Load pretrained model if available (only when starting fresh)
    if os.path.exists('save/UNet.pth'):
        model.load_state_dict(torch.load('save/UNet.pth'))
        print('Loaded pretrained model from save/UNet.pth')

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        img, seg_target = data
        img = img.cuda()
        seg_target = seg_target.cuda()
        
        optimizer.zero_grad()
        
        pred_seg = model(img)
        loss = criterion(pred_seg, seg_target)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
                
        if batch_idx % 20 == 0:
            scheduler.step()
        
        if batch_idx % 20 == 0:
            # use get_last_lr() to avoid deprecated get_lr() warning
            try:
                lr = float(scheduler.get_last_lr()[0])
            except Exception:
                lr = float(scheduler.get_lr()[0])
            print('Train Epoch: {:>6} [{:>6}/{:>6} ({:>2}%)]\tLoss: {:.6f}\t\t lr: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                int(100. * batch_idx / len(train_loader)), loss.item() / len(data), lr))
    print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / len(train_loader.dataset)))

def save_checkpoint(epoch, model, optimizer, scheduler, path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)
    print(f'Checkpoint saved at epoch {epoch}')
    
def validation():
    model.eval()
    val_loss= 0
    with torch.no_grad():
        for data in val_loader:
            img, seg_target = data
            img = img.cuda()
            seg_target = seg_target.cuda()
            pred_seg = model(img)
            
            # sum up batch loss
            val_loss += criterion(pred_seg, seg_target).item()
            
        
    print('====> Test set loss: {:.8f}'.format(val_loss / len(val_loader.dataset)))

for epoch in range(START_EPOCH, EPOCH):
    train(epoch)
    validation()
    
    # Save checkpoint every epoch
    save_checkpoint(epoch, model, optimizer, scheduler, CHECKPOINT_PATH)
    
    # Save model weights every epoch
    torch.save(model.state_dict(), MODEL_PATH+'/'+MODEL_NAME+f'_ep{epoch}'+'.pth')
    

torch.save(model.state_dict(), MODEL_PATH+'/'+MODEL_NAME+'.pth')

with torch.no_grad():
    model.eval()
    for data in test_loader:
        img, seg_target = data
        img = img.cuda()
        seg_target = seg_target.cuda()
        pred_seg = model(img)
        
        pred_seg = pred_seg.cpu().numpy()
        for i in range(TEST_BATCH_SIZE):
            img = seg2img(np.moveaxis(pred_seg[i],0,2))
            cv.imwrite(f'result{i}.png',img)
        break