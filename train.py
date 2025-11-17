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
import glob
import re

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
BASE_MODEL = ''  # 追加学習したい既存モデルのパス。未指定なら空文字のまま

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
# Build Model :: In: 3x512x512 -> Out: 7x512x512
model = UNet()
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.98)
criterion = nn.BCEWithLogitsLoss()

# Prefer loading a BASE_MODEL for additional training if specified
if BASE_MODEL and os.path.exists(BASE_MODEL):
    print(f'Loading base model from {BASE_MODEL} for additional training...')
    try:
        state = torch.load(BASE_MODEL, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            print(f'Warning: strict load failed ({e}). Retrying with strict=False...')
            missing_unexpected = model.load_state_dict(state, strict=False)
            try:
                missing, unexpected = missing_unexpected  # PyTorch returns (missing, unexpected)
                if missing:
                    print(f'  Missing keys: {missing}')
                if unexpected:
                    print(f'  Unexpected keys: {unexpected}')
            except Exception:
                pass
        START_EPOCH = 0
        print('Base model loaded. Starting additional training from epoch 0.')
    except Exception as e:
        print(f'Warning: Failed to load base model: {e}')
        print('Proceeding without base model (random initialization).')
        START_EPOCH = 0
elif os.path.exists(CHECKPOINT_PATH):
    # Load checkpoint if exists (used when BASE_MODEL is not specified)
    print(f'Loading checkpoint from {CHECKPOINT_PATH}...')
    try:
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        START_EPOCH = checkpoint['epoch'] + 1
        print(f'Resuming training from epoch {START_EPOCH}')
    except RuntimeError as e:
        print(f'Warning: Failed to load checkpoint: {e}')
        print('Model architecture mismatch detected. Starting training from scratch.')
        START_EPOCH = 0
else:
    print('No checkpoint found. Starting training from scratch.')
    # Load pretrained model if available (only when starting fresh)
    if os.path.exists('save/UNet.pth'):
        try:
            model.load_state_dict(torch.load('save/UNet.pth'))
            print('Loaded pretrained model from save/UNet.pth')
        except RuntimeError as e:
            print(f'Warning: Failed to load pretrained model: {e}')
            print('Starting with randomly initialized weights.')

# Detect per-epoch saved models (e.g. model/UNet_ep{n}.pth) and resume from the latest if newer
try:
    ep_pattern = os.path.join(MODEL_PATH, MODEL_NAME + '_ep*.pth')
    ep_files = glob.glob(ep_pattern)
    max_epoch = -1
    max_fp = None
    for fp in ep_files:
        m = re.search(r'_ep(\d+)\.pth$', fp)
        if m:
            try:
                e = int(m.group(1))
            except Exception:
                continue
            if e > max_epoch:
                max_epoch = e
                max_fp = fp
    if max_epoch >= 0:
        # Only override START_EPOCH if this epoch is newer than current START_EPOCH
        if max_epoch + 1 > START_EPOCH:
            try:
                print(f'Found saved epoch model: {max_fp} (epoch {max_epoch}). Loading and resuming from next epoch {max_epoch+1}...')
                state = torch.load(max_fp, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
                if isinstance(state, dict) and 'model_state_dict' in state:
                    state = state['model_state_dict']
                model.load_state_dict(state)
                START_EPOCH = max_epoch + 1
                print(f'Successfully loaded epoch {max_epoch}. START_EPOCH set to {START_EPOCH}.')
            except Exception as e:
                print(f'Warning: failed to load epoch model {max_fp}: {e}')
except Exception:
    pass

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
        # accumulate total loss over samples (for correct per-sample average)
        batch_n = img.size(0)
        train_loss += loss.item() * batch_n
        optimizer.step()
                
        if batch_idx % 20 == 0:
            scheduler.step()
        
        if batch_idx % 20 == 0:
            # use get_last_lr() to avoid deprecated get_lr() warning
            try:
                lr = float(scheduler.get_last_lr()[0])
            except Exception:
                lr = float(scheduler.get_lr()[0])
            # show processed samples count (use (batch_idx+1)*batch_n to include current batch)
            processed = (batch_idx + 1) * batch_n
            total = len(train_loader.dataset)
            percent = int(100. * processed / total) if total > 0 else 0
            print('Train Epoch: {:>6} [{:>6}/{:>6} ({:>2}%)]\tLoss: {:.6f}\t\t lr: {:.6f}'.format(
                epoch, processed, total,
                percent, loss.item(), lr))
    # compute average loss per sample
    avg_loss = train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0
    print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, avg_loss))

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

            # sum up batch loss (per-sample)
            batch_n = img.size(0)
            val_loss += criterion(pred_seg, seg_target).item() * batch_n
            
        
    avg_val = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0
    print('====> Test set loss: {:.8f}'.format(avg_val))

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