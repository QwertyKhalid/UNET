import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import UNET
from utils import SaveInference, LoadInference, Loaders, Accuracy, Predictions


#Optional arguments parser for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--TRAIN_IMAGE_PATH', help='Path to directory containing training data, default is dataset/train_images', type=str)
parser.add_argument('--TRAIN_LABEL_PATH', help='Path to directory containing training labels, default is dataset/train_labels', type=str)
parser.add_argument('--VALID_IMAGE_PATH', help='Path to directory containing validation data, default is dataset/valid_images', type=str)
parser.add_argument('--VALID_LABEL_PATH', help='Path to directory containing validation labels, default is dataset/valid_labels', type=str)
parser.add_argument('--LOAD_MODEL_PATH', help='Path to model if LOAD_MODEL is True, default is models/UNET_model.pt', type=str)
parser.add_argument('--LOAD_MODEL', help='Choose to load model, default is False', type=bool)
parser.add_argument('--IMAGE_WIDTH', help='Define image width for input size limit, default is 512', type=int)
parser.add_argument('--IMAGE_HEIGHT', help='Define image height for input size limit, default is 512', type=int)
parser.add_argument('--OPTIMIZER', help='Select optimizer, options are Adam and SGD, default is Adam', type=str)
parser.add_argument('--LEARNING_RATE', help='Set learning rate, default is 3e-4', type=int)
parser.add_argument('--BATCH_SIZE', help='Set batch size, default is 3', type=int)
parser.add_argument('--EPOCHS', help='Set number of epochs, default is 16', type=int)
parser.add_argument('--NUM_WORKERS', help='Set number of workers, default is 2', type=int)
optargs = parser.parse_args()

#Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_IMAGE_PATH = 'dataset/train_images' if not optargs.TRAIN_IMAGE_PATH else optargs.TRAIN_IMAGE_PATH
TRAIN_LABEL_PATH = 'dataset/train_labels' if not optargs.TRAIN_LABEL_PATH else optargs.TRAIN_LABEL_PATH
VALID_IMAGE_PATH = 'dataset/valid_images' if not optargs.VALID_IMAGE_PATH else optargs.VALID_IMAGE_PATH
VALID_LABEL_PATH = 'dataset/valid_labels' if not optargs.VALID_LABEL_PATH else optargs.VALID_LABEL_PATH
LOAD_MODEL_PATH = 'models/UNET_model.pt' if not optargs.LOAD_MODEL_PATH else optargs.LOAD_MODEL_PATH
LOAD_MODEL = False if not optargs.LOAD_MODEL else optargs.LOAD_MODEL
IMAGE_WIDTH = 512 if not optargs.IMAGE_WIDTH else optargs.IMAGE_WIDTH
IMAGE_HEIGHT = 512 if not optargs.IMAGE_HEIGHT else optargs.IMAGE_HEIGHT
OPTIMIZER = 'Adam' if not optargs.OPTIMIZER else optargs.OPTIMIZER
LEARNING_RATE = 3e-4 if not optargs.LEARNING_RATE else optargs.LEARNING_RATE
BATCH_SIZE = 3 if not optargs.BATCH_SIZE else optargs.BATCH_SIZE
EPOCHS = 16 if not optargs.EPOCHS else optargs.EPOCHS
NUM_WORKERS = 2 if not optargs.NUM_WORKERS else optargs.NUM_WORKERS
PIN_MEMORY = True


#Train function
def Train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        #Forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        #Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def type(str):
    for letter in str:
        print(letter, end='')
        sys.stdout.flush()
        time.sleep(0.02)
    print('\n')

def main():
    type(f'Device {DEVICE}\nOptimizer {OPTIMIZER}\nLearning rate {LEARNING_RATE}\nBatch size {BATCH_SIZE}\nEpochs {EPOCHS}\nWorkers {NUM_WORKERS}')
    type('Preparing to train custom UNET model...')

    train_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2()])

    valid_transforms = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0), ToTensorV2()])

    model = UNET(in_channels=3, out_channels=1).to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if optargs.OPTIMIZER:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_loader, valid_loader = Loaders(
        TRAIN_IMAGE_PATH,
        TRAIN_LABEL_PATH,
        VALID_IMAGE_PATH,
        VALID_LABEL_PATH,
        BATCH_SIZE,
        train_transform,
        valid_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        LoadInference(torch.load(LOAD_MODEL_PATH), model)

    Accuracy(valid_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        Train(train_loader, model, optimizer, loss_fn, scaler)

        #Save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        SaveInference(checkpoint)

        #Get accuracy
        Accuracy(valid_loader, model, device=DEVICE)

        #Save predictions to 'predictions/'
        Predictions(valid_loader, model, folder='predictions/', device=DEVICE)


if __name__ == '__main__':
    main()