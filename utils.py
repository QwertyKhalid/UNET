import torch
import torchvision
from torch.utils.data import DataLoader
from data import MyDataset


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def SaveInference(state, filename='UNET_model.pt'):
    print('Saving checkpoint...')
    torch.save(state, filename)
    print(f'Checkpoint saved as {filename}')

def LoadInference(checkpoint, model):
    print('Loading checkpoint...')
    model.load_state_dict(checkpoint['state_dict'])
    print('Checkpoint loaded')

def Loaders(
    train_path, trainmask_path,
    valid_path, validmask_path,
    batch_size,
    train_transform, valid_transform,
    num_workers=4, pin_memory=True):

    train_dataset = MyDataset(
        image_path=train_path,
        mask_path=trainmask_path,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    valid_dataset = MyDataset(
        image_path=valid_path,
        mask_path=validmask_path,
        transform=valid_transform
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, valid_loader

def Accuracy(loader, model, device=DEVICE):
    correct = 0
    pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            predict = torch.sigmoid(model(x))
            predict = (predict > 0.5).float()
            correct += (predict == y).sum()
            pixels += torch.numel(predict)
            dice_score += (2 * (predict * y).sum()) / ((predict + y).sum() + 1e-8)

    print(f'Predicted {correct}/{pixels} with accuracy {correct/pixels*100:.2f}')
    print(f'Dice score is {dice_score/len(loader)}')
    model.train()

def Predictions(loader, model, folder='predictions/', device=DEVICE):
    model.eval()
    for index, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            predict = torch.sigmoid(model(x))
            predict = (predict > 0.5).float()
        torchvision.utils.save_image(predict, f'{folder}/predictions{index}.png')
        torchvision.utils.save_image(y.unsqueeze(1), f'{folder}{index}.png')

    model.train()