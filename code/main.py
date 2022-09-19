"""
iceberg classifier Kaggle competition
Created by Elisha Goldstein 14/9/2022
"""
import os
import os.path as osp
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
from Net import Net
from data_utils import augmentations
from Custom_dataset import IcebergImageDataset
from torch.utils.data import DataLoader


def training_loop(dataloader, model, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for batch, data in enumerate(dataloader):
        # get the images and labels
        images, y = data
        images, y = images.to(device), y.to(device).squeeze()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        # print('[%d, %5d] loss: %.3f' %
        #       (epoch + 1, batch + 1, running_loss / 2000))
        # running_loss = 0.0


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for images, y in dataloader:
            images, y = images.to(device), y.to(device)
            pred = model(images)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def run():
    train_json = r'C:\Users\elish\OneDrive\Documents\DS and ML\iceberg_classifier\statoil-iceberg-classifier-challenge\data\processed\train.json'
    data_df = pd.read_json(train_json)
    train_df, test_df = train_test_split(data_df, test_size=.2)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    # train_df = train_df.loc[:200, :]  # for debugging
    # test_df = test_df.loc[:10, :]  # for debugging

    training_dataset = IcebergImageDataset(train_df, transform=augmentations())
    test_dataset = IcebergImageDataset(test_df, )

    train_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = Net(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    epochs = 20
    for epoch in range(epochs):
        print("epoch", epoch)

        training_loop(train_dataloader, model, criterion, optimizer, device, epoch)
        test_loop(test_dataloader, model, criterion, device)

    # save last checkpoint
    os.makedirs(osp.join('..', 'checkpoints'), exist_ok=True)
    torch.save(model.state_dict(), osp.join('..', 'checkpoints', 'model_weights.pth'))
    print('Finished Training')


if __name__ == '__main__':
    run()
