import argparse
import sys


import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from model import MyAwesomeModel


def train(num_epochs=10):
    # TODO: Implement training loop here
    model = MyAwesomeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    train_images= torch.load('../../data/processed/images_train.pt')
    train_labels = torch.load('../../data/processed/labels_train.pt')
    train_set=TensorDataset(train_images,train_labels)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    model.train()
    loss_l = []
    for i in range(num_epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            logits = model(images)
            # Calculate the loss with the logits and the labels
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(running_loss)
        loss_l.append(running_loss)
    plt.plot(range(num_epochs), loss_l)
    plt.savefig('../../reports/figures/train_curve.png')
    torch.save(model.state_dict(), '../../models/trained_model.pt')

if __name__ == '__main__':
    train()
