
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from model import MyAwesomeModel
import argparse
import logging
import wandb


parser = argparse.ArgumentParser(description='Train NN')
parser.add_argument('--learning_rate')
parser.add_argument('--optimizer')
parser.add_argument('--batch_size')
parser.add_argument('--epochs')
parser.add_argument('--log_interval')
# Get the hyperparameters
args = parser.parse_args()

# Pass them to wandb.init
wandb.init(config=args)


log = logging.getLogger(__name__)
optimizers={'adam':torch.optim.Adam,'sgd':torch.optim.SGD}

def train():
    torch.manual_seed(1)
    model = MyAwesomeModel()
    wandb.watch(model, log_freq=100)
    optimizer = optimizers[wandb.config.optimizer](model.parameters(),wandb.config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_images= torch.load(f'../../data/processed/images_train.pt')
    train_labels = torch.load(f'../../data/processed/labels_train.pt')
    train_set=TensorDataset(train_images,train_labels)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=wandb.config.batch_size, shuffle=True)
    model.train()
    loss_l = []
    for i in range(wandb.config.epochs):
        running_loss = 0
        for batch_idx, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % wandb.config.log_interval == 0:
                wandb.log({"loss": loss})
                validate(model, wandb)


def validate(model,wandb):
    test_images = torch.load(f'../../data/processed/images_test.pt')
    test_labels = torch.load(f'../../data/processed/labels_test.pt')
    test_set = TensorDataset(test_images,test_labels)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=wandb.config.batch_size, shuffle=True)
    with torch.no_grad():
        equals_l=torch.tensor([])
        for images, labels in testloader:
                output=model(images)
                predictions=torch.argmax(output,dim=1)
                equals = predictions == labels
                equals_l=torch.cat((equals_l,equals))
        accuracy=torch.mean(equals_l)
        wandb.log({"validation_accuracy": accuracy})

if __name__ == '__main__':
    train()
