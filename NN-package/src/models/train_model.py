
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from src.models.model import MyAwesomeModel
import hydra
import logging
import wandb
from src import _PROJECT_ROOT
import os


@hydra.main(config_path="../config", config_name='config_train.yaml')
def train(confg):
    wandb.init(config=os.path.join(_PROJECT_ROOT,'src/config/config_wandb_train.yaml'),mode=confg.wandb_env)
    log = logging.getLogger(__name__)
    torch.manual_seed(confg.seed)
    model = MyAwesomeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=confg.lr)
    criterion = nn.CrossEntropyLoss()
    train_images= torch.load(os.path.join(_PROJECT_ROOT,'data/processed/images_train.pt'))
    train_labels = torch.load(os.path.join(_PROJECT_ROOT,'data/processed/labels_train.pt'))
    train_set=TensorDataset(train_images,train_labels)
    wandb.watch(model, log_freq=100)
    trainloader = DataLoader(train_set, batch_size=confg.batch_size, shuffle=True)
    model.train()
    loss_l = []
    for i in range(confg.n_epochs):
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
        log.info(f"Epoch {i}: Loss was {running_loss}")
        loss_l.append(running_loss)
    plt.plot(range(confg.n_epochs), loss_l)
    plt.savefig(os.path.join(_PROJECT_ROOT,'reports/figures/train_curve.png'))
    torch.save(model.state_dict(), os.path.join(_PROJECT_ROOT,'models/trained_model.pt'))
    if confg.test:
        return loss_l


if __name__ == '__main__':
    train()
