import argparse
import sys


import torch
from torch.utils.data import TensorDataset
from src.models.model import MyAwesomeModel
import hydra
import logging

log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name='config_predict.yaml')
def predict_model(config):
    torch.manual_seed(config.seed)
    cwd=hydra.utils.get_original_cwd()
    model=MyAwesomeModel()
    model.load_state_dict(torch.load(f'{cwd}/models/trained_model.pt'))
    model.eval()
    test_images = torch.load(f'{cwd}/data/processed/images_test.pt')
    test_labels = torch.load(f'{cwd}/data/processed/labels_test.pt')
    test_set = TensorDataset(test_images[:config.n_samples],test_labels[:config.n_samples])
    testloader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=True)
    with torch.no_grad():
        equals_l=torch.tensor([])
        for images, labels in testloader:
                output=model(images)
                predictions=torch.argmax(output,dim=1)
                equals = predictions == labels
                equals_l=torch.cat((equals_l,equals))
        accuracy=torch.mean(equals_l)
        log.info(f'Test accuracy {accuracy}%')

if __name__ == '__main__':
    predict_model()