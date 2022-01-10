import sys
import os
sys.path.append('../NN-package')
from src.models.train_model import train
import numpy as np
import hydra
from tests import _PROJECT_ROOT




def test_loss_decline():
    with hydra.initialize(config_path='../src/config'):
        cfg = hydra.compose(config_name='train_test.yaml')
        loss = np.array(train(cfg))
        diff = np.sum(loss[1:]-loss[:-1])
        assert diff < 0



if __name__ == '__main__':
    test_loss_decline()
