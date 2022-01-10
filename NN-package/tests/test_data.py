import torch
import os
from tests import _PATH_DATA

cwd=os.getcwd()
test_images = torch.load(os.path.join(_PATH_DATA,'processed/images_test.pt'))
test_labels = torch.load(os.path.join(_PATH_DATA,'processed/labels_test.pt'))
train_images = torch.load(os.path.join(_PATH_DATA,'processed/images_train.pt'))
train_labels = torch.load(os.path.join(_PATH_DATA,'processed/labels_train.pt'))

def test_load_data_train():
    N_train=40000
    assert train_images.shape[0]== train_labels.shape[0] == N_train

def test_load_data_test():
    N_train=5000
    assert test_images.shape[0]== test_labels.shape[0] == N_train

def test_datapoint_dim():
    dim=(1,28,28)
    assert train_images.shape[1:] == dim

def test_all_labels_represented():
    test_bool=[l in test_labels for l in range(10)]
    train_bool=[l in train_labels for l in range(10)]
    assert (False not in test_bool) and (False not in train_bool)

