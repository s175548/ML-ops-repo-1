from src.models.model import MyAwesomeModel
from torch.utils.data import TensorDataset, DataLoader
import pytest
import torch
from tests import _PATH_DATA
import os

train_images = torch.load(os.path.join(_PATH_DATA,'processed/images_train.pt'))
train_labels = torch.load(os.path.join(_PATH_DATA,'processed/labels_train.pt'))

@pytest.mark.parametrize("batch_size,output_size", [(16, (16,10)), (64, (64,10))])
def check_output_dim(batch_size,output_size):
    model=MyAwesomeModel()
    train_set=TensorDataset(train_images,train_labels)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    output=model(next(iter(trainloader)))
    assert output.shape == output_size

def test_error_on_wrong_shape():
   model=MyAwesomeModel()
   with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
      model(torch.randn(1,2,3))

