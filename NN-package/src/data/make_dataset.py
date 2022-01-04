# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
import os
import numpy as np
from torchvision import  transforms



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    images_train, labels_train = load_data(set='train',path=input_filepath)
    images_test, labels_test = load_data(set='test',path=input_filepath)
    torch.save(images_train,os.path.join(output_filepath,'images_train.pt'))
    torch.save(images_test, os.path.join(output_filepath, 'images_test.pt'))
    torch.save(labels_train, os.path.join(output_filepath, 'labels_train.pt'))
    torch.save(labels_test, os.path.join(output_filepath, 'labels_test.pt'))


def load_data(set='train',path=None):
    files=[file for file in os.listdir(path) if file[0] != '.']
    data=False
    for file in files:
        if (set=='test') & (file[:4]=='test'):
            data = np.load(os.path.join(path,file))
            images = data['images']
            labels = data['labels']
        elif (set=='train') & (file[:4] !='test'):
            if data:
                data =np.load(os.path.join(path,file))
                images = np.vstack((images, data['images']))
                labels= np.append(labels,data['labels'])
            else:
                data = np.load(os.path.join(path,file))
                images = data['images']
                labels= data['labels']

    images=torch.tensor((images-0.5)/0.5)
    images=images.view(-1, 1, 28, 28)
    labels=torch.tensor(labels)
    return images,labels



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


