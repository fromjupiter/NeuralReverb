import os
import csv
import yaml
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from datetime import datetime
# Custom packages
import neuralnet.vocab
import neuralnet.utils
import neuralnet.models

from neuralnet.data_loader import get_loader
from neuralnet.train import train, runStats

def load_config():
    """
    Load the configuration from config.yaml.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser(description='Worker to train our models')
    parser.add_argument('-c', '--config', required=True, help='config file path')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader)
    config['save_path'] = config['save_path'] + '_' + timestamp + '.model' # add timestamp to model
    config['trace_path'] = config['trace_path'] + '_' + timestamp + '.data'
    logging.basicConfig(filename='out/worker_{}.log'.format(timestamp), 
                        level=logging.INFO, 
                        format='[%(asctime)s] %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info("Configuration file: {}".format(args.config))
    logging.info("Using model {}".format(config['model']))
    logging.info("Epochs: {}, Early-stop epoch: {}, Learning rate: {}, Restore path: {}".format(config['train_epoch'], config['early_stop_epoch'], config['learning_rate'], config['restore_path']))
    return config

def model_loader(config):
    """
    If restore_path is specificed in config, returns saved model
    Else, loads new model
    """
    if os.path.exists(config['restore_path']):
	    return torch.load(config['restore_path'])
    else:
        return getattr(neuralnet.model, config['model_name'])(config)

def build_loader(config):
    """
    Build and return DataLoader
    """
    return get_loader(name=config['name'], 
                datadir = config['datadir']
                batch_size = config['batch_size'], 
                num_workers = config['num_workers']) 

def main():
    # Load Data
    config                = load_config()
    train_loader          = build_loader(config['train_dataset'])
    val_loader            = build_loader(config['val_dataset'])

    model                 = model_loader(config)

    logging.info("Using {} as device".format(config.get('device', 'cpu')))
    model=model.to(device)

    runStats(model, val_loader)
    # Train
    train(model, train_loader, val_loader, config)
    logging.info("Worker completed!")

if __name__=='__main__':
    main()