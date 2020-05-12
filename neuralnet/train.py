import os
import gc
import time
import torch
import pickle
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as TF

from datetime import datetime
from neuralnet.analysis import MultiscaleFFT
from collections import namedtuple
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

def checkGPU():
    logging.info("GPU Usage - cached:{}/{} GB, allocated:{}/{} GB".format(round(torch.cuda.memory_cached()/1024/1024/1024,5), 
                round(torch.cuda.max_memory_cached()/1024/1024/1024,5), 
                round(torch.cuda.memory_allocated()/1024/1024/1024, 5), 
                round(torch.cuda.max_memory_allocated()/1024/1024/1024, 5)))

# evaluate given model using given data loader
def evaluate(model, data_loader):
    logging.info('Running stats')
    model.eval()
    with torch.no_grad():
        return _evaluate(model, data_loader)

def _evaluate(model, data_loader):
    print("_evaluate not implemented yet!")
    return 0

def train(model, train_loader, val_loader, config):
    device       = config.get('device', 'cpu')
    optimizer    = optim.Adam(model.parameters(), lr=config['learning_rate'])
    stop_counter = 0
    prev_loss    = float('inf')
    
    if os.path.exists(config['trace_path']):
        with open(config['trace_path'],'rb') as fh:
            result = pickle.load(fh, encoding='bytes')
    else:
        result = []
    
    logging.info("Start training models for {} epochs.".format(config['train_epoch']))
    for epoch in range(1,config['train_epoch']+1):
        model.train()
        ts = time.time()
        train_loss = 0.0
        counter = 0
        for iter, (y, y_orig) in enumerate(train_loader):
            y = y.to(device)
            y_orig = y_orig.to(device)
            model.set_input(y, y_orig)
            model.optimize_parameters()
            if iter % 200 == 0:
                logging.info("epoch{}, iter{}, loss: {}".format(epoch, iter, model.loss.item()))
            counter += 1
            train_loss += model.loss.item()
        
        train_loss /= max(1,counter)

        if device=='cuda':
            torch.cuda.empty_cache()
        val_loss = evaluate(model, val_loader)
        logging.info('Epoch {}, train loss - {}, validation loss - {}'.format(epoch, train_loss, val_loss))
        result.append((train_loss, val_loss))
        with open(config['trace_path'], 'wb') as fh:
            pickle.dump(result, fh, protocol=4)
        logging.info("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        checkGPU()
        if val_loss > prev_loss:
            stop_counter += 1
        else:
            torch.save(model, config['save_path'])
            stop_counter = 0
        if stop_counter >= config.get('early_stop_epoch'):
            logging.info("Early stopping..")
            break
        prev_loss = val_loss

    return result