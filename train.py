import argparse
import torch
import numpy as np
import json
from data_loader import DataLoaderFactory
from model import ModelFactory
from loss import LossFactory
import metric as module_metric
from trainer import Trainer
from datetime import datetime

def set_random_seeds(seed=7):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def main(config):

    data_loader = DataLoaderFactory.get_dataloader(config['data_loader']['type'], **config['data_loader']['args'])
    valid_data_loader = data_loader.split_validation()
    model = ModelFactory.get_model((config['arch']['type']))

    device = config['gpu']
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = True
    
    loss_config = config['loss']
    criterion = LossFactory(
        loss_type=loss_config.get('type', 'bce'),
        alpha=loss_config.get('alpha', 0.25),
        gamma=loss_config.get('gamma', 2.0),
        pos_weight=loss_config.get('pos_weight', None),
        class_weights=loss_config.get('class_weights', None)
    )

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = getattr(torch.optim, config['optimizer']['type'])
    optimizer = optimizer(trainable_params, **config['optimizer']['args'])

    lr_scheduler = getattr(torch.optim.lr_scheduler, config['lr_scheduler']['type'])
    lr_scheduler = lr_scheduler(optimizer, **config['lr_scheduler']['args'])

    trainer = Trainer(
        model, criterion, metrics, optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
        time = datetime.now().strftime('%Y%m%d-%H%M%S')
    )
    trainer.train()


if __name__ == '__main__':
    set_random_seeds()
    
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: config.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    config_path = args.parse_args().config
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    main(config)
