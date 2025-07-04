import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
from model.loss import LossSelector as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
def set_random_seeds(seed=7):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def main(config):

    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    model = config.init_obj('arch', module_arch)

    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        print(f"🧪 Using DataParallel on {len(device_ids)} GPUs: {device_ids}")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        print(f"🧪 Using single GPU: {device}")

    loss_config = config['loss']
    criterion = module_loss(
        loss_type=loss_config.get('type', 'bce'),
        alpha=loss_config.get('alpha', 0.25),
        gamma=loss_config.get('gamma', 2.0),
        pos_weight=loss_config.get('pos_weight', None),
        class_weights=loss_config.get('class_weights', None)
    ).get_loss()

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    
    trainer = Trainer(
        model, criterion, metrics, optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler
    )

    trainer.train()



if __name__ == '__main__':
    set_random_seeds()
    
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
