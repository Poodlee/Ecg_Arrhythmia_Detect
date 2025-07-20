import argparse
import torch
import numpy as np
import json
import logging
from data_loader import DataLoaderFactory
from model import ModelFactory
from lewansoul_servo_bus import ServoBus

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_random_seeds(seed=7):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def select_class_data(dataset, target_class):
    class_map = {'N': 0, 'S': 1, 'V': 2}
    target_idx = class_map.get(target_class.upper())
    if target_idx is None:
        raise ValueError("Invalid class. Choose from 'N', 'S', or 'V'.")

    indices = [i for i, label in enumerate(dataset.y) if label == target_idx]
    return indices

def infer_single_sample(model, data, device):
    model.eval()
    with torch.no_grad():
        data = {k: v.to(device) for k, v in data.items()}
        output = model(**data)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)
    return pred.item(), probs.cpu().numpy()

def move_servo_if_v(pred):
    try:
        servo_bus = ServoBus('/dev/ttyUSB0')
        if pred == 2:
            logger.info("Detected class V, moving servo from 0 to 120 degrees over 2 seconds")
            servo_bus.move_time_write(1, 120, 2.0)  
    except Exception as e:
        logger.error(f"Servo control failed: {str(e)}")

def main(config):
    
    data_loader = DataLoaderFactory.get_dataloader(config['data_loader']['type'], **config['data_loader']['args'])
    dataset = data_loader.dataset
    
    model = ModelFactory.get_model((config['arch']['type']))
    logger.info('Loading checkpoint: {} ...'.format(config['resume']))    loss_fn = criterion.get_loss()  
    checkpoint = torch.load(config['resume'], weights_only=False)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = config['gpu']
    model = model.to(device)
    model.eval()

    class_map = {0: 'N', 1: 'S', 2: 'V'}
    
    while True:
        target_class = input("Enter the class to find (N, S, V) or 'q' to quit: ").strip()
        if target_class.lower() == 'q':
            logger.info("exit program...")
            break
        try:
            indices = select_class_data(dataset, target_class)
            if not indices:
                logger.warning(f"{target_class} does not exist.")
                continue
            
            selected_idx = np.random.choice(indices)
            data, target = dataset[selected_idx]
            
            data = {k: v.unsqueeze(0).to(device) for k, v in data.items()}  # Add batch dimension
            target = torch.tensor([target])
            
            pred, _ = infer_single_sample(model, data, device)
            
            logger.info(f"Selected sample label {class_map[target.item()]}")
            logger.info(f"Predicted label {class_map[pred]}")
            
            move_servo_if_v(pred)
        except ValueError as e:
            logger.error(str(e))
            continue

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config_infer.json', type=str,
                      help='config file path (default: config_infer.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    config_path = args.parse_args().config
    with open(config_path, 'r') as f:
        config = json.load(f)
    main(config)