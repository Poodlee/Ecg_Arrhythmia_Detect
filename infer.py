import argparse
import torch
import numpy as np
import json
import logging
from tqdm import tqdm
from data_loader import DataLoaderFactory
from model import ModelFactory
from loss import LossFactory
import metric as module_metric

import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
from datetime import datetime

from sklearn.metrics import roc_curve, auc
from visualize import GradCam
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_random_seeds(seed=7):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def main(config):
    
    data_loader = DataLoaderFactory.get_dataloader(config['data_loader']['type'], **config['data_loader']['args'])
    model = ModelFactory.get_model((config['arch']['type']))
        
    loss_config = config['loss']
    criterion = LossFactory(
        loss_type=loss_config.get('type', 'bce'),
        alpha=loss_config.get('alpha', 0.25),
        gamma=loss_config.get('gamma', 2.0),
        pos_weight=loss_config.get('pos_weight', None),
        class_weights=loss_config.get('class_weights', None)
    )  
    loss_fn = criterion.get_loss()  
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config['resume']))
    checkpoint = torch.load(config['resume'], weights_only=False)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = config['gpu']
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))

    all_preds, all_targets, all_probs = [], [], []
  
    n_classes = 3
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = {k: v.to(device) for k,v in data.items()}, target.to(device)
            preds = model(**data)
            
            # Save for confusion matrix
            all_probs.append(torch.softmax(preds, dim=1).cpu().numpy())
            all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # computing loss, metrics on test set
            loss = loss_fn(preds, target)
            batch_size = config['data_loader']['args']['batch_size']
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metrics):
                total_metrics[i] += metric(preds, target, n_classes) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)
    })
    logger.info(log)

    ####################
    # === HEAT MAP === #
    ####################
    cf_matrix = confusion_matrix(all_targets, all_preds)
    classes_ = ['N', 'S', 'V']
    N = len(classes_)

    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(cf_matrix.flatten()[i]/np.sum(cf_matrix[int(i/N)])) for i in range(N*N)]
    labels_ = [f'{x}\n{y}' for x, y in zip(group_counts, group_percentages)]
    labels_ = np.asarray(labels_).reshape(N, N)

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cf_matrix,
                vmin=0,
                vmax=cf_matrix.max(),
                annot=labels_,
                linewidths=.5,
                ax=ax,
                fmt='',
                cmap='crest',
                annot_kws={"weight": "bold"},
                xticklabels=classes_,
                yticklabels=classes_
    ).set(title='Confusion Matrix: Inference')

    # Save heatmap
    time = datetime.now().strftime('%Y%m%d-%H%M%S')

    os.makedirs(config['output_dir'], exist_ok=True)
    heatmap_path = os.path.join(config['output_dir'], f'confusion_matrix_heatmap_{time}.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    logger.info(f"Heatmap saved to {heatmap_path}")
 
    #####################
    # === ROC & AUC === #
    #####################
    all_probs = np.concatenate(all_probs, axis=0)
    y_true = np.array(all_targets)
    y_true_onehot = np.eye(N)[y_true]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(N):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(6, 5))
    for i in range(N):
        plt.plot(fpr[i], tpr[i], label=f'{classes_[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Multiclass)')
    plt.legend(loc='lower right')
    rocauc_path = os.path.join(config['output_dir'], f'roc_curve_{time}.png')
    plt.savefig(rocauc_path, dpi=300, bbox_inches='tight')
    logger.info(f"Heatmap saved to {rocauc_path}")

    #####################
    # === GRAD  CAM === #
    #####################
    sample_idx = 0
    
    sample_data, _ = next(iter(data_loader))
    sample = {k: v.to(device) for k,v in sample_data.items()}
    target_layer = model.conv3 if hasattr(model, 'conv3') else list(model.children())[-1]
    gradcam = GradCam(model, target_layer)
    heatmap = gradcam.generate_heatmap(sample, sample_idx=sample_idx)
    overlay = GradCam.overlay_heatmap(sample['x1'], heatmap)
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.title("Original"); plt.imshow(sample['x1'][sample_idx].cpu().squeeze().permute(1,2,0), cmap='gray'); plt.axis('off')
    plt.subplot(1,2,2); plt.title("Grad-CAM"); plt.imshow(overlay.transpose(1,2,0), cmap='jet'); plt.axis('off')
    plt.savefig(os.path.join(config['output_dir'], f'gradcam_sample_{time}.png'), dpi=300, bbox_inches='tight')

    print(f"GRAD CAM result saved to {os.path.join(config['output_dir'], f'gradcam_sample_{time}.png')}")
  

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