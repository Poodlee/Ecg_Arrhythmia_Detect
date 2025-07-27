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
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_random_seeds(seed=7):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def main(config):
    set_random_seeds()
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
        for i, (x1, x2, target) in enumerate(tqdm(data_loader)):
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            preds = model(x1, x2)
            
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
    labels_ = [f'{x}' for x in group_counts]
    labels_ = np.asarray(labels_).reshape(N, N)

    # Get ACC and F1-score from log
    acc = log.get('accuracy_multiclass', None)
    f1 = log.get('f1_score_macro', None)
    sensitivity = log.get('sensitivity_macro', None)
    title_str = 'Confusion Matrix: Inference'
    title_str += f' (ACC={acc:.3f}, F1={f1:.3f}), Recall={sensitivity:.3f}'

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
    )

    ax.set_xlabel("Prediction", fontsize=12, weight="bold")
    ax.set_ylabel("Label", fontsize=12, weight="bold")
    ax.set_title(title_str, fontsize=13, weight="bold")

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
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import deprocess_image

    from torchvision.transforms.functional import normalize
    sample_idx = 0
    
    x1, x2, _ = next(iter(data_loader))
    x1, x2 = x1.to(device), x2.to(device)
            
    target_layer = model.conv3 if hasattr(model, 'conv3') else list(model.children())[-1]

    class WrappedModel(torch.nn.Module):
        def __init__(self, original_model, x2):
            super().__init__()
            self.original_model = original_model
            self.x2 = x2
        
        def forward(self, x1):
            return self.original_model(x1, self.x2)
    
    wrapped_model = WrappedModel(model, x2)

    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])

    model.eval()
    with torch.no_grad():
        outputs = model(x1, x2)
        preds = torch.argmax(outputs, dim=1)


    targets = [ClassifierOutputTarget(preds[sample_idx].item())]

    # Generate CAM
    grayscale_cam = cam(
        input_tensor=x1,
        targets=targets,
        aug_smooth=True,
        eigen_smooth=False
    )[sample_idx]  # shape: (H, W)

    # Prepare original image
    rgb_image = x1[sample_idx].cpu().permute(1, 2, 0).numpy()
    
    # Normalize image to [0,1] range
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
    rgb_image = np.clip(rgb_image, 0, 1)  # Ensure values are in valid range
    
    # Overlay CAM on image
    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.5  # Balance between image and heatmap
    )
    
    # Create visualization
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(rgb_image)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title(f"Grad-CAM (Class {preds[sample_idx].item()})")
    plt.imshow(visualization)
    plt.axis("off")

    os.makedirs(config['output_dir'], exist_ok=True)
    output_path = os.path.join(
        config['output_dir'],
        f'gradcam_sample_{time}.png'
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM visualization saved to {output_path}")  

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