import argparse
import torch
import numpy as np
import json
import logger
from tqdm import tqdm
from data_loader import DataLoaderFactory
from model import ModelFactory
from loss import LossFactory
import metric as module_metric

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
    checkpoint = torch.load(config['resume'])
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

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            preds = model(data)

            # computing loss, metrics on test set
            loss = loss_fn(preds, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metrics):
                total_metrics[i] += metric(preds, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)
    })
    logger.info(log)


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



####################################
########## PMAT Algorithm ##########
####################################
with torch.no_grad():
    y_true, y_pred = y_test, net.predict(test_ds)
    y_true = np.array(y_true).astype(np.int64)
    y_pred = np.array(y_pred).astype(np.int64)
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
    print(classification_report(y_true, y_pred, digits=4))
    
    import seaborn as sns
classes_ = ['N', 'S', 'V']
N = 3
group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(cf_matrix.flatten()[i]/np.sum(cf_matrix[int(i/N)])) for i in range(N*N)]
labels_ = [f'{y}' for x,y in zip(group_percentages, group_counts)]
#labels_ = [f'{y}' for x,y in zip(group_counts, group_percentages)]
labels_ = np.asarray(labels_).reshape(N,N)
fig, ax = plt.subplots(figsize=(4,3))

sns.heatmap(cf_matrix, 
            vmin=0, 
            vmax=420,
            annot=labels_, 
            linewidths=.5, 
            ax=ax, 
            fmt='', 
            cmap='crest', # Blues, RdBu_r, BuPu, crest
            annot_kws={"weight": "bold"},
            xticklabels=classes_, 
            yticklabels=classes_
           ).set(title='Training on MIT-BIH Testing on STT')


# ROC and AUC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get predicted probabilities
with torch.no_grad():
    y_proba = net.predict_proba(test_ds)  # Probabilities for each class

# Convert true labels to one-hot encoding for multiclass ROC calculation
num_classes = y_proba.shape[1]
y_true_onehot = np.eye(num_classes)[y_true]

# Calculate ROC and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curves
plt.figure()
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f"{classes_[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guess
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
# Gradient-weighted Class Activation Mapping
class GradCam:
    def __init__(self, skorch_net, target_layer):
        """
        Initialize Grad-CAM with the skorch NeuralNetClassifier and the target layer.
        """
        self.net = skorch_net.module_  # Access the PyTorch model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_feature_maps)
        self.target_layer.register_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, input, output):
        """
        Save the feature maps during the forward pass.
        """
        self.feature_maps = output

    def _save_gradients(self, module, grad_input, grad_output):
        """
        Save the gradients during the backward pass.
        """
        self.gradients = grad_output[0]

    def generate_heatmap(self, x, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        """
        self.net.zero_grad()  # Reset gradients on the underlying PyTorch model
        output = self.net(**x)  # Pass input through the PyTorch model

        if class_idx is None:
            class_idx = torch.argmax(output)

        # Backpropagate to get gradients
        class_score = output[:, class_idx]
        class_score.backward()

        # Compute Grad-CAM
        gradients = self.gradients.detach()
        feature_maps = self.feature_maps.detach()
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
                
        cam = torch.sum(weights * feature_maps, dim=1).squeeze() 
        
        if F.relu(cam).sum()!=0:
            cam = F.relu(cam)  # Apply ReLU to focus on positive contributions 
        else:
            cam = 1-cam
        #plt.imshow(cam.to('cpu'))
        
        # Normalize heatmap        
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    @staticmethod
    def overlay_heatmap(image, heatmap, alpha, cmap='jet'):
        """
        Overlay the Grad-CAM heatmap on the original image.
        """
        import cv2
    
        # Convert heatmap to numpy array
        heatmap = heatmap.cpu().numpy()        
        image = image.squeeze().cpu().numpy()  
        # Resize heatmap to match the input image dimensions
        #heatmap_resized = cv2.resize(heatmap, (image.shape[2], image.shape[1]))  # Resize to (H, W)
        heatmap_resized = cv2.resize(heatmap, (120, 120))  # Resize to (H, W)                
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_resized = np.where(heatmap_resized>100,heatmap_resized,heatmap_resized-10)
        
        #heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)  # Apply a colormap
        
        a, b = image.min(), image.max()        
        #heatmap_resized = normalize_tensor(heatmap_resized, a, b)                        
        # Normalize and convert image to range 0-255
        if len(image.shape) == 2:  # If grayscale
            #print('GRAYSCALE')
            image = np.uint8(255 * (image - image.min()) / (image.max() - image.min()))
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel (BGR)
        else:  # If RGB or multi-channel
            image = np.uint8(255 * (image - image.min()) / (image.max() - image.min()))
    
        # Blend the heatmap with the original image
                    
        #plt.imshow(heatmap_colored)
        #overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)                
        return image + heatmap_resized*alpha
    
def read_beat(path):
    path_splitted = path.split('_')
    n, r = int(path_splitted[-2]), int(path_splitted[-1].split('.')[0])    
    ecg = scaled_signals2_mit[n]
    annotations = np.array(ann_list2_mit[n])
    rpeaks = np.array(r_peak_list2_mit[n])
    
     #r_peak_list1_mit[0][idx]
    idx = np.where(rpeaks==r)[0][0]
    #print(annotations[idx])
    
    return ecg[r-110:r+120]

idx = np.array([i for i in range(len(y_pred)) if y_pred[i]==2 and y_test[i]==2])
#print(idx[1000:1100])

from matplotlib.pyplot import cm

iterator = iter(test_ds)
class_idx = 2
i=-1
beat_no = 30599 
while(i<beat_no):
    item=next(iterator)[0]
    i+=1

path = test_infos['x1'][beat_no]
print(path)
beat = read_beat(path)
# Example Usage:
target_layer = net.module_.conv3  # Specify the target convolutional layer
grad_cam = GradCam(skorch_net=net, target_layer=target_layer)

# Prepare inputs for Grad-CAM
sample_x1, sample_x2 = item['x1'], torch.tensor(item['x2']).to('cuda')  # Replace with a valid sample from your dataset

sample_x1 = sample_x1.unsqueeze(0).to('cuda')  # Add batch dimension
sample_x2 = sample_x2.unsqueeze(0).to('cuda')

# Generate Grad-CAM heatmap
heatmap = grad_cam.generate_heatmap({'x1': sample_x1, 'x2': sample_x2}, class_idx=class_idx)  # Replace with the desired class index

# Overlay and visualize
overlayed_image = GradCam.overlay_heatmap(sample_x1, heatmap, alpha=0.3)

plt.figure(figsize=(10,3))
plt.subplot(131)
plt.plot(beat)
plt.subplot(132)
plt.imshow(item['x1'][0], cmap='jet')
plt.subplot(133)
plt.imshow(overlayed_image, cmap='jet')

plt.axis('off')
plt.show()