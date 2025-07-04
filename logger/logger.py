# wandb logging 

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as tr
import torchvision.models as models

from IPython.display import clear_output
from model.metric import macro_metrics, PerClassMetrics  # metric 함수들 임포트 필요


# def seed_everything(seed = 21):
#     random.seed(seed)
#     np.random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True

#data loader
# def make_loader(batch_size, train = True, shuffle = True):
#     full_dataset = torchvision.datasets.MNIST(root='./data/MNIST',
#                                               train=train,
#                                               download=True,
#                                               transform=tr.ToTensor())
#     loader = DataLoader(dataset = full_dataset, batch_size = batch_size, shuffle = shuffle, pin_memory = True)
#     return loader

# # Total params: 30,762
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(1,16,3)
#         self.conv2 = nn.Conv2d(16,32,3)
#         self.fc1 = nn.Linear(32*5*5, 32)
#         self.fc2 = nn.Linear(32,10)
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x,2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x,2)
#         x = x.view(-1, 32*5*5)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
# # Total params: 53,018
# class MLPNet(nn.Module):
#     def __init__(self):
#         super(MLPNet, self).__init__()
#         self.fc1 = nn.Linear(784, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 16)
#         self.fc4 = nn.Linear(16, 10)

#     def forward(self, x):
#         x = x.float()
#         x = F.relu(self.fc1(x.view(-1, 784)))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         return x

#test training model, new model_train and model_evaluate should replace these funcs.  
# def model_train(model, 
#                 data_loader, 
#                 criterion, 
#                 optimizer, 
#                 device, 
#                 scheduler=None, 
#                 tqdm_disable=False):
#     """
#     Model train (for multi-class classification)

#     Args:
#         model (torch model)
#         data_loader (torch dataLoader)
#         criterion (torch loss)
#         optimizer (torch optimizer)
#         device (str): 'cpu' / 'cuda' / 'mps'
#         scheduler (torch scheduler, optional): lr scheduler. Defaults to None.
#         tqdm_disable (bool, optional): if True, tqdm progress bars will be removed. Defaults to False.

#     Returns:
#         loss, accuracy: Avg loss, acc for 1 epoch
#     """
#     model.train()

#     running_loss = 0
#     correct = 0

#     for X, y in tqdm(data_loader, disable=tqdm_disable):
#         X, y = X.to(device), y.to(device)

#         optimizer.zero_grad()

#         output = model(X)
#         loss = criterion(output, y)
#         loss.backward()
#         optimizer.step()

#         # multi-class classification
#         _, pred = output.max(dim=1)
#         correct += pred.eq(y).sum().item()
#         running_loss += loss.item() * X.size(0)

#     if scheduler:
#         scheduler.step()

#     accuracy = correct / len(data_loader.dataset) # Avg acc
#     loss = running_loss / len(data_loader.dataset) # Avg loss

#     return loss, accuracy


# def model_evaluate(model, 
#                    data_loader, 
#                    criterion, 
#                    device):
#     """
#     Model validate (for multi-class classification)

#     Args:
#         model (torch model)
#         data_loader (torch dataLoader)
#         criterion (torch loss)
#         device (str): 'cpu' / 'cuda' / 'mps'

#     Returns:
#         loss, accuracy: Avg loss, acc for 1 epoch
#     """
#     model.eval()

#     with torch.no_grad():
#         running_loss = 0
#         correct = 0

#         sample_batch = []
#         sample_label = []
#         sample_prediction = []

#         for i, (X, y) in enumerate(data_loader):
#             X, y = X.to(device), y.to(device)

#             output = model(X)

#             # multi-class classification
#             _, pred = output.max(dim=1)
#             correct += torch.sum(pred.eq(y)).item()
#             running_loss += criterion(output, y).item() * X.size(0)

#             if i == 0:
#                 sample_batch.append(X)
#                 sample_label.append(y)
#                 sample_prediction.append(pred)

#         accuracy = correct / len(data_loader.dataset) # Avg acc
#         loss = running_loss / len(data_loader.dataset) # Avg loss

#         return loss, accuracy, sample_batch[0][:16], sample_label[0][:16], sample_prediction[0][:16]

# #Function to convert config dictionary to string for wandb run name
def map_dict_to_str(config):
    config_str = ', '.join(f"{key}: {value}" for key, value in config.items() if key not in ['dataset', 'epochs', 'batch_size'])
    return config_str


# Change config here
# config = {'dataset': 'MNIST',
#           'model': 'CNN',
#           'epochs': 10,
#           'batch_size': 64,
#           'optimizer': 'sgd',
#           'learning_rate': 1e-2,
#           'weight_decay': 0}


# Function to initialize wandb run with config
# def run(config):
#     # Import wandb and login
#     import wandb
#     wandb.login()
#     wandb.init(project='ppp6131-yonsei-university', config=config)
#     wandb.run.name = map_dict_to_str(config)

#     print('------')
#     print(map_dict_to_str(config))
#     print('------\n')

#     #config에 맞는 optimizer, model 정의
#     config = wandb.config
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     train_loader = make_loader(batch_size=config.batch_size, train=True)
#     test_loader = make_loader(batch_size=config.batch_size, train=False)
    
#     if config.model == 'CNN':
#         model = ConvNet().to(device)
#     if config.model == 'MLP':
#         model = MLPNet().to(device)

#     criterion = nn.CrossEntropyLoss()

#     if config.optimizer == 'sgd':
#         optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
#     if config.optimizer == 'adam':
#         optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
#     if config.optimizer == 'adamw':
#         optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

#     #선택된 model이 wandb.watch()에 전잘됨, gradient 정보 확인 가능
#     wandb.watch(model, criterion, log="all")

#     #모델 학습 진행
#     max_loss = np.inf

#     #wandb.log()로 우리의 metric 기록 가능.
#     #wandb.log()에 dictionary로 한번에 전달가능 / 여기서는 따로따로 logging 진행
#     #step = epoch+1 전달시, 그래프으 x축이 epoch와 동일해짐. (1~config.epochs)
#     for epoch in range(config.epochs):
#         train_loss, train_acc = model_train(model, train_loader, criterion, optimizer, device, None)
#         val_loss, val_acc, sample_batch, sample_label, sample_prediction = model_evaluate(model, test_loader, criterion, device)

#         #커스텀 메트릭 계산 (sigmoid 후 multi-label metric으로 가정)
#         with torch.no_grad():
#             sigmoid = torch.nn.Sigmoid()
#             preds_tensor = sigmoid(sample_prediction)
#             labels_tensor = sample_label.float()

#             metrics = macro_metrics(preds_tensor, labels_tensor, threshold=0.7)

#         wandb.log({
#             "Train Loss": train_loss,
#             "Train Accuracy": train_acc,
#             "Validation Loss": val_loss,
#             "Validation Accuracy": val_acc,
#             "Val Macro Accuracy": metrics["accuracy"],
#             "Val Macro Sensitivity": metrics["sensitivity"],
#             "Val Macro Precision": metrics["precision"],
#             "Val Macro F1": metrics["f1_score"],
#             "examples": [wandb.Image(image, caption=f"Pred: {pred}, Label: {label}")
#                          for image, pred, label in zip(sample_batch, sample_prediction, sample_label)]
#         }, step=epoch + 1)

#         # 모델 저장
#         if val_loss < max_loss:
#             max_loss = val_loss
#             torch.save(model.state_dict(), 'Best_Model.pth')

#     #최종 평가: Per-class metric 계산 (각 epoch 별로 best model 고른 후, class별 성능 계산)
#     model.load_state_dict(torch.load('Best_Model.pth', map_location=device))
#     model.eval()

#     perclass = PerClassMetrics(num_classes=sample_label.shape[1])
#     for images, labels in test_loader:
#         with torch.no_grad():
#             images = images.to(device)
#             labels = labels.to(device).float()
#             preds = sigmoid(model(images))
#             perclass.update(preds, labels, threshold=0.7)

#     final_metrics = perclass.compute_metrics()

#     #Per-class metric wandb에 기록
#     for i in range(config.get("num_classes", sample_label.shape[1])):
#         wandb.log({
#             f"Class_{i}/Precision": final_metrics["precision"][i],
#             f"Class_{i}/Sensitivity": final_metrics["sensitivity"][i],
#             f"Class_{i}/F1": final_metrics["f1_score"][i],
#             f"Class_{i}/Accuracy": final_metrics["accuracy"][i],
#         })

#     # Macro 평균 기록
#     wandb.log({
#         "Best Test Loss": val_loss,
#         "Best Test Accuracy": val_acc,
#         "Best Macro Precision": sum(final_metrics["precision"]) / len(final_metrics["precision"]),
#         "Best Macro Sensitivity": sum(final_metrics["sensitivity"]) / len(final_metrics["sensitivity"]),
#         "Best Macro F1": sum(final_metrics["f1_score"]) / len(final_metrics["f1_score"]),
#         "Best Macro Accuracy": sum(final_metrics["accuracy"]) / len(final_metrics["accuracy"]),
#     })

#     return 'Done'

#hyperparameter 조합 모두 실험 가능. 아래는 example.
# model_list = ['CNN', 'MLP']
# optimizer_list = ['sgd', 'adam', 'adamw']
# learning_rate_list = [1e-2, 1e-3, 1e-4]
# weight_decay_list = [0, 1e-2]

# for model in model_list:
#     for optimizer in optimizer_list:
#         for learning_rate in learning_rate_list:
#             for weight_decay in weight_decay_list:
#                 config = {'dataset': 'MNIST',
#                           'model': model,
#                           'epochs': 10,
#                           'batch_size': 64,
#                           'optimizer': optimizer,
#                           'learning_rate': learning_rate,
#                           'weight_decay': weight_decay}

#                 run(config)
#                 clear_output(wait=True)


import wandb
import torch
import os

class WandbWriter:
    def __init__(self, log_dir, logger, config, project_name="ppp6131-yonsei-university"):
        """
        WandbWriter for logging metrics, images, and models to Weights & Biases.
        
        Args:
            log_dir (str): Directory to save model checkpoints.
            logger: Logger object for debugging (e.g., logging.Logger).
            config (dict): Configuration dictionary for wandb.init and experiment settings.
            project_name (str): W&B project name (default: 'ppp6131-yonsei-university').
        """
        self.log_dir = log_dir
        self.logger = logger
        self.config = config

        self._scope_steps = {}

        # Initialize wandb
        wandb.login()
        wandb.init(project=project_name, config=config)
        wandb.run.name = self._map_dict_to_str(config)
        self.logger.info(f"W&B run initialized: {wandb.run.name}")

    def _map_dict_to_str(self, config):
        """Convert config dict to a string for run name."""
        parts = []
        for key, value in config.config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    parts.append(f"{key}_{subkey}={subvalue}")
            else:
                parts.append(f"{key}={value}")
        return "_".join(parts)

    def set_step(self, step, scope=None):
        if scope is None:
            self._current_step = step
        else:
            self._scope_steps[scope] = step

    def get_step(self, scope=None):
        return self._scope_steps.get(scope, self._current_step)

    def add_scalar(self, key, value, step=None):
        wandb.log({key: value}, step=step if step is not None else self._current_step)

    def add_image(self, key, image, caption=None, step=None):
        wandb.log({key: wandb.Image(image, caption=caption)}, step=step if step is not None else self._current_step)

    def add_histogram(self, name, values, step=None):
        """
        Logs a histogram. wandb.histogram() expects a 1D tensor or numpy array.
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        wandb.log({name: wandb.Histogram(values)}, step=step if step is not None else self._current_step)

    def watch(self, model, criterion=None, log="all"):
        wandb.watch(model, criterion, log=log)

    def save_model(self, model, path="Best_Model.pth"):
        save_path = os.path.join(self.log_dir, path)
        torch.save(model.state_dict(), save_path)
        wandb.save(save_path)
        self.logger.info(f"Saved model to {save_path} and uploaded to W&B")

    def log(self, metrics, step=None):
        wandb.log(metrics, step=step if step is not None else self._current_step)

    def finish(self):
        wandb.finish()
        
        