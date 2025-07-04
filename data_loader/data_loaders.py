from torchvision import datasets, transforms
from base import BaseDataLoader

import os
import numpy as np
import torch
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.ecg_pipeline import bandpass_filter, stockwell_transform, standardize_signal
from PIL import Image


# Datasets 및 DataLoader 설정

# class MnistDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class Mit_bihDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size=32, shuffle=True, validation_split=0.1, num_workers=2, fs=360):
        self.dataset = Mit_bihDataset(data_path=data_dir, fs=fs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class Mit_bihDataset(Dataset):
    def __init__(self, data_path, split='train', fs=360, transform=None):
        """
        Args:
            data_path (str): Path to the directory containing mit_bih .npy files.
            split (str): 'train' or 'test'
            resize (tuple): Desired image size for model input (H, W)
            fs (int): Sampling frequency of ECG signals
        """
        self.fs = fs
        self.data = np.load(os.path.join(data_path, "data.npy"))
        self.labels = np.load(os.path.join(data_path, "label.npy"))
        self.groups = np.load(os.path.join(data_path, "group.npy"))
        self.train_ind = np.load(os.path.join(data_path, "train_ind.npy"))
        self.test_ind = np.load(os.path.join(data_path, "test_ind.npy"))

        if split == "train":
            self.indices = np.where(self.train_ind)[0]
        else:
            self.indices = np.where(self.test_ind)[0]

        self.transform = transform
        
        self.class_names = ['N', 'S', 'V', 'F', 'Q']
        self.class_to_idx = {name: idx for idx,name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        signal = self.data[index]  # 1D ECG signal
        label = self.groups[index]  # You can map this to int class if needed

        # --- Preprocessing ---
        signal = bandpass_filter(signal, self.fs, lowcut=0.5, highcut=40)
        st_result, _, _ = stockwell_transform(signal, fs=self.fs, fmin=0.5, fmax=40)

        # Real, Imag, Magnitude 채널 생성 및 정규화
        real = standardize_signal(np.real(st_result))
        imag = standardize_signal(np.imag(st_result))
        magnitude = standardize_signal(np.abs(st_result))

        # Resize to 384x384
        ch1 = cv2.resize(real, (384, 384), interpolation=cv2.INTER_LINEAR)
        ch2 = cv2.resize(imag, (384, 384), interpolation=cv2.INTER_LINEAR)
        ch3 = cv2.resize(magnitude, (384, 384), interpolation=cv2.INTER_LINEAR)

        img3 = np.stack([ch1, ch2, ch3], axis=0)  # shape: (3, 384, 384)
        img_tensor = torch.from_numpy(img3).float()
         
        # one-hot encoding for label
        label_idx = self.class_to_idx[label]
        label_one_hot = np.zeros(self.num_classes, dtype=np.float32)
        label_one_hot[label_idx] = 1.0                
        
        label = torch.from_numpy(label_one_hot).float()
        
        return img_tensor, label 

    
