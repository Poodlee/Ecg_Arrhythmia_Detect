from torchvision import datasets, transforms
from base import BaseDataLoader

import os
import numpy as np
import torch
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ecg_pipeline import bandpass_filter, stockwell_transform, standardize_signal
from PIL import Image
from ecg_pipeline import prepare_scaled_records, getXY

class DataLoaderFactory:
    _dataloader_map = {
        'mnist': 'MnistDataLoader',
        'mit_bih': 'Mit_bihDataLoader',
        'incart': 'IncartDataLoader',
        'european_stt': 'EuropeansttDataLoader'
    }

    @staticmethod
    def get_dataloader(data_type, *args, **kwargs):
        if data_type.lower() not in DataLoaderFactory._dataloader_map:
            raise ValueError(f"지원하지 않는 데이터 타입입니다: {data_type}")
        dataloader_class_name = DataLoaderFactory._dataloader_map[data_type.lower()]
        dataloader_class = globals().get(dataloader_class_name)
        if dataloader_class is None:
            raise ImportError(f"{dataloader_class_name} 클래스를 찾을 수 없습니다.")
        return dataloader_class(*args, **kwargs)
    

class Mit_bihDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size=32, shuffle=True, validation_split=0.1, num_workers=8, fs=360):
        self.dataset = Mit_bihDataset(data_path=data_dir, split='train' if validation_split > 0 else 'test', fs=fs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class Mit_bihDataset(Dataset):
    def __init__(self, data_path, split='train', fs=360, transform=None):
        self.data_path = data_path
        self.split = split
        self.fs = fs
        self.transform = transform
        
        # MIT-BIH 데이터셋의 레코드 목록
        records_mit_train = ['101', '106', '109', '112', '114', '115', '116', '118', '119', '122', 
                             '124', '201', '203', '205', '207', '208', '215', '220', '223', '230', 
                             '232', '222']
        records_mit_test = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', 
                            '209', '210', '212', '213', '214', '219', '221', '228', '231', '233', '234']
        
        # split에 따라 레코드 선택
        self.records = records_mit_train if split == 'train' else records_mit_test
        
        # 처리된 데이터 저장 경로
        self.processed_dir = os.path.join(data_path, 'processed', split)
                
        # 데이터가 이미 존재하는지 확인
        self.data_file = os.path.join(self.processed_dir, f"{split}_infos.pt")
        
        if os.path.exists(self.data_file):
            
            split_infos = torch.load(self.data_file, weights_only=False)
            self.x1 = split_infos['x1']
            
            x2_np = np.array(split_infos['x2']).astype(np.float64)
            
            # StandardScaler 적용
            mean = x2_np.mean(axis=0)
            std = x2_np.std(axis=0)
            std[std == 0] = 1.0  
            x2_scaled = (x2_np - mean) / std
            self.x2 = torch.tensor(x2_scaled, dtype=torch.float32)
            self.y = torch.LongTensor(split_infos['y'])
            
        else:
            os.makedirs(self.processed_dir, exist_ok=True)

            # 데이터 전처리
            scaled_signals, r_peak_list, ann_list = prepare_scaled_records(
                self.records, database='mit_bih', sampling_rate=self.fs, path_str=self.data_path
            )
            before, after = 90, 100 # MIT

            # 특징 추출 및 데이터 준비
            self.x1, self.x2, self.y = getXY(
                scaled_signals, r_peak_list, ann_list, database='mit_bih', sampling_rate=self.fs, train=(split == 'train'), before=before, after=after
            )
            self.x1 = torch.tensor(self.x1, dtype=torch.float32) if not isinstance(self.x1, torch.Tensor) else self.x1.to(torch.float32)
            self.x2 = torch.tensor(self.x2, dtype=torch.float32) if not isinstance(self.x2, torch.Tensor) else self.x2.to(torch.float32)
            self.y = torch.tensor(self.y, dtype=torch.long) if not isinstance(self.y, torch.Tensor) else y.to(torch.long)

            # 데이터 저장
            torch.save({
                'x1': self.x1,
                'x2': self.x2,
                'y': self.y
            }, self.data_file)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        path_x1 = self.x1[idx]
        x1 = torch.load(path_x1) 
        
        x2 = self.x2[idx].detach().clone()
        y = self.y[idx].detach().clone()
        
        if self.transform:
            x1 = self.transform(x1)
        
        return {'x1': x1,'x2': x2}, y

class IncartDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size=32, shuffle=True, validation_split=0.1, num_workers=8, fs=360):
        self.dataset = IncartDataset(data_path=data_dir, split='train' if validation_split > 0 else 'test', fs=fs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class IncartDataset(Dataset):
    def __init__(self, data_path, split='train', fs=274, transform=None):
        self.data_path = data_path
        self.split = split
        self.fs = fs
        self.transform = transform
        
        # MIT-BIH 데이터셋의 레코드 목록
        records_incart_train = ['I03', 'I04',  'I05', 'I08', 'I09', 'I10', 'I11', 'I12', 'I13', 'I14', 'I16', 'I17', 'I18', 'I19', 'I25', 'I26', 'I27', 'I28', 'I33', 'I34', 'I35', 'I36', 'I37', 'I44', 'I45', 'I46', 'I47', 'I48', 'I54', 'I55', 'I56', 'I68', 'I69', 'I70', 'I71', 'I72', 'I73', 'I74', 'I75', 'I42', 'I43', 'I62', 'I63', 'I64', 'I06', 'I07']
        records_incart_test =  ['I01', 'I02', 'I15', 'I20', 'I21', 'I22', 'I23', 'I24', 'I29', 'I30', 'I31', 'I32', 'I38', 'I39', 'I40', 'I41', 'I49', 'I50', 'I51', 'I52', 'I53', 'I57', 'I58', 'I59', 'I60', 'I61', 'I65', 'I66', 'I67']
        
        # split에 따라 레코드 선택
        self.records = records_incart_train if split == 'train' else records_incart_test
        
        # 처리된 데이터 저장 경로
        self.processed_dir = os.path.join(data_path, 'processed', split)
                
        # 데이터가 이미 존재하는지 확인
        self.data_file = os.path.join(self.processed_dir, f"{split}_infos.pt")
        
        if os.path.exists(self.data_file):
            
            split_infos = torch.load(self.data_file, weights_only=False)
            self.x1 = split_infos['x1']
            
            x2_np = np.array(split_infos['x2']).astype(np.float64)
            
            # StandardScaler 적용
            mean = x2_np.mean(axis=0)
            std = x2_np.std(axis=0)
            std[std == 0] = 1.0  
            x2_scaled = (x2_np - mean) / std
            self.x2 = torch.tensor(x2_scaled, dtype=torch.float32)
            self.y = torch.LongTensor(split_infos['y'])
            
        else:
            os.makedirs(self.processed_dir, exist_ok=True)

            # 데이터 전처리
            scaled_signals, r_peak_list, ann_list = prepare_scaled_records(
                self.records, database='incart', sampling_rate=fs, path_str=self.data_path
            )
            
            before, after = 70, 75
            
            # 특징 추출 및 데이터 준비
            self.x1, self.x2, self.y = getXY(
                scaled_signals, r_peak_list, ann_list, database='incart', sampling_rate=self.fs, train=(split == 'train'), before=before, after=after
            )
            
            self.x1 = torch.tensor(self.x1, dtype=torch.float32) if not isinstance(self.x1, torch.Tensor) else self.x1.to(torch.float32)
            self.x2 = torch.tensor(self.x2, dtype=torch.float32) if not isinstance(self.x2, torch.Tensor) else self.x2.to(torch.float32)
            self.y = torch.tensor(self.y, dtype=torch.long) if not isinstance(self.y, torch.Tensor) else y.to(torch.long)

            
            # 데이터 저장
            torch.save({
                'x1': self.x1,
                'x2': self.x2,
                'y': self.y
            }, self.data_file)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        path_x1 = self.x1[idx]
        x1 = torch.load(path_x1) 
        
        x2 = self.x2[idx].detach().clone()
        y = self.y[idx].detach().clone()
        
        if self.transform:
            x1 = self.transform(x1)
        
        return {'x1': x1,'x2': x2}, y

class EuropeansttDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size=32, shuffle=True, validation_split=0.1, num_workers=8, fs=360):
        self.dataset = EuropeansttDataset(data_path=data_dir, split='train' if validation_split > 0 else 'test', fs=fs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

## TEST 용 ##
class EuropeansttDataset(Dataset):
    def __init__(self, data_path, split='train', fs=250, transform=None):
        self.data_path = data_path
        self.split = split
        self.fs = fs
        self.transform = transform
        
        # MIT-BIH 데이터셋의 레코드 목록
        records_incart_train = ['I03', 'I04',  'I05', 'I08', 'I09', 'I10', 'I11', 'I12', 'I13', 'I14', 'I16', 'I17', 'I18', 'I19', 'I25', 'I26', 'I27', 'I28', 'I33', 'I34', 'I35', 'I36', 'I37', 'I44', 'I45', 'I46', 'I47', 'I48', 'I54', 'I55', 'I56', 'I68', 'I69', 'I70', 'I71', 'I72', 'I73', 'I74', 'I75', 'I42', 'I43', 'I62', 'I63', 'I64', 'I06', 'I07']
        records_incart_test =  ['I01', 'I02', 'I15', 'I20', 'I21', 'I22', 'I23', 'I24', 'I29', 'I30', 'I31', 'I32', 'I38', 'I39', 'I40', 'I41', 'I49', 'I50', 'I51', 'I52', 'I53', 'I57', 'I58', 'I59', 'I60', 'I61', 'I65', 'I66', 'I67']
        
        # split에 따라 레코드 선택
        self.records = records_incart_train if split == 'train' else records_incart_test
        
        # 처리된 데이터 저장 경로
        self.processed_dir = os.path.join(data_path, 'processed', split)
                
        # 데이터가 이미 존재하는지 확인
        self.data_file = os.path.join(self.processed_dir, f"{split}_infos.pt")
        
        if os.path.exists(self.data_file):
            
            split_infos = torch.load(self.data_file, weights_only=False)
            self.x1 = split_infos['x1']
            
            x2_np = np.array(split_infos['x2']).astype(np.float64)
            
            # StandardScaler 적용
            mean = x2_np.mean(axis=0)
            std = x2_np.std(axis=0)
            std[std == 0] = 1.0  
            x2_scaled = (x2_np - mean) / std
            self.x2 = torch.tensor(x2_scaled, dtype=torch.float32)
            
            self.y = torch.LongTensor(split_infos['y'])
            
        else:
            os.makedirs(self.processed_dir, exist_ok=True)

            path_str_stt = 'data/europeanstt'
            mlii_records_stt = []
            with open(f'{path_str_stt}/RECORDS', 'r') as file:    
                for record_name in file.readlines():
                    record_name=record_name.replace('\n','')
                    with open(f'{path_str_stt}/{record_name}.hea') as file_header:
                        for line_number, line in enumerate(file_header, start=1):
                            if 'MLIII' in line:
                                # ('e0103', 1) are the record name and the channel of the mlii lead
                                mlii_records_stt.append((record_name, line_number-2))
                                break
                        file_header.close()
                file.close()            
            # 데이터 전처리
            scaled_signals1_stt, r_peak_list1_stt, ann_list1_stt = prepare_scaled_records(mlii_records_stt,'stt', fs, path_str_stt)            
            before, after = 62, 70
            
            # 특징 추출 및 데이터 준비
            self.x1, self.x2, self.y = getXY(scaled_signals1_stt, r_peak_list1_stt, ann_list1_stt, 'stt',250, True, before=before, after=after)
            
            self.x1 = torch.tensor(self.x1, dtype=torch.float32) if not isinstance(self.x1, torch.Tensor) else self.x1.to(torch.float32)
            self.x2 = torch.tensor(self.x2, dtype=torch.float32) if not isinstance(self.x2, torch.Tensor) else self.x2.to(torch.float32)
            self.y = torch.tensor(self.y, dtype=torch.long) if not isinstance(self.y, torch.Tensor) else self.y.to(torch.long)
            
            # 데이터 저장
            torch.save({
                'x1': self.x1,
                'x2': self.x2,
                'y': self.y
            }, self.data_file)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        path_x1 = self.x1[idx]
        x1 = torch.load(path_x1) 
        
        x2 = self.x2[idx].detach().clone()
        y = self.y[idx].detach().clone()
        
        if self.transform:
            x1 = self.transform(x1)
        
        return {'x1': x1,'x2': x2}, y
