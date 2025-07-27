import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import scipy.signal as sg
import wfdb
from scipy.stats import skew, kurtosis, median_abs_deviation, skewtest, kurtosistest, fligner, shapiro, power_divergence, tmean
from scipy.signal import welch, periodogram

import pywt
import cv2
import torch
import os

from tqdm import tqdm

from scipy.signal import resample
from scipy.signal import butter, filtfilt, medfilt
import numpy as np

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=250, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def median_filter(signal, kernel_ms=200, fs=250):
    kernel_size = int(kernel_ms * fs / 1000)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(signal, kernel_size)

def resample_signal(signal, original_fs=360, target_fs=250):
    duration = len(signal) / original_fs
    target_length = int(duration * target_fs)
    return resample(signal, target_length)
    
def min_max_normalize(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    return (signal - min_val) / (max_val - min_val + 1e-8)

def z_score_normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + 1e-8)

def moving_average(signal, window):
    weights = np.repeat(1.0, window) / window            
    ma = np.convolve(signal, weights, 'valid')
    return ma

def add_gaussian_noise(signal, noise_level=0.01):
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def pmat(signal, max_window, direction):
    N = len(signal)
    M = np.ndarray(shape=(max_window, N))
    padded_signal = np.concatenate((np.ones(N) * signal[0], signal, np.ones(N) * signal[-1]))    
    for w in range(1, max_window + 1):
        if direction == 'Left':
            M[w - 1] = moving_average(padded_signal[N - w:2 * N - 1], window=w) 
        elif direction == 'Right':
            M[w - 1] = moving_average(padded_signal[N:2 * N - 1 + w], window=w)             
    return M

invalid_anns = ['|', '~', '!', '+', '[', ']', '"', 'x', 'f', 'Q', 's', 'T', 'n', 'B']
PhysioBank = {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # N
        "A": 1, "a": 1, "S": 1, "J": 1,  # S
        "V": 2, "E": 2,  # V
        "F": 3,  # F
        "/": 4, # Peaced beats        
}

def prepare_scaled_records(records, database, sampling_rate, path_str, preprocess):
    scaled_signals = []
    r_peak_list = []
    ann_list = []
    for record in records:
        if database=='stt':
            tol = 0.1
            chanel_number = record[1]
            record = record[0]            
            ecg = wfdb.rdrecord(f'{path_str}/{record}').p_signal[:, chanel_number]
        else:
            tol = 0.05
            ecg = wfdb.rdrecord(f'{path_str}/{record}').p_signal[:, 0 if record!='114' else 1] # record 114 mit-bih is inversed
            
        anns = wfdb.rdann(f'{path_str}/{record}', extension='atr')
        r_peaks, annotations = anns.sample, anns.symbol                                        
        
        # === Apply Preprocessing ===
        if isinstance(preprocess, dict):  # new flexible config format
            fs = sampling_rate

            if preprocess.get("resample_rate") and preprocess["resample_rate"] != fs:
                ecg = resample_signal(ecg, sampling_rate, preprocess["resample_rate"])
                fs = preprocess["resample_rate"]
                print(f'Resampled ECG from {sampling_rate}Hz to {fs}Hz')
            else:
                fs = sampling_rate

            if preprocess.get("denoise", {}).get("bpf"):
                lowcut, highcut = preprocess["denoise"]["bpf"]
                ecg = bandpass_filter(ecg, lowcut=lowcut, highcut=highcut, fs=fs)
                print(f'Applied bandpass filter: {lowcut}-{highcut}Hz')
                
            if preprocess.get("denoise", {}).get("median"):
                for k in preprocess["denoise"]["median"]:
                    ecg = median_filter(ecg, kernel_ms=k, fs=fs)
                print(f'Applied median filter with kernel size(s): {preprocess["denoise"]["median"]}ms')

            if preprocess.get("normalize") == "zscore":
                ecg = z_score_normalize(ecg)
                print('Applied Z-score normalization')
            elif preprocess.get("normalize"):
                ecg = min_max_normalize(ecg)
                print('Applied Min-Max normalization')

            if preprocess.get("augmentation", {}).get("noise"):
                ecg = add_gaussian_noise(ecg, noise_level=preprocess["augmentation"]["noise"])
                print(f'Added Gaussian noise with level: {preprocess["augmentation"]["noise"]}')
        else:
            raise ValueError("Unsupported preprocess format. Please use a dictionary with appropriate keys.")
        
        scaled_signals.append(ecg)
        
        # align r-peaks
        newR = []
        for r_peak in r_peaks:
            r_left = np.maximum(r_peak - int(tol * fs), 0)
            r_right = np.minimum(r_peak + int(tol * fs), len(ecg))  # len(ecg)로 수정
            segment = ecg[r_left:r_right]  # 현재 ECG 신호에서 슬라이스
            if len(segment) == 0:
                continue  # 혹은 예외 처리
            newR.append(r_left + np.argmax(segment))
        r_peaks = np.array(newR, dtype="int")
        r_peak_list.append(r_peaks)        
        ann_list.append(annotations) 
    return scaled_signals, r_peak_list, ann_list

def get_peaks_ecg(ecg, rpeak, rr_avg, rr_next, sampling_rate):    
    """
    Identifies P, Q, R, S and T peak positions around a given R-peak.
    """
    
    # Define a short window (26ms) to search for Q and S peaks near the R peak.
    # S-peak
    b1 = min(rpeak + int(sampling_rate*0.026), len(ecg)-1)
    sp = ecg[rpeak:b1].argmin()    
    speak = rpeak + sp if rpeak + sp < len(ecg) else rpeak
    # Q-peak
    b2 = max(0, rpeak-int(sampling_rate*0.026))
    qp = ecg[b2:rpeak].argmin() if b2<rpeak else rpeak
    qpeak = rpeak-int(sampling_rate*0.026)+qp
    qpeak = 0 if qpeak<0 else qpeak

    # P-peak: typically a small positive deflection before the QRS complex. 
    p_start = max(0, qpeak - int(rr_avg/4))    
    p_end = qpeak
    if p_start == p_end:
        ppeak = rpeak
    elif p_start>p_end:
        t = p_start
        p_start = p_end
        p_end = t
        if len(ecg[p_start:p_end])==0:
            print(p_start,p_end,rpeak,qpeak)
        ppeak = p_start + ecg[p_start:p_end].argmax()
    else:        
        ppeak = p_start + ecg[p_start:p_end].argmax()
        if len(ecg[p_start:p_end])==0:
            pass
            #print(p_start,p_end,rpeak)
    
    # T-peak
    t_start = speak + int(sampling_rate*0.166) # 166 ms    
    t_end = min(rpeak + int((rr_next)/2), len(ecg))
    tpeak = t_start + ecg[t_start:t_end].argmax() if t_start<t_end else t_start
    tpeak = tpeak if tpeak < len(ecg) else rpeak
    return ppeak, qpeak, rpeak, speak, tpeak

def getXY(scaled_signals, r_peak_list, ann_list, database, sampling_rate, train, before, after, xy_method): 
    
    if xy_method == 'pmat':
        x1, x2, y = pmat_xy(scaled_signals, r_peak_list, ann_list, database, sampling_rate, train, before, after)
        return x1, x2, y
    elif xy_method == 'simple':
        x, y = simple_cnn_xy(scaled_signals, r_peak_list, ann_list, database, sampling_rate, train, before, after)
        return x, y

    else:
        raise ValueError(f"[getXY] Unsupported xy_method: {xy_method}")
    
def pmat_xy(scaled_signals, r_peak_list, ann_list, database, sampling_rate, train, before, after):
    wavelet = "gaus4"  # mexh, morl, gaus8, gaus4
    scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 80, 1)    
    
    count = 0
    x1, y = [], []
    x2 = []
    
    for i in tqdm(range(len(scaled_signals)), desc="Processing ECG Records"):
        
        # needed for extract limited beats from ST-T database
        counter_beats = {0:0,1:0,2:0}
        scaled_ecg = scaled_signals[i]         
        r_peaks = r_peak_list[i]
        anns = ann_list[i]        
        
        NP = len(scaled_ecg)
                
        avg_rri = np.mean(np.diff(r_peaks))        
        
        all_peaks = [get_peaks_ecg(scaled_ecg, rpeak=r_peaks[k], 
                        rr_avg=r_peaks[k]-r_peaks[k-1] if k>0 and anns[k-1] not in invalid_anns else avg_rri, 
                        rr_next=r_peaks[k+1]-r_peaks[k] if k+1<len(r_peaks) and anns[k+1] not in invalid_anns else avg_rri, 
                        sampling_rate=sampling_rate) for k in range(len(r_peaks))]

        # Hand craft features
        valid_peaks = [all_peaks[k] for k in range(1,len(r_peaks)-1) if anns[k-1] not in invalid_anns and anns[k] not in invalid_anns and anns[k+1] not in invalid_anns]               
        avg_RT = np.mean([peaks[4]-peaks[2] for peaks in valid_peaks])
        avg_PR = np.mean([peaks[2]-peaks[0] for peaks in valid_peaks])
        avg_SQ = np.mean([peaks[3]-peaks[1] for peaks in valid_peaks])
        avg_TQ = np.mean([all_peaks[k][1]-all_peaks[k-1][-1] for k in range(1,len(all_peaks)) if anns[k] not in invalid_anns and anns[k-1] not in invalid_anns])
        avg_TP = np.mean([all_peaks[k][0]-all_peaks[k-1][-1] for k in range(1,len(all_peaks)) if anns[k] not in invalid_anns and anns[k-1] not in invalid_anns])
        avg_P = np.mean([peaks[0] for peaks in valid_peaks])
        
        # For dynamic permutating beteween heatbeats
        m_current = []

        # Per beats
        for k in range(len(r_peaks)):
            #skipp 1st and last rpeak
            if k==0 or k == len(r_peaks)-1:
                continue
            
            ppeak_prev, qpeak_prev, r_prev, speak_prev, tpeak_prev = all_peaks[k-1]
            ppeak, qpeak, _, speak, tpeak = all_peaks[k]            
            ppeak_next, qpeak_next, r_next, speak_next, tpeak_next = all_peaks[k+1]

            r, ann = r_peaks[k], anns[k]
            
            if ann=='J':
                continue
            
            if ann in invalid_anns or ann not in PhysioBank.keys():
                continue
            
            # continue if the previous beat is unknown
            if anns[k-1] in invalid_anns or anns[k+1] in invalid_anns:
                continue
            
            if r_peaks[k + 1] - r_peaks[k] == 0 or r_peaks[k] - r_peaks[k-1]==0:
                continue
            
            # continue if this r_peak is negative            
            if r_peaks[i]<0:
                continue
            
            if r<before or r+after >= NP:
                pass            
            
            if r_prev<before or NP - r_next<after:
                continue                                                     
            
            label = PhysioBank[ann] 
            label_prev = PhysioBank[anns[k-1]]
            if label == 3:
                continue              
                
            counter_beats[label]+=1
            if database=='stt' and label==0 and counter_beats[0]>500:                
                continue                                                   
            
            #Calculate wave duration
            PR, RT, SQ = r-ppeak, tpeak-r, speak-qpeak                                                                        
                                    
            # The heartbeat that will be classified, and its previous and next
            heartbeat = scaled_ecg[r-before:r+after]                        
            heartbeat_prev = scaled_ecg[r_prev-before:r_prev+after]            
            heartbeat_next = scaled_ecg[r_next-before:r_next+after]
                        

            # Skip if all = 0
            if heartbeat.any()==0:
                print(i)
                continue 
                
            # statistics
            sktest = skewtest(heartbeat)            
            #shapirotest = shapiro(heartbeat)                       
                        
            # Scale the heartbeat            
            heartbeat = (heartbeat-heartbeat.min())/(heartbeat.max()-heartbeat.min())                                    
            heartbeat_prev = (heartbeat_prev-heartbeat_prev.min())/(heartbeat_prev.max()-heartbeat_prev.min())                         
            heartbeat_next = (heartbeat_next-heartbeat_next.min())/(heartbeat_next.max()-heartbeat_next.min())                                                                                                                                               
                                        
            if len(m_current)==0:
                m_prev = pmat(heartbeat_prev, max_window=100, direction='Left') 
                m_prev = cv2.resize(m_prev, (120, 120))
                
                m_current = pmat(heartbeat, max_window=100, direction='Left') 
                m_current = cv2.resize(m_current, (120, 120)) 
            else:
                m_prev=m_current
                m_current=m_next
                        
            m_next = pmat(heartbeat_next, max_window=100, direction='Left') 
            m_next = cv2.resize(m_next, (120, 120))                                    
            
            m = torch.tensor(np.array([m_prev, m_current, m_next])).reshape([3,120, 120]).float()
            
            # OR Take only the current heartbeat
            '''
            m_current = pmat(heartbeat, max_window=100, direction='Left')
        
            m_current = cv2.resize(m_current, (120, 120))
            m = torch.tensor(np.array([m_current])).reshape([1,120, 120]).float()
            torch.save(m, path_current)  
            '''
            
            a = np.maximum(k - 18, 0)
            b = np.maximum(a + 18, k + 1)
            avg_rri_local = np.mean(np.diff(r_peaks[a:b]))
            coef = sampling_rate/360
            input_2 = np.array([
                (r_peaks[k] - r_peaks[k - 1]) / avg_rri,  # previous RR Interval
                (r_peaks[k + 1] - r_peaks[k]) / avg_rri,  # post RR Interval
                (r_peaks[k] - r_peaks[k - 1]) / (r_peaks[k + 1] - r_peaks[k]),  # ratio RR Interval                
                avg_rri_local / avg_rri,  # local RR Interval
            ], dtype=np.float32)          
            
            x1.append(m)
            x2.append(input_2)
            y.append(label)
            count +=1

    return x1, x2, y

def ecg_to_cwt_image(signal, fs=250, wavelet='morl', scale_range=(1, 128), output_size=(128, 128)):
    scales = np.arange(scale_range[0], scale_range[1] + 1)
    coef, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1.0/fs)
    
    # 정규화 및 이미지 변환
    cwt_image = np.abs(coef)
    cwt_image = (cwt_image - cwt_image.min()) / (cwt_image.max() - cwt_image.min() + 1e-8)

    # Resize to fixed output size (optional)
    cwt_image = cv2.resize(cwt_image, output_size)

    return cwt_image  # shape: [H, W]


def simple_cnn_xy(signals, r_peaks_list, ann_list, database, fs, train, before, after):
    X = []
    Y = []

    for ecg, r_peaks, annotations in zip(signals, r_peaks_list, ann_list):
        for i in range(1, len(r_peaks) - 1):
            r = r_peaks[i]
            ann = annotations[i]

            if ann not in PhysioBank or ann in invalid_anns:
                continue

            label = PhysioBank[ann]
            if label == 3:
                continue

            start = r - before
            end = r + after
            if start < 0 or end > len(ecg):
                continue

            segment = ecg[start:end]
            segment = (segment - np.min(segment)) / (np.max(segment) - np.min(segment) + 1e-8)

            # 💡 Wavelet transform to 2D image
            cwt_img = ecg_to_cwt_image(segment, fs=fs, wavelet='morl', output_size=(128, 128))

            X.append(cwt_img)
            Y.append(label)

    X = np.array(X)  # shape: [N, H, W]
    Y = np.array(Y)
    return X, Y
