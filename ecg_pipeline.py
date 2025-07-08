import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ECG signal pre-processing (denoising, standardization, feature extraction, etc.)

def bandpass_filter(signal_array, fs, lowcut=0.5, highcut=40, order=4):
    """
    Apply Butterworth bandpass filter to ECG signals.
    
    Args:
        signal_array: numpy array of shape (N,) or (N, T)
        fs: sampling frequency
        lowcut: low cutoff frequency in Hz
        highcut: high cutoff frequency in Hz
        order: filter order
        
    Returns:
        Filtered signal (same shape as input)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    
    if signal_array.ndim == 1:
        return signal.filtfilt(b, a, signal_array)
    else:
        return np.array([signal.filtfilt(b, a, sig) for sig in signal_array])


def detrend_signal(signal_array):
    """
    Remove linear trend from signal.
    
    Returns:
        Detrended signal
    """
    if signal_array.ndim == 1:
        return signal.detrend(signal_array)
    else:
        return signal.detrend(signal_array, axis=-1)


def standardize_signal(signal_array):
    """
    Apply Min-Max normalization to ECG signals.
    
    Returns:
        Standardized signal
    """

    def min_max(sig):
        min_val = np.min(sig)
        max_val = np.max(sig)

        if max_val - min_val == 0:
            return np.zeros_like(sig)
        return (sig - min_val) / (max_val - min_val)
    
    if signal_array.ndim == 1:
        return min_max(signal_array)
    else:
        return np.array([min_max(sig) for sig in signal_array])
    
def stockwell_transform(signal, fs, fmin=0, fmax=None):
    from stockwell import st
    """
    Apply Stockwell Transform (S-transform) to a 1D signal.

    Args:
        signal (np.ndarray): 1D array of the signal
        fs (float): Sampling frequency in Hz
        fmin (float): Minimum frequency (Hz) for transform
        fmax (float or None): Maximum frequency (Hz) for transform.
                              If None, defaults to fs/2.

    Returns:
        st_result (np.ndarray): 2D array of complex S-transform (freq x time)
        freqs (np.ndarray): Frequency axis values
        times (np.ndarray): Time axis values
    """
    N = len(signal)
    duration = N / fs
    t = np.linspace(0, duration, N)

    df = 1.0 / duration
    if fmax is None:
        fmax = fs / 2

    fmin_samples = int(fmin / df)
    fmax_samples = int(fmax / df)

    # Apply Stockwell Transform
    st_result = st.st(signal, fmin_samples, fmax_samples)

    freqs = np.linspace(fmin, fmax, fmax_samples - fmin_samples)
    return st_result, freqs, t
    
def moving_average(signal, window):
    weights = np.repeat(1.0, window) / window            
    ma = np.convolve(signal, weights, 'valid')
    return ma

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
import scipy.signal as sg
import wfdb

def prepare_scaled_records(records, database, sampling_rate, path_str):
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
        
        baseline = sg.medfilt(sg.medfilt(ecg, int(0.2 * sampling_rate) - 1), int(0.6 * sampling_rate) - 1)
        
        filtered_signal = ecg - baseline
        
        
        scaled_signal = filtered_signal        
        scaled_signals.append(scaled_signal)
        
        # align r-peaks
        newR = []
        for r_peak in r_peaks:
            r_left = np.maximum(r_peak - int(tol * sampling_rate), 0)
            r_right = np.minimum(r_peak + int(tol * sampling_rate), len(scaled_signal))
            newR.append(r_left + np.argmax(scaled_signal[r_left:r_right]))
        r_peaks = np.array(newR, dtype="int")
        
        r_peak_list.append(r_peaks)        
        ann_list.append(annotations) 
    return scaled_signals, r_peak_list, ann_list

def get_peaks_ecg(ecg, rpeak, rr_avg, rr_next, sampling_rate):    
    b1 = min(rpeak + int(sampling_rate*0.026), len(ecg)-1)
    b2 = max(0, rpeak-int(sampling_rate*0.026))

    sp = ecg[rpeak:b1].argmin()    
    speak = rpeak + sp
    speak = rpeak if speak>=len(ecg) else speak
    
    qp = ecg[b2:rpeak].argmin() if b2<rpeak else rpeak
    qpeak = rpeak-int(sampling_rate*0.026)+qp
    qpeak = 0 if qpeak<0 else qpeak

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
        
    t_start = speak + int(sampling_rate*0.166) # 0.166 ms    
    t_end = min(rpeak + int((rr_next)/2), len(ecg))
    tpeak = t_start + ecg[t_start:t_end].argmax() if t_start<t_end else t_start
    tpeak = tpeak if tpeak < len(ecg) else rpeak
    return ppeak, qpeak, rpeak, speak, tpeak
from scipy.stats import skew, kurtosis, median_abs_deviation, skewtest, kurtosistest, fligner, shapiro, power_divergence, tmean
from scipy.signal import welch, periodogram
import pywt
import cv2
import torch
import os
def getXY(scaled_signals, r_peak_list, ann_list, database, sampling_rate, train, before, after): 
    
    wavelet = "gaus4"  # mexh, morl, gaus8, gaus4
    scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 80, 1)    
    
    count = 0
    x1, y = [], []
    x2 = []
    
    for i in range(len(scaled_signals)):
        
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
            
            if count % 20000 ==0:
                print(f'{count} done')              
            
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
            
            subset = 'train' if train else 'test'
            os.makedirs(f'.\\data\\{database}\\pmat\\{subset}', exist_ok=True)                        
            path_current = f'.\\data\\{database}\\pmat\\{subset}\\x1_{i}_{r}.pt' 
            
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
            torch.save(m, path_current)
            
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
            ])            
            
            x1.append(path_current)
            x2.append(input_2)
            y.append(label)
            count +=1
    print(f'{count} done')
    return x1, x2, y