# ECG 데이터 형태 맞추기
import argparse
import os
import random
import wfdb
import numpy as np
from collections import Counter

# Define the label group mapping based on AAMI standard classification
label_group_map = {
    'N': 'N',  # Normal beat
    'L': 'N',  # Left bundle branch block beat
    'R': 'N',  # Right bundle branch block beat
    'A': 'S',  # Atrial premature beat
    'a': 'S',  # Aberrated atrial premature beat
    'J': 'S',  # Nodal (junctional) premature beat
    'j': 'S',  # Nodal (junctional) escape beat
    'S': 'S',  # Supraventricular premature beat
    'e': 'S',  # Atrial escape beat
    'V': 'V',  # Premature ventricular contraction
    'E': 'V',  # Ventricular escape beat
    'F': 'F',  # Fusion of ventricular and normal beat
    'f': 'Q',  # Fusion of paced and normal beat
    '/': 'Q',  # Paced beat
    'Q': 'Q',  # Unknown beat
}

def resample(signal, fs_in, fs_out):
    from scipy.signal import resample_poly
    if fs_in == fs_out:
        return signal
    else:
        return resample_poly(signal, fs_out, fs_in, padtype='mean')

def mit_bih_arrhythmia_process(input_path, output_path, fs_out=360):
    """
    Process the MIT-BIH Arrhythmia dataset to extract ECG beat segments and unify their format.

    This function reads all ECG records listed in the 'RECORDS' file inside `input_path`.
    For each record, it loads the ECG signal (specifically the 'MLII' lead) and corresponding annotations.
    It then extracts fixed-length beat segments centered on each annotated beat,
    resamples them to a common sampling frequency (`fs_out`), and labels them according to
    the provided label group mapping.

    The function splits the dataset into training and test sets based on the `test_ratio`,
    and saves the processed data, labels, group labels, patient IDs, and train/test indices
    as numpy files in `output_path`.

    Args:
        input_path (str): Path to the input ECG data file.
        output_path (str): Path to save the processed ECG data file.
        fs_out (int): Output sampling frequency. Default is 360 Hz.
    """
    test_ratio = 0.2 
    
    with open(os.path.join(input_path, 'RECORDS')) as f:
        records = f.read().strip().split('\n')
    test_set = set(random.sample(records, int(len(records) * test_ratio)))

    data, labels, groups, pids, train_flags = [], [], [], [], []

    for rec in records:
        try:
            ann = wfdb.rdann(os.path.join(input_path, rec), 'atr')
            sig, meta = wfdb.rdsamp(os.path.join(input_path, rec))
            fs = meta['fs']
            lead_names = meta['sig_name']
        except Exception as e:
            print(f"Failed to read {rec}: {e}")
            continue

        if 'MLII' not in lead_names:
            print(f"{rec}: MLII not found in {lead_names}")
            continue

        ch = lead_names.index('MLII')
        ecg = sig[:, ch]
        left, right = int(fs * 0.4), int(fs * 0.4) # left and right padding for beat extraction

        for idx, sym in zip(ann.sample, ann.symbol):
            if sym not in label_group_map:
                continue
            start, end = idx - left, idx + right
            if start < 0 or end >= len(ecg):
                continue
            beat = resample(ecg[start:end], fs, fs_out)
            data.append(beat)
            labels.append(sym)
            groups.append(label_group_map[sym])
            pids.append(rec)
            train_flags.append(rec not in test_set)

        print(f"{rec}: {len(data)} beats collected so far")

    data = np.array(data)
    labels = np.array(labels)
    groups = np.array(groups)
    pids = np.array(pids)
    train_flags = np.array(train_flags)
    test_flags = ~train_flags

    print("Total samples:", len(data))
    print("Label dist:", Counter(labels))
    print("Group dist:", Counter(groups))
    print("Train/Test:", sum(train_flags), "/", sum(test_flags))

    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, 'data.npy'), data)
    np.save(os.path.join(output_path, 'label.npy'), labels)
    np.save(os.path.join(output_path, 'group.npy'), groups)
    np.save(os.path.join(output_path, 'pid.npy'), pids)
    np.save(os.path.join(output_path, 'train_ind.npy'), train_flags)
    np.save(os.path.join(output_path, 'test_ind.npy'), test_flags)
                                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Pre-processing Script")
    parser.add_argument("--data_type", type=str, required=True, help="Type of ECG data to process (e.g., 'raw', 'processed')")
    parser.add_argument("--input_path", type=str, default="./data/", help="Path to the input ECG data file")
    parser.add_argument("--output_path", type=str, default="./data/processed", help="Path to save the processed ECG data file")
    
    args = parser.parse_args()
    
    input_path = os.path.join(args.input_path, args.data_type)
    output_path = os.path.join(args.output_path, args.data_type)
    
    if args.data_type == 'mit_bih':
        mit_bih_arrhythmia_process(input_path, output_path)