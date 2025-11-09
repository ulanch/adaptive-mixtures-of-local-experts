"""
Data loader for Peterson & Barney (1952) vowel formant dataset.
Loads, filters, and preprocesses the 4-class vowel discrimination task
used in Jacobs et al. (1991) Table 1 experiments.
"""

import numpy as np
import csv


def load_pb52_vowels(filepath, vowels=['i', 'I', 'A', 'V'], train_speakers=50):
    """
    Load and preprocess PB-52 vowel formant data for speaker-independent classification.
    
    Data consists of F1 and F2 formant frequencies for vowels [i], [ɪ], [a], [ʌ]
    from 75 speakers (males, females, children) in an hVd context.
    Split: first N speakers for training, rest for testing.
    Preprocessing: scale to kHz, then z-score normalize based on training stats.
    """
    # Load raw CSV data and filter to requested vowel classes
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vowel = row['vowel'].strip().strip('"')
            if vowel in vowels:
                speaker = int(row['speaker'])
                f1 = float(row['f1'])
                f2 = float(row['f2'])
                data.append({
                    'speaker': speaker,
                    'vowel': vowel,
                    'f1': f1,
                    'f2': f2
                })
    
    vowel_labels = list(vowels)
    vowel_to_idx = {v: i for i, v in enumerate(vowel_labels)}
    
    # Speaker-independent split: first N speakers train, rest test
    speakers = sorted({int(d['speaker']) for d in data})
    train_ids = set(speakers[:train_speakers])
    
    train_data = []
    test_data = []
    
    for sample in data:
        if sample['speaker'] in train_ids:
            train_data.append(sample)
        else:
            test_data.append(sample)
    
    # Extract features (F1, F2) and labels
    X_train = np.array([[d['f1'], d['f2']] for d in train_data])
    y_train_idx = np.array([vowel_to_idx[d['vowel']] for d in train_data])
    
    X_test = np.array([[d['f1'], d['f2']] for d in test_data])
    y_test_idx = np.array([vowel_to_idx[d['vowel']] for d in test_data])
    
    # Scale formants from Hz to kHz
    X_train = X_train / 1000.0
    X_test = X_test / 1000.0

    # Z-score normalize using training set statistics
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    # Convert to one-hot encoding for multi-class targets
    num_classes = len(vowel_labels)
    y_train = np.eye(num_classes)[y_train_idx]
    y_test = np.eye(num_classes)[y_test_idx]
    
    return X_train, y_train, X_test, y_test, vowel_labels


def get_speaker_split(filepath, train_speakers=50):
    """Get the list of speaker IDs for train/test split"""
    speakers = set()
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            speakers.add(int(row['speaker']))
    
    all_speakers = sorted(speakers)
    train_speaker_ids = all_speakers[:train_speakers]
    test_speaker_ids = all_speakers[train_speakers:]
    
    return train_speaker_ids, test_speaker_ids
