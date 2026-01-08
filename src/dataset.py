import os
import random
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Any
from numpy.linalg import norm

# Preprocessing Dataset (WAV -> MERT)
class AudioDataset(Dataset):
    def __init__(self, tracks_dir: str, audio_processor=None):
        self.tracks_dir = tracks_dir
        self.tracklist = sorted([
            t for t in os.listdir(tracks_dir) 
            if os.path.isdir(os.path.join(tracks_dir, t)) and not t.startswith('.')
        ])
        
        self.audio_processor = audio_processor
        self.sample_rate = None

    def __len__(self) -> int:
        return len(self.tracklist)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        audios = {}
        track_name = self.tracklist[idx]
        track_path = os.path.join(self.tracks_dir, track_name)
        
        versions = sorted([
            v for v in os.listdir(track_path) 
            if os.path.isdir(os.path.join(track_path, v)) and not v.startswith('.')
        ])

        for version in versions:
            version_path = os.path.join(track_path, version)
            
            files = sorted([
                f for f in os.listdir(version_path) 
                if f.endswith('.wav')
            ])
            
            if not files:
                continue

            audios[version] = []
            for file in files:
                file_path = os.path.join(version_path, file)
                waveform, sr = torchaudio.load(file_path)
                self.sample_rate = sr
                audios[version].append(waveform)

            if self.audio_processor:
                target_sr = self.audio_processor.sampling_rate
                processed_audios = []
                for waveform in audios[version]:
                    
                    if self.sample_rate is not None and self.sample_rate != target_sr:
                        waveform = F.resample(waveform, int(self.sample_rate), target_sr)
                    
                    inputs = self.audio_processor(
                        waveform.squeeze().numpy(), 
                        sampling_rate=target_sr, 
                        return_tensors="pt"
                    )["input_values"].squeeze()
                    
                    processed_audios.append(inputs)
                
                audios[version] = processed_audios
        
        if audios:
            min_frames = min([len(audios[v]) for v in audios.keys()])
            for v in audios.keys():
                audios[v] = audios[v][:min_frames]
        
        return {
            'track': track_name,
            'audios': audios
        }

def audio_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_dict = {}
    
    max_len = 0
    for item in batch:
        for version in item["audios"]:
            for segment in item["audios"][version]:
                if segment.shape[-1] > max_len:
                    max_len = segment.shape[-1]

    for item in batch:
        track_name = item["track"]
        batch_dict[track_name] = []
        
        for version in item["audios"]:
            padded_segments = []
            for segment in item["audios"][version]:
                pad_amount = max_len - segment.shape[-1]
                padded_seg = torch.nn.functional.pad(segment, (0, pad_amount))
                padded_segments.append(padded_seg)
            
            batch_dict[track_name].append(torch.stack(padded_segments))

    return batch_dict

def create_audio_dataloader(
    tracks_dir: str, 
    batch_size: int, 
    num_workers: int, 
    audio_processor=None,
) -> DataLoader:
    dataset = AudioDataset(tracks_dir, audio_processor=audio_processor)
    print(f"Audio Dataset created with {len(dataset)} tracks.")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False,
        collate_fn=audio_collate_fn
    )
    return dataloader


class TripletDataset(Dataset):
    def __init__(self, tracks_dir: str, store_dict: bool = True, length_mult: int = 1):
        super().__init__()
        self.source = sorted([s for s in os.listdir(tracks_dir) if s.endswith('.npy') and not s.startswith('.')])
        try:
            self.source.sort(key=lambda x: int(x.replace('pair_', '').replace('.npy', '')))
        except ValueError:
            self.source.sort()
            
        self.tracks_dir = tracks_dir
        self.store_dict = store_dict
        if self.store_dict:
            self.track_dict = self._create_track_dict()
        print(f"[Dataset] Loaded {len(self.source)} embedding tracks.")
        self.length_mult = length_mult

    def _create_track_dict(self):
        track_dict = {}
        print("Loading embeddings into RAM...")
        for track in self.source:
            if track not in track_dict:
                track_dict[track] = np.load(os.path.join(self.tracks_dir, track))
        return track_dict

    def __len__(self):
        return len(self.source) * self.length_mult

    def _find_best_match(self, anchor_vec, suspicious_matrix):
        # Calculate Cosine Similarity for best pairing
        scores = np.dot(suspicious_matrix, anchor_vec)
        norm_anchor = norm(anchor_vec) + 1e-8
        norm_susp = norm(suspicious_matrix, axis=1) + 1e-8
        cosine_sims = scores / (norm_susp * norm_anchor)
        return np.argmax(cosine_sims)

    def __getitem__(self, idx):
        idx = int(idx // self.length_mult)
        idx = int(idx % len(self.source))
        anchor_name = self.source[idx]
        
        if self.store_dict:
            data = self.track_dict[anchor_name]
        else:
            data = np.load(os.path.join(self.tracks_dir, anchor_name))

        # Anchor Selection
        a_ver = 0 # Original
        anchor_seg_idx = random.randint(0, data.shape[1] - 1)
        anchor_sample = data[a_ver, anchor_seg_idx]

        # Positive Mining
        p_ver = 1 # Cover
        if data.shape[0] > 1:
            cover_all_segments = data[p_ver]
            
            # Compute vectors for matching
            anchor_flat = np.mean(anchor_sample, axis=(0, 1))
            cover_flat_matrix = np.mean(cover_all_segments, axis=(1, 2))
            
            # Find the segment closest to Anchor
            best_match_idx = self._find_best_match(anchor_flat, cover_flat_matrix)
            positive_sample = data[p_ver, best_match_idx]
        else:
            # Fallback
            positive_sample = anchor_sample

        # Negative Selection
        if random.random() < 0.5:
            # Hard Negative: Other segment from Anchor version
            neg_ver = a_ver
            neg_seg = random.randint(0, data.shape[1] - 1)
            while neg_seg == anchor_seg_idx and data.shape[1] > 1:
                neg_seg = random.randint(0, data.shape[1] - 1)
            negative_sample = data[neg_ver, neg_seg]
        else:
            # Easy Negative: Random segment from different track
            neg_idx = random.randint(0, len(self.source) - 1)
            while neg_idx == idx:
                neg_idx = random.randint(0, len(self.source) - 1)
            neg_name = self.source[neg_idx]
            
            if self.store_dict:
                neg_data = self.track_dict[neg_name]
            else:
                neg_data = np.load(os.path.join(self.tracks_dir, neg_name))
            
            neg_ver = random.randint(0, neg_data.shape[0] - 1)
            neg_seg = random.randint(0, neg_data.shape[1] - 1)
            negative_sample = neg_data[neg_ver, neg_seg]

        def prepare(x):
            t = torch.from_numpy(x).float()
            if t.ndim == 3: t = t.permute(0, 2, 1)
            if t.ndim == 2: t = t.unsqueeze(-1).permute(0, 2, 1)
            return torch.cat([h for h in t], dim=0) 

        return prepare(anchor_sample), prepare(positive_sample), prepare(negative_sample)

def triplet_collate_fn(batch):
    anchors, positives, negatives = zip(*batch)
    def pad(t_list):
        max_l = max([t.shape[-1] for t in t_list])
        return torch.stack([torch.nn.functional.pad(t, (0, max_l - t.shape[-1])) for t in t_list])
    return pad(anchors), pad(positives), pad(negatives)