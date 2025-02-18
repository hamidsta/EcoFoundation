import torch
from torch.utils.data import Dataset
import numpy as np


class GeneExpressionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx] 
    


def split_data_by_cells(TMA_objects,data_type ='counts',val_ratio=0.2, seed=42):
    """
    Split TMA objects into training and validation sets while maintaining sample integrity
    and targeting a specific ratio of cells.
    
    Parameters:
    -----------
    TMA_objects : list
        List of AnnData objects
    val_ratio : float, optional
        Target ratio of cells for validation set (default: 0.2)
    seed : int, optional
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    train_dataset : Dataset
        Training dataset
    val_dataset : Dataset
        Validation dataset
    """
    torch.manual_seed(seed)
    
    sample_counts = []
    cells_per_sample = []
    
    for adata in TMA_objects:
        counts = adata.layers[data_type].A if hasattr(adata.layers[data_type], 'A') else adata.layers[data_type].toarray()
        sample_counts.append(counts)
        cells_per_sample.append(counts.shape[0])
        #print(f"Sample shape: {counts.shape}")
    
    total_cells = sum(cells_per_sample)
    target_val_cells = int(val_ratio * total_cells)
    
    indices = torch.randperm(len(sample_counts))
    
    current_val_cells = 0
    split_idx = 0
    for i in indices:
        current_val_cells += cells_per_sample[i]
        split_idx += 1
        if current_val_cells >= target_val_cells:
            break
    
    val_indices = indices[:split_idx]
    train_indices = indices[split_idx:]
    train_samples = [sample_counts[i] for i in train_indices]
    val_samples = [sample_counts[i] for i in val_indices]
    
    train_data = np.vstack(train_samples)
    val_data = np.vstack(val_samples)
    
    train_dataset = GeneExpressionDataset(torch.tensor(train_data, dtype=torch.float32))
    val_dataset = GeneExpressionDataset(torch.tensor(val_data, dtype=torch.float32))
    
    print("\nSplit results:")
    print(f"Training: {len(train_samples)} samples, {train_data.shape[0]} cells ({train_data.shape[0]/total_cells*100:.1f}%)")
    print(f"Validation: {len(val_samples)} samples, {val_data.shape[0]} cells ({val_data.shape[0]/total_cells*100:.1f}%)")
    
    return train_dataset, val_dataset