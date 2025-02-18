import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.dataloader
import torch.utils.data.dataset

class GeneExpressionDataset(Dataset):
    def __init__(self, data, batch=None, lib_size=None):
        """
        data: (N, n_genes) float tensor
        batch: (N,) long tensor with batch IDs
        lib_size: (N,) float tensor with library sizes
        """
        self.data = data
        self.batch = batch
        self.lib_size = lib_size

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Check that both batch and lib_size are provided
        if self.batch is None or self.lib_size is None:
            raise ValueError(
                f"Both 'batch' and 'lib_size' must be provided. "
                f"Received batch: {self.batch} and lib_size: {self.lib_size}."
            )

        # Retrieve the data for the given index
        x = self.data[idx]
        return x, self.batch[idx], self.lib_size[idx]

def split_data_by_cells(
    TMA_objects,
    data_type='counts',
    val_ratio=0.2,
    seed=42
):
    """
    Splits TMA objects into train/val sets while maintaining sample integrity
    (i.e. entire samples go to train or val),
    and returning a ratio of ~val_ratio cells in validation.

    Returns
    -------
    train_dataset, val_dataset : GeneExpressionDataset
    """

    torch.manual_seed(seed)

    sample_counts, cells_per_sample, sample_batch_ids, sample_lib_sizes = [], [], [], []

    for batch_id, adata in enumerate(TMA_objects):
        # Convert from sparse to dense if needed
        mat = adata.layers[data_type]
        mat = mat.toarray()

        # store the counts
        sample_counts.append(mat)
        cells_per_sample.append(mat.shape[0])

        # store batch_id for every cell in this sample
        sample_batch_ids.append([batch_id]*mat.shape[0])

        # Get library sizes
        if "nCount_RNA" in adata.obs:
            sample_lib_sizes.append(adata.obs["nCount_RNA"].values)


    total_cells = sum(cells_per_sample)
    target_val_cells = int(val_ratio * total_cells)

    # random permutation of sample indices
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
    val_samples   = [sample_counts[i] for i in val_indices]

    train_batch_ids = [sample_batch_ids[i] for i in train_indices]
    val_batch_ids   = [sample_batch_ids[i] for i in val_indices]

    train_lib_sizes = [sample_lib_sizes[i] for i in train_indices]
    val_lib_sizes   = [sample_lib_sizes[i] for i in val_indices]

    # stack them
    train_data = np.vstack(train_samples)
    val_data   = np.vstack(val_samples)

    # Flatten out the batch_ids
    train_batch_flat = np.concatenate(train_batch_ids)  # shape (sum_of_train_cells,)
    val_batch_flat   = np.concatenate(val_batch_ids)

    # If some AnnData didn't have library size, we can do a quick check
    # If ANY sample_lib_sizes is None, let's set entire lib_size to None to avoid mismatch
    # Otherwise, we can stack them
    if any([x is None for x in train_lib_sizes + val_lib_sizes]):
        train_libraries = None
        val_libraries   = None
    else:
        train_libraries = np.concatenate(train_lib_sizes)
        val_libraries   = np.concatenate(val_lib_sizes)

    # Convert to torch tensors
    train_X = torch.tensor(train_data, dtype=torch.float32)
    val_X   = torch.tensor(val_data, dtype=torch.float32)
    train_batch_t = torch.tensor(train_batch_flat, dtype=torch.long)
    val_batch_t   = torch.tensor(val_batch_flat, dtype=torch.long)

    if train_libraries is not None:
        train_lib_t = torch.tensor(train_libraries, dtype=torch.float32)
        val_lib_t   = torch.tensor(val_libraries, dtype=torch.float32)
    else:
        train_lib_t = None
        val_lib_t   = None

    # Create dataset objects
    train_dataset = GeneExpressionDataset(
        data=train_X,
        batch=train_batch_t,
        lib_size=train_lib_t
    )
    val_dataset = GeneExpressionDataset(
        data=val_X,
        batch=val_batch_t,
        lib_size=val_lib_t
    )

    print("\nSplit results:")
    print(f"Training: {len(train_indices)} samples, {train_X.shape[0]} cells "
          f"({train_X.shape[0]/total_cells*100:.1f}%)")
    print(f"Validation: {len(val_indices)} samples, {val_X.shape[0]} cells "
          f"({val_X.shape[0]/total_cells*100:.1f}%)")

    return train_dataset, val_dataset
