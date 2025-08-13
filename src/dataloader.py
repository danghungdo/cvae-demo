import numpy as np
import torch
import math

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=False):
        """
        DataLoader class
        Args:
            dataset: dataset to load
            batch_size: batch size
            shuffle: if True, shuffle the dataset
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(dataset)

    def __iter__(self):
        """
        Iterate over the dataset
        """
        self.index = 0
        self.indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        """
        Get the next batch
        """
        if self.index >= self.n_samples:
            raise StopIteration

        batch_indices = self.indices[
            self.index : min(self.index + self.batch_size, self.n_samples)
        ]
        self.index += self.batch_size

        batch_x = []
        batch_y = []
        for idx in batch_indices:
            x, y = self.dataset[idx]
            batch_x.append(x)
            batch_y.append(y)

        # stack tensors
        batch_x = torch.stack(batch_x)
        batch_y = torch.tensor(batch_y, dtype=torch.long)

        return batch_x, batch_y

    def __len__(self):
        """
        Return the number of batches
        """
        return math.ceil(self.n_samples / self.batch_size)
