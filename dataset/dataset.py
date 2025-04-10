from typing import List
import torch.utils.data as data

class TorchDataset(data.Dataset):

    def __init__(self, samples:List[dict]):        
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx:int) -> dict:
        return self.samples[idx]


def make_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
