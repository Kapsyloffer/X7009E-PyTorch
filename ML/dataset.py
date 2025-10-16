import json
import torch
from torch.utils.data import Dataset

class ItemReorderingDataset(Dataset):
    def __init__(self, json_path, target_order_fn=None):
        with open(json_path, 'r') as f:
            raw = json.load(f)
        
        # Each entry is one item
        self.items = [list(entry["data"].values()) for entry in raw]
        self.ids = [entry["id"] for entry in raw]
        
        # Define a target order (label)
        # For now, we can sort items by total processing time as a toy target
        if target_order_fn is None:
            target_order_fn = lambda x: sorted(range(len(x)), key=lambda i: sum(x[i]))
        self.target_order = target_order_fn(self.items)
        
    def __len__(self):
        return 1  # The entire dataset is one sample for now

    def __getitem__(self, idx):
        src = torch.tensor(self.items, dtype=torch.float)
        tgt = torch.tensor(self.target_order, dtype=torch.long)
        return src, tgt
