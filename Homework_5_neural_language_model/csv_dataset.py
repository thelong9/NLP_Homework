from typing import Any, Dict, Optional, Tuple
import pandas as pd 
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

class CSV_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.loc[idx, "text"]
    
if __name__ == "__main__":
    data_path = "neural_language_model/data/test.csv"
    ds = pd.read_csv(data_path)
    dataset = CSV_Dataset(ds)
    print(dataset)
    print(dataset[1])
    
    