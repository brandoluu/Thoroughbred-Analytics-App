import torch
from torch.utils.data import Dataset


num_cols = [
    "name_encoded", "rawErg", "erg", "age", "sire", "fee", "crop", "dam", 
    "ems3", "bmSire", "form", "damForm", "sex_C", "sex_F", "sex_G", "sex_R"
]

class HorseDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        return

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        batch = {
            "name_encoded":    torch.tensor(row["name_encoded"],   dtype=torch.float32),
            #"rating":  torch.tensor(row["rating"], dtype=torch.float32),
            "rawErg":  torch.tensor(row["rawErg"], dtype=torch.float32),
            "erg":     torch.tensor(row["erg"], dtype=torch.float32),
            "age":     torch.tensor(row["age"], dtype=torch.float32),
            "sire":    torch.tensor(row["sire"],   dtype=torch.float32),
            "fee":     torch.tensor(row["fee"], dtype=torch.float32),
            "crop":    torch.tensor(row["crop"], dtype=torch.float32),
            "dam":     torch.tensor(row["dam"],    dtype=torch.float32),
            "ems3":    torch.tensor(row["ems3"], dtype=torch.float32),
            "bmSire":  torch.tensor(row["bmSire"], dtype=torch.float32),
            "form":    torch.tensor(row["form"], dtype=torch.float32),
            "damForm": torch.tensor(row["damForm"], dtype=torch.float32),
            "numeric": torch.tensor(row[num_cols].values, dtype=torch.float32), # combines the hot encoded columns together
        }

        return batch