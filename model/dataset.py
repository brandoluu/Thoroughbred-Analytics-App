import torch
from torch.utils.data import Dataset

num_cols = [
    "name","rawErg","erg","age","fee","crop","ems3","form","damForm",
    "sex_C","sex_F","sex_G","sex_R"
]

class HorseDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        batch = {
            "name":    torch.tensor(row["name_id"],   dtype=torch.long),
            "sire":    torch.tensor(row["sire_id"],   dtype=torch.long),
            "dam":     torch.tensor(row["dam_id"],    dtype=torch.long),
            "bmSire":  torch.tensor(row["bmSire_id"], dtype=torch.long),
            "numeric": torch.tensor(row[num_cols].values, dtype=torch.float32),
            "rating":  torch.tensor(row["rating"], dtype=torch.float32),
        }
        return batch