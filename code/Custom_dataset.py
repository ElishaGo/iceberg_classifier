from torch.utils.data import Dataset
import numpy as np
from data_utils import log_norm
import torch



class IcebergImageDataset(Dataset):
    def __init__(self, data_table, transform=None):
        self.id = data_table.loc[:, 'id']
        self.band_1 = data_table.loc[:, 'band_1'].apply(lambda x: torch.tensor(np.array(x).reshape(75, 75)).float())
        self.band_2 = data_table.loc[:, 'band_2'].apply(lambda x: torch.tensor(np.array(x).reshape(75, 75)).float())
        # replace na with mean value
        data_table['inc_angle'] = data_table.inc_angle.replace('na', value=None)
        data_table['inc_angle'] = data_table.loc[:, 'inc_angle'].fillna(data_table.inc_angle.mean())
        # minmax normalization
        self.inc_angle = (data_table.inc_angle - data_table.inc_angle.min()) / (
                data_table.inc_angle.max() - data_table.inc_angle.min())
        self.label = torch.tensor(data_table.loc[:, 'is_iceberg']).long()
        self.transform = transform

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        # create image data and normalize
        band1 = self.band_1[idx]
        band2 = self.band_2[idx]
        band1 = log_norm(band1)
        band2 = log_norm(band2)

        # add inc_angle as a channel
        inc_angle = self.inc_angle[idx]
        image = torch.stack((torch.ones_like(band1) * inc_angle, band1, band2), axis=0)

        label = self.label[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
