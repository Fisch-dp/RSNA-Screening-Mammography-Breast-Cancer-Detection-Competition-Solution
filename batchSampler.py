import torch
import numpy as np
import pandas as pd
import math

class MultiImageBatchSampler(torch.utils.data.Sampler):
    def __init__(self, df, batch_size, sample_per_id_view=1, drop_last=False, shuffle=True, random_append=False):
        self.batch_size = batch_size
        self.df = df

        assert df["view"].isin(['CC', 'MLO']).all(), "view must be only CC or MLO"

        if shuffle:
            self.df = self.df.groupby(by=['prediction_id']).apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
        self.df = self.df.groupby(by=['prediction_id',"view"]).head(sample_per_id_view).reset_index(drop=True)

        extra_data = len(self.df) % self.batch_size
        if extra_data != 0 and not drop_last and random_append:
            df = self.df.groupby(by=['prediction_id']).apply(lambda x: x.sample(frac=1))
            self.df = pd.concat([self.df, df.head(self.batch_size - extra_data)]).reset_index(drop=True)
        elif extra_data != 0 and drop_last:
            self.df = self.df.head(len(self.df) - extra_data).reset_index(drop=True)

        self.index = self.df.index.tolist()

    def __iter__(self):
        for i in range(len(self.index) // self.batch_size):
            yield self.index[i * self.batch_size:(i + 1) * self.batch_size]

    def __len__(self):
        return math.ceil(len(self.index) / self.batch_size).astype(int)