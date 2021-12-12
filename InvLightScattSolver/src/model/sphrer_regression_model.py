from typing import List

import numpy as np
import torch

class TorchRegressionModel:
    def __init__(
            self,
            model_path: str,
            transform = None,
            batch_size: int = 1,
    ):
        self.model = torch.load(model_path, map_location='cpu')
        self.transform = transform
        self.batch_size = batch_size

    def preproc(self, indicatrice: np.ndarray):
        '''prepare input vector for using with model'''
        transformed_ind = indicatrice
        return transformed_ind

    def run(self, indicatrices: List[np.ndarray]):
        transformed_inds = [self.preproc(ind)[None] for ind in indicatrices]
        batch = np.stack(transformed_inds)

        return 0
