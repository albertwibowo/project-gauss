from typing import Any
from src.abstract.algorithm_base import AlgorithmBase
# from src.abstract.distance_base import DistanceBase
from src.distance_metrics.ce_metric import calculate_ce

import numpy as np 

# TODO: add run method
class CeCompressor(AlgorithmBase):

    def __init__(self, compressor: Any,
                #  distance_metric: DistanceBase,
                 train_array: np.array,
                 test_array: np.array,
                 to_predict_array: np.array):
        
        self.compressor = compressor
        # self.dm = distance_metric
        self.train_array = train_array
        self.test_array = test_array
        self.to_predict_array = to_predict_array

        self.final_predict_result = None 
        self.test_predict_result = None 

    def run(self):
        ... 

    def validate(self):
        ...

    def save(self):
        ...