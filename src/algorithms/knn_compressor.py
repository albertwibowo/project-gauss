from typing import Any
from src.abstract.algorithm_base import AlgorithmBase
# from src.abstract.distance_base import DistanceBase
from src.distance_metrics.ncd_metric import calculate_ncd
import numpy as np 

#TODO: add validation method based on scikit learn metrics
class KnnCompressor(AlgorithmBase):
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
        self.k = 5

        self.final_predict_result = None 
        self.test_predict_result = None 

    def run(self):

        # for data to be predicted 
        for (x1, _) in self.to_predict_array:
            cx1 = len(self.compressor.compress(x1.encode()))
            distance_from_x1 = []
            for (x2, _) in self.train_array:
                cx2 = len(self.compressor.compress(x2.encode())) 
                x1x2 = " ".join([x1, x2])
                cx1x2 = len(self.compressor.compress(x1x2.encode()))
                ncd = calculate_ncd(cx1=cx1, cx2=cx2, cx1cx2=cx1x2)
                distance_from_x1.append(ncd) 
                sorted_idx = np.argsort(np.array(distance_from_x1))
                top_k_class = self.train_array[sorted_idx[:self.k], 1]
                self.final_predict_result = max(set(top_k_class), 
                                    key=top_k_class.count)

    def validate(self):
        ...

    def save(self):
        ...