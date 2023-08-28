from src.abstract.algorithm_base import AlgorithmBase
from src.distance_metrics.ncd_metric import calculate_ncd
from statistics import mode 
import numpy as np 
import pandas as pd 
import multiprocessing as mp
import itertools

#TODO: too many duplicated code - can be abstracted
class KnnCompressor(AlgorithmBase):
    def __init__(self,
                 base_df: pd.DataFrame,
                 to_predict_df: pd.DataFrame
                 ):
        
        self.base_df = base_df
        self.to_predict_df = to_predict_df
    
    def run(self, text_col:str, target_col:str, 
            k: int, sample_frac:float=1.0) -> pd.DataFrame:
        """Method to run the algorithm.

        Args:
            text_col (str): The name of the text column.
            target_col (str): The name of the target column.
            k (int): The number of neighbours to be considered.
            sample_frac: (float): The sampling fraction - 1.0 means no sampling.

        Returns:
            result_df (pd.DataFrame): result dataframe 
        """    
        
        temp_df = self.base_df.groupby(target_col).apply(lambda x: x.sample(frac=sample_frac)).reset_index(drop = True)
        temp_df[text_col] = temp_df[text_col].astype('str')
        to_predict_list = self.to_predict_df[text_col].values.tolist()
        base_list = temp_df[text_col].values.tolist()

        # all combination of text
        text_list = [x for x in itertools.product(to_predict_list, base_list)]

        # holder for result 
        preds = []
        distance = []
        
        # multiprocessing
        pool = mp.Pool(mp.cpu_count())
        for d in pool.starmap(calculate_ncd, text_list, chunksize=5):
            distance.append(d)
        pool.close()

        # function to chunk distance data per item to be predicted
        chunking = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
        chunks = chunking(distance, len(base_list))
        for i in range(len(chunks)):
            sorted_idx = np.argsort(np.array(chunks[i]))
            top_k_class = temp_df.reset_index(drop=True).loc[sorted_idx[:k], target_col]
            preds.append(mode(top_k_class))
        
        result_df = self.to_predict_df
        result_df['prediction'] = preds

        return result_df