# from src.abstract.distance_base import DistanceBase
# normalised compression distance

def calculate_ncd(cx1: int, cx2: int, cx1cx2: int):
    """Function to calculate normalised compression distance.

    The distance measure the 'effort' needed to change one
    text to another. The lower the value, the easier it 
    is to convert text A -> text B. This means text B belong
    to the same class as text A.

    Args:
        cx1 (int): Length of compressed text - text to be classifed.
        cx2 (int): Length of compressed text - reference text.
        cx1cx2 (int): Combined length of cx1 and cx2. 

    Returns:
        distance (float): Normalised compression distance 
    """
    return ((cx1cx2 - min(cx1, cx2)) / max(cx1, cx2))

# class NcdMetric(DistanceBase):

#     def __init__(self):
#         ...

#     def calculate(self, cx1: int, 
#                   cx2: int, cx1cx2: int):
#         return ((cx1cx2 - min(cx1, cx2)) / max(cx1, cx2))