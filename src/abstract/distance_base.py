from abc import ABC, abstractmethod

class DistanceBase(ABC):

    def __init__(self):
        pass 

    @abstractmethod
    def calculate(self):
        ...