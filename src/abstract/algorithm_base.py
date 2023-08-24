from abc import ABC, abstractmethod

class AlgorithmBase(ABC):

    def __init__(self):
        pass 

    @abstractmethod
    def run(self):
        ...

    @abstractmethod
    def save(self):
        ...