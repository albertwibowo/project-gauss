from abc import ABC, abstractmethod


class AlgorithmBase(ABC):

    @abstractmethod
    def run(self): ...
