from abc import ABC, abstractmethod

class ForecastModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fit(self, y_train, X_train=None) -> None:
        ...

    @abstractmethod
    def predict(self, steps: int, X_test=None):
        """Return an array-like of length = steps"""
        ...

