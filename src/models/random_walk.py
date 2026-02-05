import numpy as np
from src.models.base import ForecastModel

class RandomWalk(ForecastModel):
    @property
    def name(self) -> str:
        return "RandomWalk"
    
    def fit(self, y_train, X_train=None):
        self.last_value = float(y_train.iloc[-1])

    def predict(self, steps: int, X_test=None):
        return np.repeat(self.last_value, steps)