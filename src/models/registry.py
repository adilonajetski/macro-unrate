from src.models.random_walk import RandomWalk
from src.models.arima import ARIMA
from src.models.arimax import ARIMAX
#from src.models.sklearn_models import DirectRidge

def get_models():
    return [
        RandomWalk(),
        ARIMA(order=(2,1,2)),
        ARIMAX(order=(2,1,2)),
        #DirectRidge(alpha=1.0),
        #DirectRidge(alpha=10.0),
    ]