from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.models.base import ForecastModel

class ARIMAX(ForecastModel):
    def __init__(self, order=(2,1,2), trend="c"):
        self.order = order
        self.trend = trend

    @property
    def name(self) -> str:
        return f"ARIMAX{self.order}"

    def fit(self, y_train, X_train) -> None:
        self.res = SARIMAX(
            y_train,
            exog=X_train,
            order=self.order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False, maxiter=2000, method="lbfgs")

    def predict(self, steps: int, X_test):
        return self.res.get_forecast(steps, exog=X_test).predicted_mean