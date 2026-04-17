import numpy as np
import pandas as pd

from prediction.interfaces import Evaluator


class SimpleEvaluator(Evaluator):
    @staticmethod
    def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
        true_values = pd.to_numeric(y_true, errors="coerce")
        pred_values = pd.to_numeric(y_pred, errors="coerce")
        valid = true_values.notna() & pred_values.notna()
        if valid.sum() == 0:
            return float("nan")
        return float(np.abs(true_values[valid] - pred_values[valid]).mean())
