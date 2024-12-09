import numpy as np
import torch
from sklearn import metrics
from torchmetrics import CalibrationError


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


class Metrics:
    def __init__(self, loss=None, predicts=None, targets=None, logits=None):
        self._loss = loss if loss is not None else 0.0
        self._targets = targets if targets is not None else []
        self._predicts = predicts if predicts is not None else []
        self._logits = logits if logits is not None else []
        self._num_classes = None

    def update(self, other):
        if other is not None:
            self._predicts.extend(to_numpy(other._predicts))
            self._targets.extend(to_numpy(other._targets))
            self._logits.extend(to_numpy(other._logits))
            self._loss += other._loss

    def _calculate(self, metric, **kwargs):
        return metric(self._targets, self._predicts, **kwargs)

    @property
    def ece(self):
        if len(self._targets) > 0:
            self._num_classes = len(np.unique(self._targets))
            ece_fn = CalibrationError(task="multiclass", num_classes=self._num_classes)
            return ece_fn(torch.tensor(np.array(self._logits)), torch.tensor(np.array(self._targets)))
        else:
            return 0.0

    @property
    def loss(self):
        if len(self._targets) > 0:
            return self._loss / len(self._targets)
        else:
            return 0.0

    @property
    def macro_precision(self):
        score = self._calculate(
            metrics.precision_score, average="macro", zero_division=0
        )
        return score * 100

    @property
    def macro_recall(self):
        score = self._calculate(metrics.recall_score, average="macro", zero_division=0)
        return score * 100

    @property
    def micro_precision(self):
        score = self._calculate(
            metrics.precision_score, average="micro", zero_division=0
        )
        return score * 100

    @property
    def micro_recall(self):
        score = self._calculate(metrics.recall_score, average="micro", zero_division=0)
        return score * 100

    @property
    def accuracy(self):
        if self.size == 0:
            return 0
        score = self._calculate(metrics.accuracy_score)
        return score * 100

    @property
    def corrects(self):
        return self._calculate(metrics.accuracy_score, normalize=False)

    @property
    def size(self):
        return len(self._targets)
