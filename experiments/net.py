"""
Baseline MLP for tabular datasets.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Simple feed-forward network for tabular features.
    Expected input: batch_size x input_dim
    Output: batch_size x num_classes
    """

    def __init__(self, params):
        """
        Args:
            params: (Params) should contain at least:
                - input_dim (int): number of tabular features
                - hidden_dim (int): hidden width (or use num_channels as hidden width)
                - num_classes (int): number of classes (2 for benign/malignant)
                - dropout_rate (float)
        """
        super(Net, self).__init__()

        self.input_dim = getattr(params, "input_dim")
        self.hidden_dim = getattr(params, "hidden_dim", getattr(params, "num_channels", 128))
        self.num_classes = getattr(params, "num_classes", 2)
        self.dropout_rate = params.dropout_rate

        # MLP blocks
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)

        self.fc3 = nn.Linear(self.hidden_dim, self.num_classes)

        
    def forward(self, x):
        """
        Args:
            x: (Tensor) batch_size x input_dim

        Returns:
            logits: (Tensor) batch_size x num_classes
        """
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        logits = self.fc3(x)
        return logits


def loss_fn(outputs, labels):
    """
    Standard cross entropy for classification.
    outputs: batch_size x num_classes (logits)
    labels: batch_size (int class indices)
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def accuracy(outputs, labels):
    """
    outputs: (np.ndarray) logits or probabilities, shape (batch_size, num_classes)
    labels: (np.ndarray) shape (batch_size,)
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


metrics = {
    "accuracy": accuracy,
}

#### inference cost accounting (mirrors experiments/model.py's DLN accounting) ####

def get_mlp_ops_dict(input_dim, hidden_dim, num_classes, fold_batchnorm=True):
    ops = {'multiplication': 0, 'addition': 0, 'comparison': 0, 'division': 0}

    def add_linear(I, O):
        ops['multiplication'] += I * O
        ops['addition'] += I * O

    def add_batchnorm(H):
        if fold_batchnorm:
            ops['multiplication'] += H
            ops['addition'] += H
        else:
            ops['addition'] += 2 * H
            ops['multiplication'] += H
            ops['division'] += H

    def add_relu(H):
        ops['comparison'] += H

    add_linear(input_dim, hidden_dim); add_batchnorm(hidden_dim); add_relu(hidden_dim)
    add_linear(hidden_dim, hidden_dim); add_batchnorm(hidden_dim); add_relu(hidden_dim)
    add_linear(hidden_dim, num_classes)

    return {k: v for k, v in ops.items() if v > 0}


def get_mlp_basic_gate_ops(input_dim, hidden_dim, num_classes, dtype='float16', fold_batchnorm=True):
    from experiments.utils import get_num_basic_gate_ops_from_dtype
    ops = get_mlp_ops_dict(input_dim, hidden_dim, num_classes, fold_batchnorm)
    num_hl_ops = sum(ops.values())
    num_bg_ops = sum(count * get_num_basic_gate_ops_from_dtype(op, dtype) for op, count in ops.items())
    return num_hl_ops, num_bg_ops, ops
