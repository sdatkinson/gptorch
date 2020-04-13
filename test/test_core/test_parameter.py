# File: test_param.py
# File Created: Wednesday, 10th July 2019 11:47:40 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest
import numpy as np
import torch
from torch.distributions import transforms

from gptorch.core import Parameter


class TestParameter(object):
    def test_init(self):
        x = torch.eye(3) + torch.ones(3, 3)

        Parameter(x)
        Parameter(x, transform=transforms.ExpTransform())
        Parameter(x, transform=transforms.LowerCholeskyTransform())

    def test_access(self):
        """
        Test accessing the value.
        """
        p = Parameter(torch.DoubleTensor([1.0]))
        assert isinstance(p.data, torch.DoubleTensor)
        assert isinstance(p.data.numpy(), np.ndarray)

    def test_transform_inverse(self):
        x = torch.rand(3, 3)
        t = transforms.ExpTransform()
        p = Parameter(x, transform=t)

        expected_data = t.inv(x)
        actual_data = p.data
        assert all(
            [
                e.data.numpy() == pytest.approx(a.data.numpy())
                for e, a in zip(expected_data.flatten(), actual_data.flatten())
            ]
        )

    def test_transform_forward(self):
        x = torch.rand(3, 3)
        t = transforms.ExpTransform()
        p = Parameter(x, transform=t)

        actual_forward = p.transform()
        assert all(
            [
                e.data.numpy() == pytest.approx(a.data.numpy())
                for e, a in zip(x.flatten(), actual_forward.flatten())
            ]
        )
