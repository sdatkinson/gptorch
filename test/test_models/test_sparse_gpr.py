# File: test_sparse_gpr.py
# File Created: Sunday, 1st September 2019 9:14:58 am
# Author: Steven Atkinson (steven@atkinson.mn)

import os

import pytest
import numpy as np
import torch

from gptorch.models.sparse_gpr import VFE, SVGP
from gptorch.kernels import Matern32
from gptorch import likelihoods
from gptorch import mean_functions
from gptorch.util import torch_dtype

from .common import gaussian_predictions

torch.set_default_dtype(torch_dtype)

_data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "models", 
    "sparse_gpr")


def atleast_col(func):
    """
    Decorator making sure that matrices loaded are column vectors if 1D
    """

    def wrapped():
        outputs = func()
        if isinstance(outputs, tuple):
            outputs = [o[:, np.newaxis] if o.ndim == 1 else o for o in outputs]
        else:
            outputs = outputs[:, np.newaxis] if outputs.ndim == 1 else outputs
        return outputs

    return wrapped


def _get_matrix(name):
    return np.loadtxt(os.path.join(_data_dir, name + ".dat"))


class _InducingData(object):
    """
    A few pieces in common with these models
    """

    @staticmethod
    @atleast_col
    def _xy():
        return _get_matrix("x"), _get_matrix("y")
    
    @staticmethod
    @atleast_col
    def _x_test():
        return _get_matrix("x_test")
    
    @staticmethod
    @atleast_col
    def _z():
        return _get_matrix("z")


class TestVFE(_InducingData):
    def test_init(self):
        x, y = _InducingData._xy()
        kernel = Matern32(x.shape[1], ARD=True)
        VFE(x, y, kernel)
        VFE(x, y, kernel, inducing_points=_InducingData._z())

        # TODO mean

    def test_compute_loss(self):
        x, y = _InducingData._xy()
        z = _InducingData._z()
        kernel = Matern32(1)
        kernel.length_scales.data = torch.zeros(1)
        kernel.variance.data = torch.zeros(1)
        likelihood = likelihoods.Gaussian(variance=1.0)

        model = VFE(x, y, kernel, inducing_points=z, likelihood=likelihood,
            mean_function=mean_functions.Zero(1))
        loss = model.compute_loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.ndimension() == 0
        # Computed while I trust the result.
        assert loss.item() == pytest.approx(8.842242323920674)

    def test_predict(self):
        """
        Just the ._predict() method
        """

        x, y = _InducingData._xy()
        z = _InducingData._z()
        kernel = Matern32(1)
        kernel.length_scales.data = torch.zeros(1)
        kernel.variance.data = torch.zeros(1)
        likelihood = likelihoods.Gaussian(variance=1.0)

        model = VFE(x, y, kernel, inducing_points=z, likelihood=likelihood,
            mean_function=mean_functions.Zero(1))

        x_test = torch.Tensor(_InducingData._x_test())
        mu, s = TestVFE._y_pred()
        gaussian_predictions(model, x_test, mu, s)

    @staticmethod
    @atleast_col
    def _y_pred():
        return _get_matrix("vfe_y_mean"),  _get_matrix("vfe_y_cov")  


class TestSVGP(_InducingData):
    def test_init(self):
        x, y = _InducingData._xy()
        kernel = Matern32(x.shape[1], ARD=True)
        SVGP(x, y, kernel)
        SVGP(x, y, kernel, inducing_points=_InducingData._z())

        SVGP(x, y, kernel, mean_function=mean_functions.Constant(y.shape[1]))
        SVGP(x, y, kernel, mean_function=torch.nn.Linear(x.shape[1], y.shape[1]))

    def test_compute_loss(self):
        x, y = _InducingData._xy()
        z = _InducingData._z()
        u_mu, u_l_s = TestSVGP._induced_outputs()
        kernel = Matern32(1)
        kernel.length_scales.data = torch.zeros(1)
        kernel.variance.data = torch.zeros(1)
        likelihood = likelihoods.Gaussian(variance=1.0)

        model = SVGP(x, y, kernel, inducing_points=z, likelihood=likelihood,
            mean_function=mean_functions.Zero(1))
        model.induced_output_mean.data = torch.Tensor(u_mu)
        model.induced_output_chol_cov.data = model.induced_output_chol_cov.\
            _transform.inv(torch.Tensor(u_l_s))

        loss = model.compute_loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.ndimension() == 0
        # Computed while I trust the result.
        assert loss.item() == pytest.approx(9.534628739243518)

        model_minibatch = SVGP(x, y, kernel, batch_size=1)
        loss_mb = model_minibatch.compute_loss()
        assert isinstance(loss_mb, torch.Tensor)
        assert loss_mb.ndimension() == 0

        model_full_mb = SVGP(x, y, kernel, inducing_points=z, likelihood=likelihood,
            mean_function=mean_functions.Zero(1), batch_size=x.shape[0])
        model_full_mb.induced_output_mean.data = torch.Tensor(u_mu)
        model_full_mb.induced_output_chol_cov.data = model_full_mb.induced_output_chol_cov.\
            _transform.inv(torch.Tensor(u_l_s))
        loss_full_mb = model_full_mb.compute_loss()
        assert isinstance(loss_full_mb, torch.Tensor)
        assert loss_full_mb.ndimension() == 0
        assert loss_full_mb.item() == pytest.approx(loss.item())

        model.compute_loss(model.X, model.Y)  # Just make sure it works!

    def test_predict(self):
        """
        Just the ._predict() method
        """

        x, y = _InducingData._xy()
        z = _InducingData._z()
        u_mu, u_l_s = TestSVGP._induced_outputs()
        kernel = Matern32(1)
        kernel.length_scales.data = torch.zeros(1)
        kernel.variance.data = torch.zeros(1)
        likelihood = likelihoods.Gaussian(variance=1.0)

        model = SVGP(x, y, kernel, inducing_points=z, likelihood=likelihood,
            mean_function=mean_functions.Zero(1))
        model.induced_output_mean.data = torch.Tensor(u_mu)
        model.induced_output_chol_cov.data = model.induced_output_chol_cov.\
            _transform.inv(torch.Tensor(u_l_s))

        x_test = torch.Tensor(_InducingData._x_test())
        mu, s = TestSVGP._y_pred()
        gaussian_predictions(model, x_test, mu, s)

    @staticmethod
    @atleast_col
    def _induced_outputs():
        return _get_matrix("q_mu"), _get_matrix("l_s")

    @staticmethod
    @atleast_col
    def _y_pred():
        return _get_matrix("svgp_y_mean"), _get_matrix("svgp_y_cov")
