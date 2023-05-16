import numpy as np
import time
import torch as th
import torch.nn as nn

from captum._utils.models.linear_model import LinearModel

from scipy.stats import invgamma
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from typing import Any, Dict


class BayesianLinearRegression:
    """
    Bayesian Linear Regression model.

    Args:
        percent (int): Percentage for the credible intervals.
            Default to 95
        l2 (bool): Whether to use l2 regularisation.
            Default to ``True``
    """

    def __init__(self, percent=95, l2=True):
        self.percent = percent
        self.l2 = l2

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        compute_creds=True,
    ):
        """
        Fit the bayesian linear regression.

        Arguments:
            X (np.ndarray): Data.
            y (np.ndarray): Target.
            sample_weight (np.ndarray): Sample weights.
            compute_creds (bool): Whether to compute credible intervals.
                Default to ``True``
        """

        # store weights
        weights = sample_weight

        # add intercept
        X = np.concatenate((np.ones(X.shape[0])[:, None], X), axis=1)
        diag_pi_z = np.zeros((len(weights), len(weights)))
        np.fill_diagonal(diag_pi_z, weights)

        if self.l2:
            V_Phi = np.linalg.inv(
                X.transpose().dot(diag_pi_z).dot(X) + np.eye(X.shape[1])
            )
        else:
            V_Phi = np.linalg.inv(X.transpose().dot(diag_pi_z).dot(X))

        Phi_hat = V_Phi.dot(X.transpose()).dot(diag_pi_z).dot(y)

        N = X.shape[0]
        Y_m_Phi_hat = y - X.dot(Phi_hat)

        s_2 = (1.0 / N) * (
            Y_m_Phi_hat.dot(diag_pi_z).dot(Y_m_Phi_hat)
            + Phi_hat.transpose().dot(Phi_hat)
        )

        self.score = s_2

        self.s_2 = s_2
        self.N = N
        self.V_Phi = V_Phi
        self.Phi_hat = Phi_hat
        self.coef_ = Phi_hat[1:]
        self.intercept_ = Phi_hat[0]
        self.weights = weights

        if compute_creds:
            self.creds = self.get_creds(percent=self.percent, n_samples=len(X))
        else:
            self.creds = None

        self.crit_params = {
            "s_2": self.s_2,
            "N": self.N,
            "V_Phi": self.V_Phi,
            "Phi_hat": self.Phi_hat,
            "creds": self.creds,
        }

        return self

    def predict(self, data):
        """
        The predictive distribution.
        Arguments:
            data: The data to predict
        """
        q_1 = np.eye(data.shape[0])
        data_ones = np.concatenate(
            (np.ones(data.shape[0])[:, None], data), axis=1
        )

        # Get response
        response = np.matmul(data, self.coef_)
        response += self.intercept_

        # Compute var
        temp = np.matmul(data_ones, self.V_Phi)
        mat = np.matmul(temp, data_ones.transpose())
        var = self.s_2 * (q_1 + mat)
        diag = np.diagonal(var)

        return response, np.sqrt(diag)

    def get_ptg(self, desired_width):
        """
        Compute the ptg perturbations.
        """
        cert = (desired_width / 1.96) ** 2
        S = self.coef_.shape[0] * self.s_2
        T = np.mean(self.weights)
        return 4 * S / (self.coef_.shape[0] * T * cert)

    def get_creds(self, percent=95, n_samples=10_000, get_intercept=True):
        """
        Get the credible intervals.
        Arguments:
            percent: the percent cutoff for the credible interval, i.e., 95 is 95% credible interval
            n_samples: the number of samples to compute the credible interval
            get_intercept: whether to include the intercept in the credible interval
        """
        samples = self.draw_posterior_samples(
            n_samples, get_intercept=get_intercept
        )
        creds = np.percentile(
            np.abs(samples - (self.Phi_hat if get_intercept else self.coef_)),
            percent,
            axis=0,
        )
        return creds

    def draw_posterior_samples(self, num_samples, get_intercept=False):
        """
        Sample from the posterior.

        Arguments:
            num_samples: number of samples to draw from the posterior
            get_intercept: whether to include the intercept
        """

        sigma_2 = invgamma.rvs(
            self.N / 2, scale=(self.N * self.s_2) / 2, size=num_samples
        )

        phi_samples = []
        for sig in sigma_2:
            sample = multivariate_normal.rvs(
                mean=self.Phi_hat, cov=self.V_Phi * sig, size=1
            )
            phi_samples.append(sample)

        phi_samples = np.vstack(phi_samples)

        if get_intercept:
            return phi_samples
        else:
            return phi_samples[:, 1:]


class NormLayer(nn.Module):
    def __init__(self, mean, std, n=None, eps=1e-8) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)


def train_bayes_model(
    model: LinearModel,
    dataloader: DataLoader,
    construct_kwargs: Dict[str, Any],
    norm_input: bool = False,
    **fit_kwargs,
):
    r"""
    Fit a BayesianLinearRegression model.

    Args
        model
            The model to train.
        dataloader
            The data to use. This will be exhausted and converted to numpy
            arrays. Therefore please do not feed an infinite dataloader.
        norm_input
            Whether or not to normalize the input
        construct_kwargs
            Additional arguments provided to the `sklearn_trainer` constructor
        fit_kwargs
            Other arguments to send to `sklearn_trainer`'s `.fit` method
    """
    num_batches = 0
    xs, ys, ws = [], [], []
    for data in dataloader:
        if len(data) == 3:
            x, y, w = data
        else:
            assert len(data) == 2
            x, y = data
            w = None

        xs.append(x.cpu().numpy())
        ys.append(y.cpu().numpy())
        if w is not None:
            ws.append(w.cpu().numpy())
        num_batches += 1

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if len(ws) > 0:
        w = np.concatenate(ws, axis=0)
    else:
        w = None

    if norm_input:
        mean, std = x.mean(0), x.std(0)
        x -= mean
        x /= std

    t1 = time.time()
    blr = BayesianLinearRegression(**construct_kwargs)
    blr.fit(x, y, sample_weight=w, **fit_kwargs)

    t2 = time.time()

    # extract model device
    device = model.device if hasattr(model, "device") else "cpu"

    num_outputs = blr.coef_.shape[0] if blr.coef_.ndim > 1 else 1
    weight_values = th.FloatTensor(blr.coef_).to(device)
    bias_values = th.FloatTensor([blr.intercept_]).to(device)
    model._construct_model_params(
        norm_type=None,
        weight_values=weight_values.view(num_outputs, -1),
        bias_value=bias_values.squeeze().unsqueeze(0),
        classes=None,
    )

    if norm_input:
        model.norm = NormLayer(mean, std)

    # Save creds to model if provided
    if blr.creds is not None:
        model.creds = (
            th.Tensor(blr.creds).unsqueeze(0)
            if model.creds is None
            else th.cat([model.creds, th.Tensor(blr.creds).unsqueeze(0)])
        )

    return {"train_time": t2 - t1}


class BLRLinearModel(LinearModel):
    def __init__(self, l2: bool, **kwargs) -> None:
        r"""
        Factory class to construct a `LinearModel` with BLR training method.

        Args:
            l2
                L2 regularisation
            kwargs
                The kwargs to pass to the construction of the sklearn model
        """
        super().__init__(train_fn=train_bayes_model, l2=l2, **kwargs)
        self.creds = None


class BLRRegression(BLRLinearModel):
    def __init__(self, **kwargs) -> None:
        r"""
        Factory class. Trains a model with BayesianLinearRegression(l2=False).
        """
        super().__init__(l2=False, **kwargs)


class BLRRidge(BLRLinearModel):
    def __init__(self, **kwargs) -> None:
        r"""
        Factory class. Trains a model with BayesianLinearRegression(l2=False).
        """
        super().__init__(l2=True, **kwargs)
