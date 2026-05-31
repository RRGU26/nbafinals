"""Bayesian linear regression with conjugate Normal-InverseGamma prior.

Standardizes features internally for numerical stability. Returns predictive
mean and variance for new inputs.
"""

import numpy as np


class BayesianLinearModel:
    def __init__(self, prior_precision: float = 1.0):
        self.prior_precision = prior_precision
        self.beta_mean = None
        self.beta_cov = None
        self.sigma2 = None
        self.feature_mean = None
        self.feature_std = None

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.feature_mean = X.mean(axis=0)
            self.feature_std = X.std(axis=0)
            # Avoid div-by-zero for constant features
            self.feature_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
        return (X - self.feature_mean) / self.feature_std

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        Xs = self._standardize(X, fit=True)
        Xs = np.hstack([np.ones((n, 1)), Xs])  # intercept

        if sample_weight is None:
            sample_weight = np.ones(n, dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
        sample_weight = sample_weight * n / sample_weight.sum()

        # Weighted normal equations (efficient, no diag matrix construction)
        sw = sample_weight[:, None]  # (n, 1)
        XtWX = (Xs * sw).T @ Xs
        XtWy = (Xs * sw).T @ y

        prior_prec_mat = self.prior_precision * np.eye(d + 1)
        prior_prec_mat[0, 0] = 1e-6  # weak prior on intercept

        Sigma_inv = prior_prec_mat + XtWX
        Sigma = np.linalg.inv(Sigma_inv)
        mu = Sigma @ XtWy

        residuals = y - Xs @ mu
        eff_n = sample_weight.sum()
        self.sigma2 = float((sample_weight * residuals**2).sum() / max(eff_n - d - 1, 1))
        self.beta_mean = mu
        self.beta_cov = Sigma * self.sigma2

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        Xs = self._standardize(X, fit=False)
        n = Xs.shape[0]
        Xs = np.hstack([np.ones((n, 1)), Xs])
        mean = Xs @ self.beta_mean
        var = self.sigma2 + np.einsum("ij,jk,ik->i", Xs, self.beta_cov, Xs)
        return mean, var

    def predict_distribution(self, X: np.ndarray) -> dict:
        mean, var = self.predict(X)
        std = np.sqrt(var)
        return {
            "mean": mean,
            "std": std,
            "lo_10": mean - 1.282 * std,
            "hi_90": mean + 1.282 * std,
            "lo_25": mean - 0.674 * std,
            "hi_75": mean + 0.674 * std,
        }

    def win_prob(self, X: np.ndarray) -> np.ndarray:
        from scipy.stats import norm
        mean, var = self.predict(X)
        return norm.cdf(mean / np.sqrt(var))

    def coefficients_native_scale(self, feature_names: list[str]) -> dict:
        """Return coefficients in original (unstandardized) feature scale."""
        # standardized: y = β₀ + Σ βᵢ * (xᵢ - μᵢ)/σᵢ
        # native:      y = (β₀ - Σ βᵢ μᵢ/σᵢ) + Σ (βᵢ/σᵢ) xᵢ
        intercept = float(self.beta_mean[0])
        coefs_std = self.beta_mean[1:]
        coefs_native = coefs_std / self.feature_std
        intercept_native = intercept - (coefs_std * self.feature_mean / self.feature_std).sum()
        out = {"intercept": float(intercept_native)}
        for name, c in zip(feature_names, coefs_native):
            out[name] = float(c)
        return out
