import pandas as pd
from ...core.algorithm.regression import RollingRegressor, BatchRegressionResult
import math
import numpy as np

def cross_sectional_regression(
        factor: pd.DataFrame | pd.Series | list[pd.Series] | list[np.ndarray] | BatchRegressionResult, 
        return_adj: pd.DataFrame, 
        window: int = 252, 
        cov_type : str | None =None,
        skip_time_series_regression: bool = False,
        n_jobs=4, 
        verbose: int = 0
) -> BatchRegressionResult:
    """
    Run cross sectional  regression.
    input:
    factor : pd.DataFrame | pd.Series
    return_adj : pd.DataFrame
    window : int
    cov_type : "HAC" for newey west estimation or None for OLS

    return:
    risk premium : result.beta
    anomaly : result.alpha
    anomaly t statistics : result.alpha_t
    """
    if isinstance(factor, pd.Series):
        # Convert series to DataFrame for consistency
        factor = pd.concat([factor] * return_adj.shape[1], axis=1)
        factor.columns = return_adj.columns
    if isinstance(factor, list[pd.Series]):
        N = return_adj.shape[1]
        factor = np.stack([np.tile(f.values.reshape(-1, 1), (1, N)) for f in factor], axis=0)
    if not isinstance(return_adj, pd.DataFrame) and not isinstance(return_adj, pd.Series):
        raise ValueError("return_adj must be a pandas DataFrame.")
    # Note: Calculate excess returns if necessary
    # return_adj = return_adj - risk_free_rate
    # excess return is different in many cases, we leave it to the user to handle this.

    # First step: Time-series regressions
    if skip_time_series_regression:
        assert isinstance(factor, BatchRegressionResult), "factor should be a BatchRegressionResult if skip_time_series_regression is True."
        r = factor
    else:
        r = RollingRegressor(factor, return_adj, None, fit_intercept=True).fit(window, n_jobs=n_jobs, verbose=verbose)

    mean_return = []
    T= return_adj.shape[0]
    N= return_adj.shape[1]
    for t in range(T-window):
        mean_ret = return_adj.iloc[t:t+window,:].mean(axis=0)
        mean_return.append(mean_ret)

    # Second step: Cross sectional regressions
    mean_return_df = pd.DataFrame(mean_return, index= return_adj.index[-len(mean_return):]).reindex(return_adj.index)
    if cov_type is None:
        m=RollingRegressor(r.beta, mean_return_df, None, fit_intercept=True).fit(window=None, axis=1, n_jobs=n_jobs, verbose=verbose)
    else:
        optimal_lags = math.floor(4 * (N / 100) ** (2 / 9))
        m=RollingRegressor(r.beta, mean_return_df, None, fit_intercept=True).fit(window=None, axis=1, cov_type= cov_type, cov_kwds={'maxlags': optimal_lags},n_jobs=n_jobs, verbose=verbose)

    return BatchRegressionResult(beta=m.beta, alpha=m.alpha)
