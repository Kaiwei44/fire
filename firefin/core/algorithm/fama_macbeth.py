import numpy as np
import pandas as pd
from ...core.algorithm.newey_west_ttest_1samp import NeweyWestTTest
from ...core.algorithm.regression import RollingRegressor, BatchRegressionResult


class FamaMacBeth:

    @staticmethod
    def run_regression(
        factor: pd.DataFrame | pd.Series | np.ndarray | list[pd.Series] | list[np.ndarray] | BatchRegressionResult, 
        return_adj: pd.DataFrame, 
        window: int = 252,
        skip_time_series_regression: bool = False,
        n_jobs=4, 
        verbose: int = 0
    ) -> BatchRegressionResult:
        """
        Run Fama-MacBeth regression."

        factor: pd.DataFrame | pd.Series | np.ndarray | list[pd.Series] | list[np.ndarray]
            Factor data, can be a single series, a single array, a list of series, or a list of arrays.
            Note: we consider each pd.Series or np.ndarray or pd.DataFrame as a single factor.
        return_adj: pd.DataFrame
            Adjusted return data.
        window: int
            Window size for time series regression.
        skip_time_series_regression: bool
            Whether to skip time series regression.(If True, factor should be a BatchRegressionResult.Beta)
        n_jobs: int
            Number of jobs to run in parallel.
        """
        # 仅需要调整 pd.Series 和 list[pd.Series] 的情况
        if isinstance(factor, pd.Series):
            # Convert series to DataFrame for consistency
            factor = pd.concat([factor] * return_adj.shape[1], axis=1)
            factor.columns = return_adj.columns
        if isinstance(factor, list[pd.Series]):
            N=return_adj.shape[1]
            factor = np.stack([np.tile(f.values.reshape(-1, 1), (1, N)) for f in factor], axis=0 )

        if not isinstance(return_adj, pd.DataFrame):
            raise ValueError("return_adj must be a pandas DataFrame.")

        # Note: Calculate excess returns if necessary
        # return_adj = return_adj - risk_free_rate
        # excess return is different in many cases, we leave it to the user to handle this.

        # First step: Time-series regressions
        if skip_time_series_regression:
            r = factor
            assert isinstance(r, BatchRegressionResult), "factor should be a BatchRegressionResult if skip_time_series_regression is True."
        else:
            r = RollingRegressor(factor, return_adj, None, fit_intercept=True).fit(window, n_jobs=n_jobs, verbose=verbose)
        # Second step: Cross-sectional regressions
        # This step involves regressing the time-series regression coefficients on the factors

        lambda_sum_df = None
        for tau in range(window):
            ret_prime = return_adj.shift(tau)
            r_tau = RollingRegressor(r.beta, ret_prime, None, fit_intercept=True).fit(window=None, axis=1, n_jobs=n_jobs, verbose=verbose)
            if lambda_sum_df is None:
                lambda_sum_df = r_tau.beta.copy()
            else:
                lambda_sum_df = lambda_sum_df.add(r_tau.beta)

        lambda_df = lambda_sum_df/window
        return lambda_df

    @staticmethod
    def test_statistics(results: BatchRegressionResult) -> pd.Series:
        # mean and std

        mean_beta = results.beta.mean()
        std_beta = results.beta.std()

        mean_alpha = results.alpha.mean()
        std_alpha = results.alpha.std()

        # t-statistics

        t_stat, p_value, se = NeweyWestTTest.newey_west_ttest_1samp(results.beta, popmean=0, lags=6, nan_policy="omit")

        return pd.Series(
            {
                "mean_beta": mean_beta,
                "std_beta": std_beta,
                "mean_alpha": mean_alpha,
                "std_alpha": std_alpha,
                "t_stat": t_stat,
                "p_value": p_value,
                "se": se,
            }
        )
