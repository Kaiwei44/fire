import pandas as pd

from ...core.algorithm.newey_west_ttest_1samp import NeweyWestTTest
from ...core.algorithm.regression import RollingRegressor, BatchRegressionResult


class FamaMacBeth:

    @staticmethod
    def run_regression(
        factor: pd.DataFrame | pd.Series, return_adj: pd.DataFrame, window: int = 252, n_jobs=4, verbose: int = 0
    ) -> BatchRegressionResult:
        """
        Run Fama-MacBeth regression."
        """
        if isinstance(factor, pd.Series):
            # Convert series to DataFrame for consistency
            factor = pd.concat([factor] * return_adj.shape[1], axis=1)
            factor.columns = return_adj.columns
        elif isinstance(factor, pd.DataFrame):
            pass
        else:
            raise ValueError("Factor must be a pandas Series or DataFrame.")

        # Note: Calculate excess returns if necessary
        # return_adj = return_adj - risk_free_rate
        # excess return is different in many cases, we leave it to the user to handle this.

        # First step: Time-series regressions
        r = RollingRegressor(factor, return_adj, None, fit_intercept=True).fit(window, n_jobs=n_jobs, verbose=verbose)

        # Second step: Cross-sectional regressions
        # This step involves regressing the time-series regression coefficients on the factors
        r = RollingRegressor(r.beta, return_adj, None, fit_intercept=True).fit(window=None, axis=1, n_jobs=n_jobs, verbose=verbose)

        return r

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
