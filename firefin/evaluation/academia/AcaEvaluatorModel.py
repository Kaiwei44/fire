import typing
import pandas as pd
from ...core.eva_utils import QuantileReturns
from ...core.algorithm.regression import least_square, RollingRegressor, BatchRegressionResult
from ...common.config import logger
from ...core.algorithm.anomaly_test import AnomalyTest
from ...core.algorithm.fama_macbeth import FamaMacBeth

class AcaEvaluatorModel:
    def __init__(self, factor_portfolio: pd.DataFrame, return_adj: pd.DataFrame, n_jobs: int = 10, verbose: int = 0):
        """
        Parameters:
            factor_portfolio: pd.DataFrame
                factor_portfolio (Time × K-factors)
            return_adj: pd.DataFrame
                DataFrame of adjusted returns (Time × Stock)
            n_jobs: int
                Number of jobs to run in parallel
            verbose: int
                Verbosity level
        """

        self.factor_portfolio = factor_portfolio
        self.return_adj = return_adj
        self.n_jobs = n_jobs
        self.verbose = verbose
        
    def run_fama_macbeth(self,
                         window: int = 252,
                         return_stats: bool = False):
        """
        Perform Fama-MacBeth two-stage cross-sectional regression estimation.

        Parameters:
            window: int
                Rolling window size for the first-stage regressions (default is 252, i.e., one year)
            return_stats: bool
                Whether to return t-statistics and significance test results

        Returns:
            If return_stats is True:
                Tuple[RegressionResult, dict] → (regression results, statistics)
            Otherwise:
                RegressionResult
        """

        results = FamaMacBeth.run_regression(self.factor_portfolio, self.return_adj, window=window, n_jobs=self.n_jobs, verbose=self.verbose)
        if return_stats:
            stats = FamaMacBeth.test_statistics(results)
            return results, stats
        return results

    def run_regression(self, rolling: bool = False, window: int = 60, fit_intercept: bool = True) -> BatchRegressionResult | dict:
        """
        Run either static or rolling regression of returns on factor exposures.

        Parameters
        ----------
        rolling : bool, optional
            Whether to perform rolling regression, by default False.
        window : int, optional
            Rolling window size (only used if rolling=True), by default 60.
        fit_intercept : bool, optional
            Whether to include an intercept in the regression, by default True.

        Returns
        -------
        BatchRegressionResult | dict
            Regression result object (static) or a dictionary of rolling results.
        """
        if rolling:
            # Use rolling_regression function
            result = RollingRegressor(
                x = self.factor_portfolio, 
                y = self.return_adj,  
                fit_intercept = fit_intercept
                ).fit(
                    window = window,
                    n_jobs = self.n_jobs, 
                    verbose = self.verbose
                )
            
        else:
            # Time-by-time regression using least_square
            fields = ['alpha', 'beta', 'r2', 'r2_adj', 'residuals']
            results = {key: [] for key in fields}
            for t in self.factor.index:
                x_t = self.factor.loc[t]
                y_t = self.return_adj.loc[t]
                if x_t.isnull().any() or y_t.isnull().any():
                    continue
                reg_result = least_square(
                    x = x_t, 
                    y = y_t, 
                    fit_intercept = fit_intercept
                )
                for key in fields:
                    results[key].append(getattr(reg_result, key))
            result = BatchRegressionResult(**results)
        return result
        
    def run_anomaly_test(self,
                         portfolio_returns: QuantileReturns,
                         cov_type: typing.Optional[str] = None,
                         cov_kwds: typing.Optional[dict] = None,
                         return_stats: bool = False):
        """
        Perform anomaly test by regressing portfolio returns on a factor model.

        Parameters:
            return_stats : bool
                Whether to return regression statistics summary.

        Returns:
            If return_stats is True:
                Tuple[AnomalyTest, pd.DataFrame]
            Else:
                AnomalyTest
        """
        mkt_ret = pd.DataFrame(self.return_adj.mean(axis=1))
        tester = AnomalyTest(portfolio_returns= portfolio_returns, factor_model=mkt_ret)
        
        if return_stats:
            summary = tester.fit(cov_type=cov_type, cov_kwds=cov_kwds).test_statistics()
            return summary
        return tester


    def run_all(self) -> dict:
        """
        Run all available evaluation methods and return results in a dictionary.

        Returns
        -------
        dict
            A dictionary containing the results of all evaluation methods.
        """
        results = {}
        #Fama-MacBeth Regression
        logger.info("Running Fama-MacBeth Regression")
        results['fama_macbeth_res'], results['fama_macbeth_stat']= self.run_fama_macbeth(
            window=252,
            return_stats=True
        )
        logger.info("Fama-MacBeth Regression Completed")

        # Static Regression
        logger.info("Running Static Regression")
        results['regression'] = self.run_regression(rolling=False, fit_intercept=True)
        logger.info("Static Regression Completed")
            
        # Anomaly Test
        logger.info("Running Anomaly Test")
        for k, v in results['single_sort_res'].items():
            results['anomaly_stat'] = {k:self.run_anomaly_test(portfolio_returns= pd.DataFrame(v.iloc[:,-1]), return_stats= True)}
        logger.info("Anomaly Test Completed")

        return results