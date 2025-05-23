import typing
import pandas as pd
import numpy as np
from ...core.eva_utils import QuantileReturns
from ...core.algorithm.regression import least_square, RollingRegressor, BatchRegressionResult
from ...common.config import logger
from ...core.algorithm.cross_sectional_regression import cross_sectional_regression
from ...core.algorithm.fama_macbeth import FamaMacBeth

class AcaEvaluatorModel:
    def __init__(self, factor_portfolio: list[pd.Series], 
                        return_adj: pd.DataFrame, 
                        n_jobs: int = 10,
                        time_series_window: int = 60,
                        all_time_series_regression: bool = True,
                        verbose: int = 0):
        """
        Parameters:
            factor_portfolio: list[pd.Series]
                factor_portfolio (Time × K-factors)
            return_adj: pd.DataFrame
                DataFrame of adjusted returns (Time × Stock)
            n_jobs: int
                Number of jobs to run in parallel
            time_series_window: int
                Window size for time series regression
            all_time_series_regression: bool
                Whether to run all time series regression at once
            verbose: int
                Verbosity level
        """

        self.factor_portfolio = factor_portfolio
        self.return_adj = return_adj
        self.n_jobs = n_jobs
        self.verbose = verbose

        # define time series window or all time series regression
        if all_time_series_regression:
            self.time_series_window = return_adj.shape[0]
        else:
            self.time_series_window = time_series_window

        # verify the data shape
        for factor in factor_portfolio:
            if factor.shape[0] != return_adj.shape[0]:
                raise ValueError("The number of rows in factor_portfolio and return_adj must be the same")

    def run_time_series_regression(self, fit_intercept: bool = True):
        """
        Parameters:
            window: int
                Window size for regression
            if_all_in_one: bool
                Whether to run all in one or not
        """
        # multi-linear regression, x should be a list of pd.DataFrame Like return_adj, expand series to frame
        x = []
        for portfolio in self.factor_portfolio:
            # expand portfolio to a pd.DataFrame like return_adj, copy series to each column
            x.append(pd.DataFrame(np.tile(portfolio.values[:,np.newaxis], (1, self.return_adj.shape[1])),index=self.return_adj.index, columns=self.return_adj.columns))

        # perform multiple time-series regressions
        self.time_series_res = RollingRegressor(x, self.return_adj, fit_intercept=fit_intercept).fit(window=self.time_series_window, n_jobs=self.n_jobs)


    def run_cross_sectional_regression(self):
        """
        Parameters:
            None
        Returns:
            cross_sectional_res: BatchRegressionResult
                Cross-sectional regression result
        """
        # if already done time series regression, use the result
        if self.time_series_res is None:
            self.run_time_series_regression()
        
        return cross_sectional_regression(self.time_series_res, self.return_adj, window=self.time_series_window, skip_time_series_regression=True, n_jobs=self.n_jobs, verbose=self.verbose)

    def run_fama_macbeth_regression(self):
        """
        
        Fama-MacBeth regression

        Parameters:
            None

        Returns:
            fama_macbeth_res: BatchRegressionResult
                Fama-MacBeth regression result
        """

        # if already done time series regression, use the result
        if self.time_series_res is None:
            self.run_time_series_regression()

        return FamaMacBeth.run_regression(self.time_series_res, self.return_adj, window=self.time_series_window, skip_time_series_regression=True, n_jobs=self.n_jobs, verbose=self.verbose)