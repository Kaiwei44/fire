"""
Portfolio Sort Implementation for Academic Research
---------------------------------------------------
This module provides a class for performing single and double portfolio sorts
based on characteristics, market capitalization, and returns. The implementation
focuses on clarity, documentation, and best practices for financial research.
"""

import typing
import numpy as np
import pandas as pd
from ...core.algorithm.newey_west_ttest_1samp import NeweyWestTTest
from ..eva_utils import factor_to_quantile, factor_to_quantile_dependent_double_sort
from ..eva_utils import _compute_quantile_df, _compute_weighted_quantile_df
from ..eva_utils import ForwardReturns, QuantileReturns

StatisticResults = typing.NewType("StatisticResults", dict[str, pd.DataFrame])

class PortfolioSort:
    """
    Class to perform single and double portfolio sorts based on characteristics.
    """

    @staticmethod
    def single_sort(
        factor: pd.DataFrame,
        forward_returns: ForwardReturns,
        quantiles: int,
        value_weighted: bool = True,
        get_quantile_sorts: bool = False,
        market_cap: pd.DataFrame | None = None,
    ) -> typing.Union[QuantileReturns, pd.DataFrame]:
        """
        Perform single portfolio sort based on characteristic and create long-short portfolio.
        
        Args:
            factor: (Time x Stock) DataFrame of characteristic exposures
            forward_returns: period : (Time x Stock) DataFrame of returns
            market_cap: (Time x Stock) DataFrame of market capitalizations
            quantiles: number of quantiles
            value_weighted: Use market cap weighting (True) or equal weighting (False)
            get_quantile_sorts: Return portfolio assignments
        Returns:
            Portfolio returns and statistical results
        """
        # 1. DATA PREPARATION
        # assume factor, forward_return, market_cap are aligned DataFrames in our case
        # 2. QUANTILE CALCULATIONS
        quantile_sorts = factor_to_quantile(factor, quantiles)

        # Early exit if quantile assignments requested
        if get_quantile_sorts:
            return quantile_sorts
        
        # 3. RETURN CALCULATIONS
        # TODO: Add support for other weighting schemes
        # TODO: Add transaction costs
        if value_weighted:
            portfolio_returns = QuantileReturns ({
                period: _compute_weighted_quantile_df(quantile_sorts, period_returns, market_cap,quantiles=quantiles)
                for period, period_returns in forward_returns.items()
                })
        else:
            # equal weighted
            portfolio_returns = QuantileReturns ({
                period: _compute_quantile_df(quantile_sorts, period_returns, quantiles=quantiles)
                for period, period_returns in forward_returns.items()
                })

        # 4. HEDGE PORTFOLIO (High-Low)
        for period, _ in forward_returns.items():
            portfolio_returns[period]["H-L"] = (
                portfolio_returns[period][quantiles] - portfolio_returns[period][1]
            )
    
        return portfolio_returns

    @staticmethod
    def double_sort(
        factor1: pd.DataFrame,
        factor2: pd.DataFrame,
        forward_returns: ForwardReturns,
        quantiles: typing.Tuple[int, int] = (5, 5),
        dependent: bool = False,
        value_weighted: bool = True,
        get_quantile_sorts: bool = False,
        market_cap: pd.DataFrame | None = None,
    ) -> typing.Union[QuantileReturns, pd.DataFrame]:
        """
        Sort securities based on two characteristics.

        Args:
            factor1: (Time x Stock) DataFrame of characteristic exposures
            factor2: (Time x Stock) DataFrame of characteristic exposures
            forward_returns: period : (Time x Stock) DataFrame of returns
            market_cap: (Time x Stock) DataFrame of market capitalizations
            dependent: Whether to use dependent sorting (True) or independent sorting (False)
            quantiles: number of quantiles
            value_weighted: Use market cap weighting (True) or equal weighting (False)
            get_quantile_sorts: Return portfolio assignments
        Returns:
            Portfolio returns and statistical results
        """
        # Ensure that factor1 and factor2 have the same index and columns
        assert factor1.index.equals(factor2.index) and factor1.columns.equals(factor2.columns), \
            "factor1 and factor2 must have the same index and columns"

        # 1. DATA PREPARATION
        # assume factor1, factor2, forward_return, market_cap are aligned DataFrames in our case
        # 2. QUANTILE CALCULATIONS
        if dependent:
            # Dependent sorting (conditional double sorting)
            """
            Note from Professor SHI:

            Suppose we first sort the stocks based on X1, dividing all stocks into L1 groups. Then, within each of
            these L1 groups, we further sort the stocks based on X2, dividing the stocks into L2 groups. Again, a
            total of L1 × L2 groups

            The two sorting variables are NOT treated equally: the first sorting variable acts solely as a control
            variable, and the main interest is the relationship between the second sorting variable and asset
            returns. A factor should only be constructed based on the second sorting variable

            Lets assume factor1 is the control variable and factor2 is the main variable of interest.
            We will first sort the stocks based on factor1, then within each group, we will sort the stocks based on factor2.
            """

            combined_sorts =  factor_to_quantile_dependent_double_sort(factor1, factor2, quantiles)
        else:
            # independent sorting (unconditional double sorting)
            # Independent sorting will result some NONE quantile

            quantile_sorts_factor1 = factor_to_quantile(factor1, quantiles[0]).astype(int)
            quantile_sorts_factor2 = factor_to_quantile(factor2, quantiles[1]).astype(int)
            # quantile_sorts to string and add them to q1_q2 format
            combined_sorts = quantile_sorts_factor1.astype(str) + "_" + quantile_sorts_factor2.astype(str)

        # Initialize a dictionary to store the portfolio returns
        portfolio_returns = {}

        # 3. RETURN CALCULATIONS
        for period, period_returns in forward_returns.items():

            # Calculate returns for each combined quantile
            if value_weighted:
                period_portfolio_returns = _compute_weighted_quantile_df(
                    combined_sorts, period_returns, market_cap, reindex=False, quantiles= quantiles[0] * quantiles[1]
                )
            else:
                # equal weighted
                period_portfolio_returns = _compute_quantile_df(
                    combined_sorts, period_returns, reindex=False, quantiles= quantiles[0] * quantiles[1]
                )
            # Store the results
            portfolio_returns[period] = period_portfolio_returns

        # 4. HEDGE PORTFOLIO (High-High vs Low-Low)
        for period, _ in forward_returns.items():
            high_high = portfolio_returns[period].xs(f"{quantiles[0]}_{quantiles[1]}", axis=1)
            low_low = portfolio_returns[period].xs("1_1", axis=1)
            portfolio_returns[period]['HH-LL'] = high_high - low_low

        # Early exit if quantile assignments requested
        if get_quantile_sorts:
            return combined_sorts

        return QuantileReturns(portfolio_returns)

    @staticmethod
    def get_statistics(result: QuantileReturns, quantiles: int) -> StatisticResults:
        """
        Compute statistical results for single portfolio sort.

        TODO: 
            1. Add more statistics
            2. plot the results
        """        
        # T-Test for all periods
        # periods * (quantiles + H-L)
        t_stats = np.empty((len(result), quantiles + 1), dtype=float)
        p_values = np.empty((len(result), quantiles + 1), dtype=float)
        se_values = np.empty((len(result), quantiles + 1), dtype=float)
        mean_returns = np.empty((len(result), quantiles + 1), dtype=float)
                                
        for n, (_, period_returns) in enumerate(result.items()):
            # T-Test for all periods
            t_stats[n], p_values[n], se_values[n] = np.apply_along_axis(
                NeweyWestTTest.newey_west_ttest_1samp,
                axis=0,
                arr=period_returns,
                popmean=0,
                lags=6,
                nan_policy='omit'
            )
            # other statistics can be added here
            mean_returns[n] = np.nanmean(period_returns, axis=0)

        return StatisticResults({'t_stats': pd.DataFrame(t_stats, index=result.keys(), columns=period_returns.columns),
                'p_values': pd.DataFrame(p_values, index=result.keys(), columns=period_returns.columns),
                'se_values': pd.DataFrame(se_values, index=result.keys(), columns=period_returns.columns),
                'mean_returns': pd.DataFrame(mean_returns, index=result.keys(), columns=period_returns.columns)})