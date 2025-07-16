import pandas as pd
from .portfolio_sort import PortfolioSort
from .MSR_Test import MSRTest
from ..eva_utils import ForwardReturns

class AcaEvaluatorModelComparison:
    def __init__(self, factor1: pd.DataFrame, factor2: pd.DataFrame, forward_returns: ForwardReturns):
        """
        Parameters:
            factor1 & factor2: pd.DataFrame
                Factor exposure data (Time × Stock)
            forward_returns: dict[str, pd.DataFrame]
                A dictionary where each key is a holding period, and the value is a DataFrame of future returns (Time × Stock)
        """

        self.factor1 = factor1
        self.factor2 = factor2
        self.forward_returns = forward_returns

    def run_double_sort(self,
                        quantiles: tuple = (5, 5),
                        dependent: bool = False,
                        value_weighted: bool = True,
                        market_cap: pd.DataFrame = None,
                        get_quantile_sorts: bool = False):
        """
        Perform double-factor sorting by jointly grouping assets based on factor1 and factor2, and calculate returns.

        Parameters:
            quantiles: Tuple[int, int]
                Number of quantile groups for the primary and secondary factors (e.g., (5, 5))
            dependent: bool
                Whether to use conditional (nested) sorting
            value_weighted: bool
                Whether to use value-weighted portfolios
            market_cap: pd.DataFrame
                Market capitalization data, same dimensions as the factors; required if value_weighted is True
            get_quantile_sorts: bool
                Whether to return portfolio labels (i.e., the group each stock belongs to)

        Returns:
            QuantileReturns or dict[str, pd.DataFrame] (if get_quantile_sorts is True)
        """

        if value_weighted and market_cap is None:
            raise ValueError("You must provide market_cap when value_weighted=True.")

        return PortfolioSort.double_sort(
            factor1=self.factor1,
            factor2=self.factor2,
            forward_returns=self.forward_returns,
            market_cap=market_cap,
            quantiles=quantiles,
            dependent=dependent,
            value_weighted=value_weighted,
            get_quantile_sorts=get_quantile_sorts
        )

    def run_msr_test(self, regularize=True):
        """
        Compare the Maximum Sharpe Ratios of two factor models using a Z-test.
        Args:
            regularize_covariance (bool): If True, regularize the covariance matrix.
        Returns:
            dict: {
                'msr_a': float,  # MSR of model A
                'msr_b': float,  # MSR of model B
                'test_stat': float,  # Z-statistic
                'p_value': float  # two-sided p-value
            }
        """
        return MSRTest.run_msr_comparison(model_a=self.factor1, model_b=self.factor2, regularize_covariance=True)
        
    def run_all(self, market_cap: pd.DataFrame = None) -> dict:
        """
        Run all evaluation methods and return results as a dictionary.
    
        Parameters:
            market_cap: pd.DataFrame（Required for value-weighted double sort）
    
        Returns:
            dict:
                {'double_sort': result of double sort,
                'msr_test': result of MSR test}
        """
        results = {}
    
        try:
            results['double_sort'] = self.run_double_sort(
                quantiles=(5, 5),
                value_weighted=True,
                market_cap=market_cap,
                get_quantile_sorts=False
            )
        except Exception as e:
            results['double_sort'] = f"Error: {e}"
    
        try:
            results['msr_test'] = self.run_msr_test(
                regularize=True
            )
        except Exception as e:
            results['msr_test'] = f"Error: {e}"
    
        return results