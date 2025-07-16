---
title: AcaEvaluatorModel
parent: Evaluate Factor
permalink: /evaluate/aca_evaluator_model/
nav_order: 6.2
---


# AcaEvaluatorModel

`AcaEvaluatorModel` is a class designed for evaluating the performance of a **single-factor model** using various asset pricing methodologies. It supports portfolio sorting, cross-sectional regression, information coefficient calculations, and anomaly tests.

---

## Class Initialization

```python
AcaEvaluatorModel(factor: pd.DataFrame, forward_returns: ForwardReturns, return_adj: pd.DataFrame)
````

**Parameters**

* `factor` *(pd.DataFrame)*: Factor exposure data (Time × Stock)
* `forward_returns` *(dict\[str, pd.DataFrame])*: Future returns mapped by holding periods (Time × Stock)
* `return_adj` *(pd.DataFrame)*: DataFrame of adjusted returns (Time × Stock)
---

## Methods

### `run_single_sort`

Perform single-factor portfolio sorting.

**Parameters**

* `quantiles` *(int)*: Number of quantile groups (e.g. 5 for quintiles)
* `value_weighted` *(bool)*: Use value-weighted portfolios if `True`; otherwise, equal-weighted
* `return_stats` *(bool)*: Whether to return statistics for H-L portfolio
* `market_cap` *(pd.DataFrame)*: Market cap data (required if `value_weighted=True`)
* `get_quantile_sorts` *(bool)*: Whether to return quantile labels for each stock

**Returns**

* If `return_stats=True`:
  `Tuple[QuantileReturns, dict]`
* Else:
  `QuantileReturns`

---

### `run_fama_macbeth`

Run two-stage Fama-MacBeth cross-sectional regression.

**Parameters**

* `window` *(int)*: Rolling window size for first-stage regression (default: 252)
* `return_stats` *(bool)*: Whether to return statistical summary

**Returns**

* If `return_stats=True`:
  `Tuple[RegressionResult, dict]`
* Else:
  `RegressionResult`

---

### `run_ic`

Compute Information Coefficients (IC) across time.

**Parameters**

* `method` *(str)*: Correlation type, one of `'pearson'`, `'spearman'`, or `'kendall'`

**Returns**

* `pd.DataFrame`: IC values by period

---

### `run_regression`

Run static or rolling regression of returns on factor exposures.

**Parameters**

* `rolling` *(bool)*: Whether to run rolling regression
* `window` *(int)*: Rolling window size (only used if `rolling=True`)
* `fit_intercept` *(bool)*: Include intercept term if `True`

**Returns**

* `BatchRegressionResult` or `dict` (if rolling)

---

### `run_anomaly_test`

Conduct anomaly tests by regressing returns on the factor.

**Parameters**

* `portfolio_returns` *(dict\[str, pd.DataFrame])*:  DataFrame of portfolio returns, with each column representing a distinct portfolio. (Quantile returns)
* `cov_type` *(Optional\[str])*: Type of covariance estimator (e.g., `'HAC'`, `'HC0'`, etc.)
* `cov_kwds` *(Optional\[dict])*: Additional keyword arguments for the covariance estimator
* `return_stats` *(bool)*: Whether to return summary statistics

**Returns**

* If `return_stats=True`:
  `Tuple[AnomalyTest, pd.DataFrame]`
* Else:
  `AnomalyTest`

---

### `run_all`

Run all available evaluation methods and return results in a dictionary.

**Returns**

* `dict`: Keys include:

  * `'single_sort_res'`, `'single_sort_stat'`
  * `'fama_macbeth'`
  * `'information_coefficient'`
  * `'regression'`
  * `'anomaly'`

---

## Notes

* `run_regression` uses either time-by-time OLS or rolling regression depending on the `rolling` flag.
* `run_all` is useful for executing a full evaluation pipeline for a single factor.
* Ensure `market_cap` is provided when performing value-weighted portfolio sorts.
* `return_adj` in `run_fama_macbeth` should be matched to the target return horizon.
