---
title: AcaEvaluator
permalink: /evaluate/aca_evaluator/
parent: Evaluate Factor
nav_order: 6.1
---

# AcaEvaluator

`AcaEvaluator` is a modular Python class designed to evaluate asset pricing factors using a comprehensive suite of financial econometrics tools. It supports portfolio sorts, cross-sectional and time-series regressions, model comparison tests, and robustness diagnostics.

---

## Key Features

| Method                         | Purpose                                                                            |
| ------------------------------ | ---------------------------------------------------------------------------------- |
| `run_single_sort()`            | Perform univariate portfolio sorting based on a factor                             |
| `run_double_sort()`            | Perform bivariate (conditional or independent) portfolio sorts                     |
| `run_fama_macbeth()`           | Run Fama-MacBeth two-pass cross-sectional regressions                              |
| `run_ic()`                     | Compute the Information Coefficient (IC) between factor values and forward returns |
| `run_msr_test()`               | Compute and compare Max Sharpe Ratios between two factor models                    |
| `run_regression()`             | Run time-series or rolling regression on test portfolios to obtain alpha and beta  |
| `get_grs_test()`               | Conduct the Gibbons-Ross-Shanken (GRS) test to check model pricing accuracy        |
| `get_hj_distance_test()`       | Compute Hansen–Jagannathan distance to assess pricing error                        |
| `compare_model_alphas()`       | Compare model alphas across multiple asset pricing models                          |
| `run_horse_race_regression()`  | Evaluate marginal explanatory power of factors (horse race regression)             |
| `get_spanning_test()`          | Check whether a new factor is spanned by an existing model                         |
| `run_subsample_analysis()`     | Perform robustness tests by splitting the sample                                   |
| `compute_vif()`                | Calculate Variance Inflation Factors to detect multicollinearity                   |

---

## Class Initialization

```python
AcaEvaluator(factor: pd.DataFrame, forward_returns: dict[str, pd.DataFrame])
```

**Parameters:**

* `factor`: A (Time × Stock) DataFrame of factor exposures
* `forward_returns`: A (Time × Stock) DataFrame of forward return

---

## Method Documentation

### `run_single_sort()`

Perform single-factor portfolio sorting.

**Parameters:**

* `quantiles` (int): Number of quantile groups (e.g., 5 for quintiles)
* `value_weighted` (bool): If True, portfolios are value-weighted; otherwise equal-weighted
* `return_stats` (bool): Whether to return statistics (mean, t-value, p-value) of high-minus-low (H-L) portfolios
* `market_cap` (pd.DataFrame): Required if `value_weighted` is True; same shape as `factor`
* `get_quantile_sorts` (bool): Return group labels of stocks by quantile

**Returns:**

* Quantile portfolio returns or tuple of (returns, statistics) if `return_stats=True`

---

### `run_double_sort()`

Perform double sorting based on two factors.

**Parameters:**

* `factor2` (pd.DataFrame): Second factor
* `quantiles` (tuple\[int, int]): Quantile group counts for each factor
* `dependent` (bool): Use nested sort if True
* `value_weighted` (bool): If True, portfolios are value-weighted; otherwise equal-weighted
* `market_cap` (pd.DataFrame): Required if `value_weighted` is True; same shape as `factor`
* `get_quantile_sorts` (bool): Return group labels of stocks by quantile

**Returns:**

* Portfolio return structure or dictionary of quantile groupings

---

### `run_fama_macbeth()`

Run two-pass Fama-MacBeth regression.

**Parameters:**

* `return_adj` (pd.DataFrame): Adjusted returns matrix
* `window` (int): First-stage rolling window (default: 252)
* `return_stats` (bool): Return t-statistics and significance

**Returns:**

* Regression result or (result, statistics) tuple

---

### `run_ic()`

Calculate Information Coefficient between factor and future returns.

**Parameters:**

* `method` (str): Correlation method ('pearson', 'spearman', or 'kendall')

**Returns:**

* `pd.DataFrame`: Time series of IC values

---

### `run_msr_test()`

Compare the maximum Sharpe ratios (MSRs) of two factor models using a z-test.

**Parameters:**

* `model_a_factors` (pd.DataFrame): Factor returns of Model A (Time × K)
* `model_b_factors` (pd.DataFrame): Factor returns of Model B (Time × K)
* `regularize_covariance` (bool): If True, regularize the covariance matrix.

**Returns:**

* Dictionary with keys:
  - `msr_a`: Maximum Sharpe ratio of Model A
  - `msr_b`: Maximum Sharpe ratio of Model B
  - `test_stat`: z-test statistic comparing MSRs
  - `p_value`: p-value of the test

---

### `run_regression()`

Run either standard or rolling time-series regression on test portfolios, based on a flag.

**Parameters:**

* `rolling` (bool): optionalWhether to perform rolling regression, by default False.
* `window` (int): optionalRolling window size (only used if rolling=True), by default 60.
* `fit_intercept` (bool): optionalWhether to include an intercept in the regression, by default True.

**Returns:**

* `BatchRegressionResult` (dict): Regression result object (static) or a dictionary of rolling results.
---

### `get_grs_test()`

Run GRS test for overall model explanatory power.

**Parameters:**

* `test_portfolios`: Time-series returns of test portfolios
* `plot`: Whether to generate visual output

**Returns:**

* Dictionary with keys: `grs_stat`, `p_value`, `alphas`, `t_stats`, `residual_cov`, `betas`

---

### `get_hj_distance_test()`

Compute HJ distance to assess pricing error.

**Parameters:**

* `test_portfolios`: Portfolio return matrix
* `plot`: Whether to visualize

**Returns:**

* Dictionary with HJ distance, t-stat, alpha, betas, residual\_cov

---

### `compare_model_alphas()`

Compare intercepts across different models.

**Parameters:**

* `models`: Dictionary of model name → factor return
* `test_portfolios`: Test portfolio returns
* `plot`: Whether to display comparison plot

**Returns:**

* Dictionary of results per model: alpha, t\_stat, mean\_abs\_alpha, mean\_abs\_t

---

### `run_horse_race_regression()`

Run horse race regression to assess marginal explanatory power.

**Parameters:**

* `candidate_factors`: Dict of factor name → exposure DataFrame
* `forward_return_key`: Key to select the return horizon
* `date`: If set, single-period regression; otherwise multi-period
* `plot`: Whether to visualize t-stats

**Returns:**

* Dictionary with `coefs`, `mean_coef`, `t_stat`, `p_value`

---

### `get_spanning_test()`

Test whether a new factor can be spanned by base factors.

**Parameters:**

* `new_factor`: Series of the new factor
* `base_model_factors`: Existing model factors (DataFrame)
* `plot`: Whether to show visualization

**Returns:**

* Dictionary with `r_squared`, `alpha`, `t_stat`, `p_value`, `beta`, `resid_std`

---

### `run_subsample_analysis()`

Run out-of-sample robustness checks across different time periods.

**Parameters:**

* `method`: "ic", "alpha", or "quantile\_returns"
* `split_dates`: List of split timestamps
* `forward_return_key`: Return name (if needed)
* `quantiles`: Group count for sorting (if used)
* `plot`: Whether to visualize comparison

**Returns:**

* Dictionary with per-sample evaluation results

---

### `compute_vif()`

Detect multicollinearity using variance inflation factors (VIF).

**Parameters:**

* `factors`: Factor exposure matrix (T × K)
* `plot`: Show bar plot of VIFs

**Returns:**

* Dictionary with `vif` (Series), `max_vif`, `mean_vif`
