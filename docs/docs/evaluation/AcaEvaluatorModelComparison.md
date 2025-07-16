---
title: AcaEvaluatorModelComparison
parent: Evaluate Factor
permalink: /evaluate/aca_evaluator_model_comparison/
nav_order: 6.3
---

# AcaEvaluatorModelComparison

`AcaEvaluatorModelComparison` is a class designed for evaluating **multi-factor models**, supporting methods such as double portfolio sorting and maximum Sharpe ratio (MSR) comparison between two models.

---

## Class Initialization

```python
AcaEvaluatorModelComparison(factor1: pd.DataFrame, factor2: pd.DataFrame, forward_returns: ForwardReturns)
````

**Parameters**

* `factor1` *(pd.DataFrame)*: First factor exposure matrix (Time × Stock)
* `factor2` *(pd.DataFrame)*: Second factor exposure matrix (Time × Stock)
* `forward_returns` *(dict\[str, pd.DataFrame])*: Future returns by holding period (Time × Stock)

---

## Methods

### `run_double_sort`

Perform double-sort portfolio sorting based on two factors.

**Parameters**

* `quantiles` *(Tuple\[int, int])*: Number of quantiles for each factor (e.g., (5, 5))
* `dependent` *(bool)*: Whether to apply dependent (conditional) sorting
* `value_weighted` *(bool)*: Use value-weighted returns if `True`; otherwise equal-weighted
* `market_cap` *(pd.DataFrame)*: Market cap data, required if `value_weighted=True`
* `get_quantile_sorts` *(bool)*: Whether to return the portfolio labels for each stock

**Returns**

* `QuantileReturns` or `dict[str, pd.DataFrame]` (if `get_quantile_sorts=True`)

---

### `run_msr_test`

Compare the Maximum Sharpe Ratios (MSR) between two models using a statistical test.

**Parameters**

* `regularize` *(bool)*: Whether to apply shrinkage regularization to the covariance matrix

**Returns**

* `dict`:

  * `'msr_a'`: Maximum Sharpe Ratio for model A
  * `'msr_b'`: Maximum Sharpe Ratio for model B
  * `'test_stat'`: Z-statistic of the MSR test
  * `'p_value'`: Corresponding two-sided p-value

---

### `run_all`

Run all available evaluation methods in the class.

**Parameters**

* `market_cap` *(pd.DataFrame)*: Required if `value_weighted=True` in `run_double_sort`

**Returns**

* `dict`:

  * `'double_sort'`: Result of double-sort sorting
  * `'msr_test'`: Result of MSR comparison between the two factor models

---

## Notes

* `run_double_sort` can support both independent and nested (conditional) sorting based on two factors.
* `run_msr_test` is based on a Z-test for comparing Sharpe Ratios under multivariate settings.
* `run_all` is a quick way to benchmark model performance using all implemented tools.