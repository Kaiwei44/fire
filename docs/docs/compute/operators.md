---
title: Operators
nav_order: 5.1
parent: Construct Factor
permalink: /construct/operators
---

## Basic Arithmetic Operators

You can simply use basic arithmetic operator to construct the factor element-wise.

```python
a = pd.DataFrame(np.random.randn(10, 5))
b = pd.DataFrame(np.random.randn(10, 5))
c = a + b
```

## Rolling Window Functions

You can simply use rolling window function to perform time-series calculation without reshape data by stock name.

```python
from firefin.compute.window import ts_corr

a = pd.DataFrame(np.random.randn(10, 5))
b = pd.DataFrame(np.random.randn(10, 5))
c = ts_corr(a, b, window=20)
```