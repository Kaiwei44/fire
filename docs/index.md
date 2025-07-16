---
title: Home
layout: home
nav_order: 1
---
# F.I.R.E. Factor Investment Research Engine

This project is the bundled opensource toolkit for book _Navigating the Factor Zoo：The Science of Quantitative Investing_. 

The Fire project serves as a development and evaluation toolkit for factor research and portfolio construction. It is designed specifically to be simple, easy to use, and built on top of popular Python libraries like pandas, numpy, and scikit-learn.

Fire focuses on three critical aspects of factor research and portfolio construction:

1. **Data Management**: Fire provides a user-friendly interface for downloading and managing financial data. By leveraging the pre-cleaned and processed data pipeline from the Fire Institute, you can focus more on research and modeling rather than data preparation.

2. **Construction (Calculation)**: Fire offers a variety of algorithms for factor construction. Additionally, it allows users to build their own factors using popular libraries such as pandas, numpy, and scikit-learn.

3. **Evaluation**: Factor evaluation is a complex and crucial step in research. Fire provides a comprehensive set of tools to assess factor performance, bridging the gap between academic and industry evaluation practices.

----

## Quick Start

## Installation

```bash
# We have not released the package to pypi yet, so you need to install from source!!!
pip install firefin

# Install from source for loacl testing!!!
## replace $ThisRepoURL with the actual repo url
git clone $ThisRepoURL 
## install from source
pip install -e .
```

## Usage

Download the data 
from [here](https://github.com/fire-institute/fire/releases/download/marketdata/AStockData.tar.gz)

run the command and download data put in correct path automatically.

```bash
# We have not released this repo yet, so you need download the data manually!!! See command below!!!
# Auto download data
firefin download
```

If you have already downloaded the data from [here](https://github.com/fire-institute/fire/releases/download/marketdata/AStockData.tar.gz), you can run the command to check the data and put the data in the correct path

```bash
# replace path_to_data.tar.gz with the actual path
firefin load path_to_data.tar.gz
```

## Start to code

```python
import firefin

# get data
data = firefin.fetch_data(["open", "close", "volume"])
open_price = data["open"]


def pv_corr(close, volume):
    # price volume correlation
    return close.rolling(20).corr(volume)


factor = pv_corr(data["close"], data["volume"])

# compute forward returns
fr = firefin.compute_forward_returns(open_price.shift(-1), [1, 5, 10])

# evaluate factor
mng = firefin.Evaluator(factor, fr)
mng.get_ic("pearson")
mng.get_quantile_returns(5)

# compute academia evaluation
from firefin.evaluation.academia.AcaEvaluatorModel import AcaEvaluatorModel

mng = AcaEvaluatorModel(factor=factor, forward_returns=fr, return_adj=data["return_adj"], n_jobs=24, verbose=10)
mng.run_all()
```