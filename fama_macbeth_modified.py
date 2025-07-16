import numpy as np
import pandas as pd
import statsmodels.api as sm
from firefin.core.algorithm.regression import *
from firefin.core.algorithm.regression import BatchRegressionResult
from firefin.core.algorithm.newey_west_ttest_1samp import NeweyWestTTest

class FamaMacBeth:
 def fama_macbeth_rolling(
    factor: pd.DataFrame | pd.Series | np.ndarray,
    return_adj: pd.DataFrame,
    window: int = 252,
    fit_intercept: bool = False,
    n_jobs: int = 4,
    verbose: int = 0,
 ) :
    """
    Fama–MacBeth (rolling‑window) implementation.

    Parameters
    ----------
    factor : pd.Series | pd.DataFrame | np.ndarray
        Factor realisations (T×K, T×K×N or K×T×N).  Supported inputs
        and how they are interpreted:

        * **pd.Series** – single factor; broadcast to every asset.
        * **pd.DataFrame** (*T×K*) – broadcast to the *N* assets.
        * **np.ndarray**
            – 2‑D (*T×K*) – broadcast to *N* assets;
            – 3‑D (*K×T×N*) or (*T×K×N*).

        In every case the data are reshaped to a common ndarray of shape
        (K,T,N) before being given to the first‑stage time‑series
        regressions.
    return_adj : pd.DataFrame (*T×N*)
        Adjusted (excess) returns of *N* assets.
    window : int, default 252
        Rolling window length used in both stages (β estimation & λ̄ averaging).
    fit_intercept : bool, default False
        Whether to include a constant in the **second‑stage** cross‑sectional
        regressions.  A constant is *always* included in the first stage and
        then discarded.
    n_jobs, verbose : see ``RollingRegressor``

    Returns
    -------
    lambda_bar : pd.DataFrame
        Each row is the average risk premia vector \bar{λ}_t for the window
        finishing at date *t* (index).  Columns are the *K* factor names (+
        "const" if a constant is requested).
    """

    N = return_adj.shape[1]  # number of assets
    T = return_adj.shape[0]  # number of time steps
    idx = return_adj.index   # timeline index

    # ---- convert `factor` to ndarray (K,T,N) ---- #
    if isinstance(factor, pd.Series):
        # broadcast single factor to each asset (N columns)
        factor_df = pd.concat([factor] * N, axis=1)
        factor_df.columns = return_adj.columns
        factor_arr = factor_df.to_numpy()[np.newaxis, ...]  # (1,T,N)
        factor_names = [factor.name or "factor0"]

    elif isinstance(factor, pd.DataFrame):
        factor_names = list(factor.columns)
        fac3d = np.repeat(factor.to_numpy()[:, :, np.newaxis], N, axis=2)  # (T,K,N)
        factor_arr = np.transpose(fac3d, (1, 0, 2))                       # (K,T,N)

    elif isinstance(factor, np.ndarray):
        if factor.ndim == 2:  # (T,K)
            if factor.shape[0] != T:
                raise ValueError("factor time dimension must equal return_adj rows")
            fac3d = np.repeat(factor[:, :, np.newaxis], N, axis=2)        # (T,K,N)
            factor_arr = np.transpose(fac3d, (1, 0, 2))                   # (K,T,N)
            factor_names = [f"factor{i}" for i in range(factor.shape[1])]

        elif factor.ndim == 3:
            # accept (K,T,N) or (T,K,N)
            if factor.shape[0] == T and factor.shape[2] == N:             # (T,K,N)
                factor_arr = np.transpose(factor, (1, 0, 2))              # (K,T,N)
            elif factor.shape[1] == T and factor.shape[2] == N:           # (K,T,N)
                factor_arr = factor
            else:
                raise ValueError("Unable to align 3‑D factor array with return_adj")
            factor_names = [f"factor{i}" for i in range(factor_arr.shape[0])]
        else:
            raise ValueError("factor ndarray must be 2‑D or 3‑D")
    else:
        raise ValueError("factor must be Series, DataFrame, or ndarray")

    K = factor_arr.shape[0]  # number of factors


    r = RollingRegressor(factor_arr, return_adj, None, fit_intercept=True).fit(
        window=window, n_jobs=n_jobs, verbose=verbose
    )

    # Collate β̂ into ndarray (K,T,N)
    if isinstance(r.beta, list):
        beta_arr = np.stack([df.to_numpy() for df in r.beta], axis=0)      # (K,T,N)
    else:  # single‑factor case
        beta_arr = r.beta.to_numpy()[np.newaxis, ...]                      # (1,T,N)

    # Re‑arrange to (N,K,T) so that the "fast" index is time
    beta_arr = np.transpose(beta_arr, (2, 0, 1))                          # (N,K,T)
    y_all = return_adj.to_numpy().T                                        # (N,T)



    lambda_bar = []
    lambda_original=[]
    columns = factor_names.copy()
    if fit_intercept:
        columns = ["const"] + columns

    for t in range(window - 1, T):
        B_t = beta_arr[:, :, t]                                            # (N,K)
        if fit_intercept:
            B_t = sm.add_constant(B_t, prepend=True)                       # (N,K+1)

        lambdas_window = np.empty((B_t.shape[1], window))                  # (K[+1],window)

        # τ runs over the rolling window
        for w, tau in enumerate(range(t - window + 1, t + 1)):
            y_tau = y_all[:, tau]                                          # (N,)
            reg = sm.OLS(y_tau, B_t, missing="drop").fit()
            lambdas_window[:, w] = np.asarray(reg.params)

        lambda_original.append(lambdas_window)
        lambda_bar.append(pd.Series(lambdas_window.mean(axis=1), index=columns))



    lambda_bar_df = pd.concat(lambda_bar, axis=1).T
    lambda_bar_df.index = idx[window - 1:]              # T*K DataFrame
    lambda_original_arr=np.array(lambda_original)       # T*K*window ndarray
    return BatchRegressionResult(lambda_bar_df), lambda_original_arr

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