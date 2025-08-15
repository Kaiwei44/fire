import re
from typing import Optional, Tuple, Dict, Sequence, List
import numpy as np
import pandas as pd

def latex_table(
    df1: pd.DataFrame,
    df1_r2: Optional[pd.Series],
    df2: pd.DataFrame,
    df2_r2: Optional[pd.Series],
    customize_factor: Optional[pd.DataFrame] = None,
    customize_r2: Optional[pd.Series] = None,
    model_names: Tuple[str, str, Optional[str]] = ("CAPM", "4-Factor", None),
    custom_name: str = "Custom",
    coef_digits: int = 2,
    t_digits: int = 2,
    r2_digits: int = 3,
    excess_df: Optional[pd.DataFrame] = None,
    excess_name: str = "Excess Return",
    excess_cols: Tuple[str, str] = ("mean daily excess_ret", "std"),
) -> str:
    """
    LaTeX regression table：
      - factor block: Factor (coefficient/t in parentheses) + Adj R^2 (provided by an independent Series)
      - customized factor block: Factor (coefficient/t in parentheses) + Adj R^2 (provided by an independent Series)
      - Optional: Excess Return block (two columns: mean/std)
    """

    def _escape_tex(s: str) -> str:
        repl = {"&": r"\&","%": r"\%","$": r"\$","#": r"\#","_": r"\_",
                "{": r"\{","}": r"\}","~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}","\\": r"\textbackslash{}"}
        return "".join(repl.get(ch, ch) for ch in str(s))

    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = df.index.astype(str)
        if df.index.duplicated().any():
            df = df.groupby(level=0).agg(lambda col: col.dropna().iloc[0] if col.notna().any() else np.nan)
        return df

    def _normalize_s(s: Optional[pd.Series]) -> Optional[pd.Series]:
        if s is None:
            return None
        s = s.copy()
        s.index = s.index.astype(str)
        if s.index.duplicated().any():
            # 对重复 index 取第一个非空
            s = s.groupby(level=0).agg(lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan)
        return s

    col_pat = re.compile(r"^\(([^,]+),\s*(coeff|tvalue)\)$", re.I)

    def _parse_columns(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        """返回 {factor: {'coeff': colname, 'tvalue': colname}}，按原列顺序保序。"""
        factors: Dict[str, Dict[str, str]] = {}
        for c in df.columns:
            if isinstance(c, tuple) and len(c) == 2:
                fac, kind = c[0], str(c[1]).lower()
                if kind in ("coeff", "tvalue"):
                    factors.setdefault(str(fac), {})[kind] = c
            else:
                cs = str(c)
                m = col_pat.match(cs)
                if m:
                    fac, kind = m.group(1), m.group(2).lower()
                    factors.setdefault(fac, {})[kind] = c

        ordered = []
        for c in df.columns:
            key = None
            if isinstance(c, tuple) and len(c) == 2:
                key = str(c[0])
            else:
                m = col_pat.match(str(c))
                if m:
                    key = m.group(1)
            if key and key in factors and key not in ordered:
                ordered.append(key)
        return {k: factors[k] for k in ordered if ("coeff" in factors[k] or "tvalue" in factors[k])}

    def _fmt_num(x, digits):
        if pd.isna(x): return ""
        try: return f"{float(x):.{digits}f}"
        except Exception: return str(x)

    def _make_cell(v, t) -> str:
        v_str = _fmt_num(v, coef_digits)
        t_str = _fmt_num(t, t_digits)
        if v_str and t_str:
            return rf"\begin{{tabular}}[t]{{@{{}}r@{{}}}}{v_str}\\({t_str})\end{{tabular}}"
        elif v_str: return v_str
        elif t_str: return rf"\begin{{tabular}}[t]{{@{{}}r@{{}}}}\\({t_str})\end{{tabular}}"
        else: return ""

    # ---------- assemble blocks (left -> right) ----------
    blocks = []
    names: Sequence[str] = []


    if customize_factor is not None:
        _dfc = _normalize_df(customize_factor)
        _r2c = _normalize_s(customize_r2)
        fc = _parse_columns(_dfc)
        blocks.append(("model", _dfc, fc, _r2c))
        names.append(model_names[2] if len(model_names) > 2 and model_names[2] else custom_name)


    if excess_df is not None:
        _dfe = _normalize_df(excess_df)
        lcol, rcol = excess_cols
        for col in [lcol, rcol]:
            if col not in _dfe.columns:
                _dfe[col] = np.nan
        blocks.append(("excess", _dfe, (lcol, rcol), None))
        names.append(excess_name)


    _df1 = _normalize_df(df1); _r2_1 = _normalize_s(df1_r2); f1 = _parse_columns(_df1)
    blocks.append(("model", _df1, f1, _r2_1)); names.append(model_names[0] or "Model 1")


    _df2 = _normalize_df(df2); _r2_2 = _normalize_s(df2_r2); f2 = _parse_columns(_df2)
    blocks.append(("model", _df2, f2, _r2_2)); names.append(model_names[1] or "Model 2")


    all_index: list[str] = []
    for kind, df, _, r2s in blocks:
        for idx in df.index.tolist():
            if idx not in all_index:
                all_index.append(idx)

        if isinstance(r2s, pd.Series):
            for idx in r2s.index.tolist():
                if idx not in all_index:
                    all_index.append(idx)


    group_specs = []
    for nm, (kind, df, spec, r2s) in zip(names, blocks):
        if kind == "model":
            subcols = list(spec.keys()) + ["Adj $R^2$"]
        else:
            lcol, rcol = spec
            subcols = [lcol, rcol]
        group_specs.append((nm, subcols))


    col_specs = ["l"]
    first_line = ["Portfolio"]
    cmid_segments = []
    col_cursor = 1
    for (nm, subcols) in group_specs:
        span = len(subcols)
        first_line.append(rf"\multicolumn{{{span}}}{{c}}{{{_escape_tex(nm)}}}")
        start, end = col_cursor + 1, col_cursor + span
        cmid_segments.append(rf"\cmidrule(lr){{{start}-{end}}}")
        col_specs.extend(["c"] * span)
        col_cursor = end
    second_line = [""] + [_escape_tex(c) for _, subcols in group_specs for c in subcols]

    header = [
        r"\toprule",
        r"\begin{tabular}{" + " ".join(col_specs) + r"}",
        " & ".join(first_line) + r" \\",
        "".join(cmid_segments),
        " & ".join(second_line) + r" \\",
        r"\midrule",
    ]

    body_lines = []
    for idx in all_index:
        row_cells = [str(idx)]
        for (kind, df, spec, r2s) in blocks:
            s = df.loc[idx] if idx in df.index else pd.Series(dtype=float)
            if kind == "model":
                for f, sub in spec.items():
                    v = s.get(sub.get("coeff"), np.nan)
                    t = s.get(sub.get("tvalue"), np.nan)
                    row_cells.append(_make_cell(v, t))

                r2_val = np.nan
                if isinstance(r2s, pd.Series):
                    r2_val = r2s.get(str(idx), np.nan)
                row_cells.append(_fmt_num(r2_val, r2_digits))
            else:
                lcol, rcol = spec
                m = s.get(lcol, np.nan)
                sd = s.get(rcol, np.nan)
                row_cells.append(_fmt_num(m, 3))
                row_cells.append(_fmt_num(sd, 3))
        body_lines.append(" & ".join(row_cells) + r" \\")
    footer = [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(header + body_lines + footer)
