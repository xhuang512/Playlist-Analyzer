from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


def mean_ci(x, confidence=0.95) -> dict[str, float]:
    values = pd.Series(x).dropna().astype(float)
    n = len(values)
    mean = values.mean()
    if n < 2:
        return {"n": n, "mean": mean, "ci_low": np.nan, "ci_high": np.nan}
    se = stats.sem(values)
    margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return {"n": n, "mean": mean, "ci_low": mean - margin, "ci_high": mean + margin}


def _cohens_d(x, y) -> float:
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)
    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled_sd


def inference_summary(df: pd.DataFrame) -> dict:
    latin = df.loc[df["language_group"].eq("Latin-script title"), "popularity"]
    chinese = df.loc[df["language_group"].eq("Chinese-script title"), "popularity"]
    if len(latin) >= 2 and len(chinese) >= 2:
        welch = stats.ttest_ind(chinese, latin, equal_var=False)
        mann = stats.mannwhitneyu(chinese, latin, alternative="two-sided")
        group_comparison = {
            "available": True,
            "label": "Chinese-script vs Latin-script popularity",
            "t": float(welch.statistic),
            "p": float(welch.pvalue),
            "cohens_d": float(_cohens_d(chinese, latin)),
            "mann_p": float(mann.pvalue),
        }
    else:
        group_comparison = {
            "available": False,
            "label": "Chinese-script vs Latin-script popularity",
            "message": "This comparison needs at least two Chinese-script and two Latin-script tracks.",
        }

    formula = (
        "popularity ~ release_year + duration_min + artist_popularity "
        "+ log_artist_followers + explicit_int + C(language_model_group)"
    )
    model_df = df.dropna(
        subset=[
            "popularity",
            "release_year",
            "duration_min",
            "artist_popularity",
            "log_artist_followers",
            "explicit_int",
            "language_model_group",
        ]
    )
    ols_available = len(model_df) >= 20 and model_df["language_model_group"].nunique() >= 1
    ols_payload = {
        "available": False,
        "message": "The playlist is too small for the regression summary.",
    }

    if ols_available:
        ols_model = smf.ols(formula, data=model_df).fit()
        robust = ols_model.get_robustcov_results(cov_type="HC3")
        _, bp_lm_p, _, _ = het_breuschpagan(ols_model.resid, ols_model.model.exog)

        exog = pd.DataFrame(ols_model.model.exog, columns=ols_model.model.exog_names)
        vif = pd.DataFrame(
            {
                "variable": exog.columns,
                "vif": [variance_inflation_factor(exog.values, i) for i in range(exog.shape[1])],
            }
        ).sort_values("vif", ascending=False)

        significant_terms = []
        params = pd.Series(robust.params, index=ols_model.params.index)
        pvalues = pd.Series(robust.pvalues, index=ols_model.params.index)
        for term, pvalue in pvalues.items():
            if term != "Intercept" and pvalue < 0.05:
                significant_terms.append(
                    {
                        "term": term,
                        "coef": float(params[term]),
                        "pvalue": float(pvalue),
                    }
                )

        ols_payload = {
            "available": True,
            "r2": float(ols_model.rsquared),
            "adj_r2": float(ols_model.rsquared_adj),
            "significant_terms": significant_terms,
            "bp_pvalue": float(bp_lm_p),
            "top_vif": vif.head(5).round(2).to_dict(orient="records"),
        }

    if df["language_group"].nunique() == 0:
        language_ci = []
    else:
        language_ci = (
            df.groupby("language_group")["popularity"]
            .apply(lambda x: pd.Series(mean_ci(x)))
            .unstack()
            .sort_values("n", ascending=False)
            .round(2)
            .reset_index()
            .to_dict(orient="records")
        )

    return {
        "language_ci": language_ci,
        "group_comparison": group_comparison,
        "ols": ols_payload,
    }
