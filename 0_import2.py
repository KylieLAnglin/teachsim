# %%
# %%
import pandas as pd
import numpy as np
from library import start
import statsmodels.formula.api as smf

# %%
FILE_NAME = "feedback_analysis_withpre_post_survey_wide.dta"
# FILE_NAME = "feedback_analysis_withpre_post_survey_wide.dta"
OUTCOMES = [
    "score",
    "resp_misunderstanding",
    "perf_total",
    "tot_ntxt_total",
    "tot_txt_total",
    "tot_desc_total",
    "tot_rest_total",
]

# %%
df = pd.read_stata(
    start.RAW_DATA_DIR + FILE_NAME,
)

df.head()

df["treat"] = df["treat"].map({"No Coaching": 0, "Coaching": 1})
# %% Control mean
for variable in [
    "score2",
    "resp_misu2",
    "perf_total2",
    "tot_ntxt_total2",
    "tot_txt_total2",
    "tot_desc_total2",
    "tot_rest_total2",
]:
    print(df[df.treat == 0][variable].mean())


# %%
centered0 = [
    "score0_bs_cent",
    "resp_misu0_bs_cent",
    "perf_total0_bs_cent",
    "tot_ntxt_total0_bs_cent",
    "tot_txt_total0_bs_cent",
    "tot_desc_total0_bs_cent",
    "tot_rest_total0_bs_cent",
]
centered1 = [
    "score1_bs_cent",
    "resp_misu1_bs_cent",
    "perf_total1_bs_cent",
    "tot_ntxt_total1_bs_cent",
    "tot_txt_total1_bs_cent",
    "tot_desc_total1_bs_cent",
    "tot_rest_total1_bs_cent",
]

miss0 = [
    "score0_bs_cent_miss",
    "resp_misu0_bs_cent_miss",
    "perf_total0_bs_cent_miss",
    "tot_ntxt_total0_bs_cent_miss",
    "tot_txt_total0_bs_cent_miss",
    "tot_desc_total0_bs_cent_miss",
    "tot_rest_total0_bs_cent_miss",
]
miss1 = [
    "score1_bs_cent_miss",
    "resp_misu1_bs_cent_miss",
    "perf_total1_bs_cent_miss",
    "tot_ntxt_total1_bs_cent_miss",
    "tot_txt_total1_bs_cent_miss",
    "tot_desc_total1_bs_cent_miss",
    "tot_rest_total1_bs_cent_miss",
]

covars = ["highseshs", "highseshs_miss"] + centered0 + centered1 + miss0 + miss1

# %%
covar_formula = ""
for covar in covars:
    covar_formula = covar_formula + " + " + covar

# %%
# %% Model 1
df = df[~df.treat.isnull()]

mod = smf.ols(
    formula="score2 ~ treat + C(strata)",
    data=df,
)
res = mod.fit(
    cov_type="cluster",
    cov_kwds={"groups": df.dropna(subset=["treat", "strata", "score2"]).strata},
)
print(res.summary())

# %%
mod = smf.ols(
    formula="score2 ~ treat + C(strata)" + covar_formula,
    data=df,
)
res = mod.fit(
    cov_type="cluster",
    cov_kwds={"groups": df.dropna(subset=["treat", "strata", "score2"]).strata},
)
print(res.summary())
# %%
mod = smf.ols(
    formula="score2 ~ treat + C(strata)" + covar_formula + "+ C(interactor2)",
    data=df,
)
res = mod.fit(
    cov_type="cluster",
    cov_kwds={"groups": df.dropna(subset=["treat", "strata", "score2"]).strata},
)
print(res.summary())

# %%

from sklearn.model_selection import train_test_split

X = []
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=44
)
