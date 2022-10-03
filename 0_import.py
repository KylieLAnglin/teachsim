# %%
import pandas as pd
import numpy as np
from library import start
import statsmodels.formula.api as smf

# %%
FILE_NAME = "feedback_analysis_withpre_post_survey_coach.dta"
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

# %% Sample selection
df = df[~df.coach2.isnull()]
df = df[~df.time.isnull()]

# %% Clean variables
df["highseshs_miss"] = np.where(df.highseshs.isnull(), 1, 0)
df["highseshs"] = np.where(df.highseshs.isnull(), 0, df.highseshs)
# %% What time is the outcome, what time is the pre-test?

df_pre0 = df[df.time == 0].set_index("id")
df_pre0 = df_pre0[OUTCOMES]
df_pre0 = df_pre0.add_suffix("0")


df_pre1 = df[df.time == 1].set_index("id")
df_pre1 = df_pre1[OUTCOMES]
df_pre1 = df_pre1.add_suffix("1")

df = df[df.time == 2].set_index("id")
df = df.merge(df_pre0, left_index=True, right_index=True)
df = df.merge(df_pre1, left_index=True, right_index=True)

# %%
# Centered baseline scores
for outcome in OUTCOMES:
    outcome = outcome + "0"
    outcome_treated_mean = df[df.treat == "Coaching"][outcome].mean()
    new_outcome = outcome + "_centered"
    df[new_outcome] = df[outcome] - outcome_treated_mean

    df[new_outcome] = np.where(~df[new_outcome].isnull(), df[new_outcome], 0)
    df[new_outcome + "_missing"] = np.where(df[new_outcome].isnull(), 1, 0)
# Centered time 1 scores
for outcome in OUTCOMES:
    outcome = outcome + "1"
    outcome_treated_mean = df[df.treat == "Coaching"][outcome].mean()
    new_outcome = outcome + "_centered"
    df[new_outcome] = df[outcome] - outcome_treated_mean

    df[new_outcome] = np.where(~df[new_outcome].isnull(), df[new_outcome], 0)
    df[new_outcome + "_missing"] = np.where(df[new_outcome].isnull(), 1, 0)

# Missing indicators
# %%
covars = []
for outcome in OUTCOMES:
    covars.append(outcome + "0_centered")
    covars.append(outcome + "0_centered_missing")
    covars.append(outcome + "1_centered")
    covars.append(outcome + "1_centered_missing")

covar_formula = ""
for covar in covars:
    covar_formula = covar_formula + " + " + covar

# %%
df["treat2"] = df["treat"].map({"Coaching": 1, "No Coaching": 0})
print(df[df.treat2 == 0]["score"].mean())

# %% Model 1
mod = smf.ols(
    formula="score ~ treat2 + C(strata)",
    data=df,
)
res = mod.fit(cov_type="cluster", cov_kwds={"groups": df["strata"]})
print(res.summary())

# %% Model 2
mod = smf.ols(
    formula="score ~ treat2 + C(strata) + highseshs + highseshs_miss" + covar_formula,
    data=df,
)
res = mod.fit(cov_type="cluster", cov_kwds={"groups": df["strata"]})
print(res.summary())


# %% Model 3
mod = smf.ols(
    formula="score ~ treat + C(strata) + C(interactor) + highseshs + highseshs_miss "
    + covar_formula,
    data=df,
)
res = mod.fit(cov_type="cluster", cov_kwds={"groups": df["strata"]})
print(res.summary())


# %%
df[df.treat == "No Coaching"].score.mean()

# TODO: Figure out how they controlled for baseline 0 and 1

# %%
