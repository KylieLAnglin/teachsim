# %%
import pandas as pd
import numpy as np
from . import start
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
# df = df[~df.coach2.isnull()]
df = df[~df.time.isnull()]

# %% Clean variables
df["highseshs_miss"] = np.where(df.highseshs.isnull(), 1, 0)
df["highseshs"] = np.where(df.highseshs.isnull(), 0, df.highseshs)


df = df[df.time == 0]
# %%
df = df.rename(columns={"treat": "treat_label"})
df["treat"] = df["treat_label"].map({"Coaching": 1, "No Coaching": 0})

covar = "CCS_GPA"
temp_df = df[~df[covar].isnull()]
print(temp_df[temp_df.treat == 0][covar].mean().round(2))
mod = smf.ols(
    formula=covar + " ~ treat + C(strata)",
    data=temp_df,
)
res = mod.fit(cov_type="cluster", cov_kwds={"groups": temp_df["strata"]})
print(res.summary())

# %%
