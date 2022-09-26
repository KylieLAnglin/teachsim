# %%
import pandas as pd
import numpy as np
from library import start
import statsmodels.formula.api as smf

# %%
FILE_NAME = "feedback_analysis_withpre_post_survey_coach.dta"
# FILE_NAME = "feedback_analysis_withpre_post_survey_wide.dta"
df = pd.read_stata(
    start.RAW_DATA_DIR + FILE_NAME,
)

df.head()

# %% What time is the outcome, what time is the pre-test?
df_pre0 = df[df.time == 0]
df_pre1 = df[df.time == 1]
df = df[df.time == 2]


# %% Sample selection
df = df[df.coach2 != ""]

# %% What is the treatment?
df.treat.value_counts()
df.treat1.value_counts()
# %% What is the outcome?

df.score.mean()

df["highseshs_miss"] = np.where(df.highseshs.isnull(), 1, 0)
# %%
mod = smf.ols(
    formula="score ~ treat + C(strata) + C(interactor) + highseshs + highseshs_miss",
    data=df,
)
res = mod.fit()
print(res.summary())

# %%
df[df.treat == "No Coaching"].score.mean()

# TODO: Figure out how they controlled for baseline 0 and 1

# %%
