# %%
# %%
import pandas as pd
import numpy as np
from teachsim.library import start

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# %%
FILE_NAME = "feedback_analysis_withpre_post_survey_wide.dta"
SEED = start.SEED

PREDICTORS = [
    "das_stress",
    "das_depression",
    "das_anxiety",
    "dass_total",
    "neo_n",
    "neo_e",
    "neo_a",
    "neo_c",
    "neo_o",
    "tses_is",
    "score0",
    "score1",
    # "rsq_total",
    "elem",
]

df = pd.read_stata(
    start.RAW_DATA_DIR + FILE_NAME,
)
df["growth"] = df.score2 - df.score1

# %%

df = df[df.treat == "Coaching"]
len(df)


# %%
df = df.dropna(subset=PREDICTORS)
df = df.dropna(subset=["growth"])
len(df)
# %%

y = df.growth
X = df[PREDICTORS]

model = RandomForestRegressor(
    n_estimators=1000,
    criterion="squared_error",
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=5,
    bootstrap=True,
    random_state=SEED,
)

model.fit(X, y)
# plot_tree(model, feature_names=X.columns)

importances = model.feature_importances_

forest_importances = pd.Series(model.feature_importances_, index=X.columns)
forest_importances = pd.DataFrame(forest_importances).sort_values(by=0, ascending=False)
forest_importances
# %%
