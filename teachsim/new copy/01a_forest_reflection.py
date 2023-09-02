# %%
# %%
import pandas as pd
import numpy as np
from . import start

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv(start.MAIN_DIR + "data/data_split.csv", index_col="id")
df = df[df.training == "train"]
df = df[df.treat == 0]
len(df)
# %%

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
]

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
    max_features=1,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=start.SEED,
)

model.fit(X, y)

importances = model.feature_importances_

forest_importances = pd.Series(model.feature_importances_, index=X.columns)
forest_importances = pd.DataFrame(forest_importances).sort_values(by=0, ascending=False)
forest_importances
# %%
