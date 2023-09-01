# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from econml.grf import CausalForest
from library import start

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# %%
FILE_NAME = "feedback_analysis_withpre_post_survey_wide.dta"

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
    "elem",
    "rsq_total",
]
df = pd.read_stata(
    start.RAW_DATA_DIR + FILE_NAME,
)
df["treat"] = np.where(df.treat == "Coaching", 1, 0)
df.treat.value_counts()
# %%

treatment = "treat"
outcome = "score2"
covariates = PREDICTORS
all_variables = [treatment] + [outcome] + covariates
df = df.dropna(axis=0, subset=all_variables)
len(df)
# %%

Y = df[outcome]
T = df[treatment]
X = df[covariates]


# %%

causal_forest = CausalForest(
    n_estimators=100,
    criterion="het",
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=5,
    max_samples=0.45,
    honest=False,
    inference=False,
    random_state=11,
)

causal_forest.fit(X=X, T=T, y=Y)

# %%
forest_importances = pd.Series(causal_forest.feature_importances_, index=PREDICTORS)
forest_importances = pd.DataFrame(forest_importances).sort_values(by=0, ascending=False)
forest_importances
# %%
df["itt_prediction"] = causal_forest.predict(X=X)
df.itt_prediction.hist()
# %%
plt.figure(figsize=(20, 10))
plot_tree(causal_forest[0], impurity=True, max_depth=3)
plt.show()
# %%
