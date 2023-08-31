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
SEED = 6
MIN_SAMPLES = 15
TEST_SIZE = 10

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
    "tses_cm",
    "tses_se",
    "tses_total",
    # "treat",
    "score0",
    "score1",
    "elem",
]
df = pd.read_stata(
    start.RAW_DATA_DIR + FILE_NAME,
)
df["treat"] = np.where(df.treat == "Coaching", 1, 0)
df.treat.value_counts()
df["growth"] = df.score2 - df.score1
# %%

treatment = "treat"
# outcome = "growth"
outcome = "score2"
covariates = PREDICTORS
all_variables = [treatment] + [outcome] + covariates
df = df.dropna(axis=0, subset=all_variables)
len(df)
# %%
train, test = train_test_split(df, test_size=0.2)

Y = df[outcome]
T = df[treatment]
X = df[covariates]
W = None
X_test = test[covariates]


# Y = train[outcome]
# T = train[treatment]
# X = train[covariates]
# W = None

# %%

causal_forest = CausalForest(
    criterion="het",
    n_estimators=100,
    # min_samples_leaf=10,
    max_depth=None,
    max_samples=0.45,
    honest=True,
    inference=True,
    random_state=5225463,
)

# fit train data to causal forest model
causal_forest.fit(X=X, T=T, y=Y)
# estimate the CATE with the test set
# %%
plt.figure(figsize=(20, 10))
plot_tree(causal_forest[0], impurity=True, max_depth=3)
plt.show()
# %%
forest_importances = pd.Series(causal_forest.feature_importances_, index=PREDICTORS)
forest_importances = pd.DataFrame(forest_importances).sort_values(by=0, ascending=False)
forest_importances
# %%
# causal_forest.estimators_

# %%
# from econml.cate_interpreter import SingleTreePolicyInterpreter

# tree = SingleTreePolicyInterpreter(
#     include_model_uncertainty=False, max_depth=2, min_samples_leaf=10
# )
# tree.interpret(causal_forest, X=df[["neo_n", "neo_o", "tses_is"]])
# %%
causal_forest = CausalForest(
    criterion="het",
    n_estimators=1,
    min_samples_leaf=10,
    max_depth=2,
    max_samples=1.0,
    honest=True,
    inference=False,
    random_state=5225463,
    subforest_size=1,
)
X = df[["neo_n", "neo_o", "tses_is"]]
causal_forest.fit(X=X, T=T, y=Y)

plt.figure(figsize=(20, 10))
plot_tree(causal_forest[0], impurity=True)
plt.show()

# %%

Y = test[outcome]
T = test[treatment]
X = test[covariates]
W = None
