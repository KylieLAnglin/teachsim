# %%
# %%
import pandas as pd
import numpy as np
from library import start

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import statsmodels.formula.api as smf

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
    "treat",
    "score0",
    "score1",
    "elementary",
]
df = pd.read_stata(
    start.RAW_DATA_DIR + FILE_NAME,
)
# %%
df["treat"] = df["treat"].map({"No Coaching": 0, "Coaching": 1})
df["growth"] = df.score2 - df.score1
df["elementary"] = np.where(df.program == "elementary", 1, 0)

df["das_stress_p"] = df.das_stress.rank(pct=True)
df["neo_n_p"] = df.neo_n.rank(pct=True)
df["neo_e_p"] = df.neo_e.rank(pct=True)
df["tses_is_p"] = df.tses_is.rank(pct=True)

df = df.dropna(subset=PREDICTORS)
df = df.dropna(subset=["score2"])
df = df.dropna(subset=["score1"])

df = df.set_index("id")

# %%
y = df.growth
X = df[PREDICTORS]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)

len(X_train)
train_df = X_train.merge(y_train, left_index=True, right_index=True)
test_df = X_test.merge(y_test, left_index=True, right_index=True)

# %%
np.random.seed(SEED)
seeds = np.random.randint(low=1, high=50000, size=10)
seeds
# %%
bootstrap_samples = []
for n in seeds:
    boot = df.sample(len(df), replace=True, random_state=n)
    bootstrap_samples.append(boot)
# %%
predictors = PREDICTORS
fig, axs = plt.subplots(2, 5)

top_features = []
all_features = []
for boot, ax in zip(bootstrap_samples, fig.get_axes()):
    y = boot.growth
    X = boot[predictors]

    model = DecisionTreeRegressor(
        min_samples_leaf=MIN_SAMPLES,
    )
    model.fit(X, y)
    plot_tree(model, feature_names=X.columns, ax=ax)

    # get the feature importances
    importances = model.feature_importances_

    # create a list of (feature name, feature importance) tuples and sort
    feature_importances = list(zip(predictors, importances))
    feature_importances.sort(key=lambda x: x[1], reverse=True)

    top_feature = feature_importances[0][0]
    top_features.append(top_feature)

    included_features = [
        feature[0] for feature in feature_importances if feature[1] > 0
    ]
    all_features.append(included_features)

plt.savefig(start.MAIN_DIR + "results/bootstrap_trees.pdf")

# %% Count features
for predictor in predictors:
    count = 0
    for sublist in all_features:
        if predictor in sublist:
            count = count + 1
    print(predictor + " " + str(count))

# %%
test_df["neo_e_cutoff"] = np.where(test_df.neo_e > 3, 1, 0)
# Extroversion, score 1
covar = "neo_e_cutoff"
# covar = "score1"
temp_df = test_df[test_df.treat == 1]
mod = smf.ols(
    formula="growth ~  " + covar,
    data=temp_df,
)
res = mod.fit()
print(res.summary())

# %%
temp_df = test_df[test_df.treat == 0]
mod = smf.ols(
    formula="growth ~  " + covar,
    data=temp_df,
)
res = mod.fit()
print(res.summary())

# %%
