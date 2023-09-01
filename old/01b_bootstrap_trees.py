# %%
# %%
import pandas as pd
import numpy as np
from library import start

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# %%
FILE_NAME = "feedback_analysis_withpre_post_survey_wide.dta"
SEED = 6
MIN_SAMPLES = 20

PREDICTORS_RAW = [
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
    # "tses_cm",
    # "tses_se",
    # "tses_total",
    "score0",
    "score1",
]

# %%
df = pd.read_stata(
    start.RAW_DATA_DIR + FILE_NAME,
)
df["treat"] = df["treat"].map({"No Coaching": 0, "Coaching": 1})
df["growth"] = df.score2 - df.score1

predictors_percentiles = []
for predictor in PREDICTORS_RAW:
    new_predictor = predictor + "_p"
    df[new_predictor] = df[predictor].rank(pct = True)
    predictors_percentiles.append(new_predictor)

predictors = predictors_percentiles + ["treat"]
# %%
df = df.dropna(subset=predictors)
df = df.dropna(subset=["growth"])
df = df.set_index("id")
# %%
np.random.seed(SEED)
seeds = np.random.randint(low = 1, high = 50000, size = 10)
seeds
# %%
bootstrap_samples = []
for n in seeds:
    boot = df.sample(len(df), replace=True, random_state=n)
    bootstrap_samples.append(boot)
# %%

fig, axs = plt.subplots(2, 5)

top_features = []
all_features = []
for boot, ax in zip(bootstrap_samples, fig.get_axes()):

    y = boot.growth
    X = boot[predictors]

    model = DecisionTreeRegressor(min_samples_leaf=MIN_SAMPLES)
    model.fit(X, y)
    plot_tree(model, feature_names=X.columns, ax=ax)

    # get the feature importances
    importances = model.feature_importances_

    # create a list of (feature name, feature importance) tuples and sort
    feature_importances = list(zip(predictors, importances))
    feature_importances.sort(key=lambda x: x[1], reverse=True)


    top_feature = feature_importances[0][0]
    top_features.append(top_feature)

    included_features = [feature[0] for feature in feature_importances if feature[1] > 0]
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
