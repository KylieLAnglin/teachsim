# %%
# %%
from re import sub
import pandas as pd
import numpy as np
from library import start
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from scipy import stats
from sklearn.metrics import mean_squared_error


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
    "tses_cm",
    "tses_se",
    "tses_total",
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

y = df.growth
X = df[predictors]

X_train = X
y_train = y

model = DecisionTreeRegressor(min_samples_leaf=MIN_SAMPLES)

model.fit(X_train, y_train)

# %%
plt.figure(figsize=(10, 8), dpi=150)
plot_tree(model, feature_names=X.columns, node_ids=True, max_depth = 5)
plt.savefig(start.MAIN_DIR + "results/tree.pdf")

# %%
print("Kathleen:", model.apply(X_train.loc[43].values.reshape(1, -1)))
print("Victoria:", model.apply(X_train.loc[105].values.reshape(1, -1)))
print("Liz:", model.apply(X_train.loc[41].values.reshape(1, -1)))
print("Alex:", model.apply(X_train.loc[57].values.reshape(1, -1)))

# %%