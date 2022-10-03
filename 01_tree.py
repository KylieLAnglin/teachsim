# %%
# %%
from re import sub
import pandas as pd
import numpy as np
from library import start
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# %%
FILE_NAME = "feedback_analysis_withpre_post_survey_wide.dta"

# %%
df = pd.read_stata(
    start.RAW_DATA_DIR + FILE_NAME,
)

df.head()

df["treat"] = df["treat"].map({"No Coaching": 0, "Coaching": 1})

# %%
predictors = [
    "das_stress",
    "neo_n",
    # "crtse_total",
    # "grit_total",
    # "imts_total",
    # "tmas_total",
    "tses_is",
    # "pck2",
    "treat",
]

# predictors = [
#     "score1",
#     "treat",
# ]


# predictors = [
#     "das_stress",
#     "das_anxiety",
#     "das_depression",
#     "neo_n",
#     "neo_e",
#     "neo_a",
#     "neo_c",
#     "rsq_total",
#     "tses_se",
#     "tses_is",
#     "tses_cm",
#     "crtse_total",
#     # "ccs_gpa",
#     "grit_total",
#     "imts_total",
#     "tmas_total",
#     "tses_total",
#     "pck2",
#     "treat",
# ]
df = df.dropna(subset=predictors)
df = df.dropna(subset=["score2"])
df = df.dropna(subset=["score1"])

df["growth"] = df.score2 - df.score1
# %%
y = df.growth
X = df[predictors]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=44
)
len(y_train)
len(y_test)

X_train = df[predictors]
y_train = df.growth
# %%
model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# %%

plt.figure(figsize=(10, 8), dpi=150)
plot_tree(model, feature_names=X.columns)
# %%

plt.savefig(start.MAIN_DIR + "results/tree.pdf")
