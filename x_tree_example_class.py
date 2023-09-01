# %%
# %%
from re import sub
import pandas as pd
import numpy as np
from . import start
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score

# %%
FILE_NAME = "feedback_analysis_withpre_post_survey.dta"
SEED = 6
TEST_SIZE = 30
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
]

# %%
df = pd.read_stata(
    start.RAW_DATA_DIR + FILE_NAME,
)

# %%
df["treat"] = df["treat"].map({"No Coaching": 0, "Coaching": 1})


df = df.dropna(subset=PREDICTORS)
df = df.dropna(subset="score")


df = df.set_index("id")
df["score_cat"] = np.where(df.score >= 5.5, 1, 0)
# %%
y = df.score_cat
X = df[PREDICTORS]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)
# %% Train
model = tree.DecisionTreeClassifier(min_samples_leaf=5)
model.fit(X_train, y_train)
plt.figure(figsize=(10, 8), dpi=150)
plot_tree(model, feature_names=X.columns)

# Training accuracy
predictions = model.predict(X_train)
accuracy_score(y_train, predictions)
# recall_score(y_train, predictions)
# precision_score(y_train, predictions)

# %% Validation accuracy
predictions = model.predict(X_test)
accuracy_score(y_test, predictions)


# %%
df["candidate_id"] = df.index
df = df.drop_duplicates(subset="candidate_id")
y = df.score_cat
X = df[PREDICTORS + ["candidate_id"]]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=X.candidate_id
)
# %% Train
model = tree.DecisionTreeClassifier(min_samples_leaf=5)
model.fit(X_train, y_train)
plt.figure(figsize=(10, 8), dpi=150)
plot_tree(model, feature_names=X.columns)

# %% Don't splot

predictions = model.predict(X_train)
accuracy_score(y_train, predictions)
# recall_score(y_train, predictions)
# precision_score(y_train, predictions)

# %% RMSE splot
predictions = model.predict(X_test)
accuracy_score(y_test, predictions)

# %%
