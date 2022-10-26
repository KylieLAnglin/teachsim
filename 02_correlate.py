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


# %%
FILE_NAME = "feedback_analysis_withpre_post_survey_wide.dta"
SEED = 6
TEST_SIZE = 30
PREDICTORS = ["dass_total", "neo_n", "tses_is", "score0", "score1", "treat"]


# %%
df = pd.read_stata(
    start.RAW_DATA_DIR + FILE_NAME,
)
df["treat"] = df["treat"].map({"No Coaching": 0, "Coaching": 1})
df["growth"] = df.score2 - df.score1
df = df.dropna(subset=PREDICTORS)
df = df.dropna(subset=["score2"])
df = df.dropna(subset=["score1"])

y = df.growth
X = df[PREDICTORS]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)

# %%
df[PREDICTORS].corr()

# %%
