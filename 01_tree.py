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
PREDICTORS = ["das_stress_p", "neo_n_p", "tses_is_p", "treat", "score0", "score1"]

# %%
df = pd.read_stata(
    start.RAW_DATA_DIR + FILE_NAME,
)
df["treat"] = df["treat"].map({"No Coaching": 0, "Coaching": 1})
df["growth"] = df.score2 - df.score1

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
# %%
len(X_train)
train_df = X_train.merge(y_train, left_index=True, right_index=True)
print(train_df.growth.mean())
# %%
model = DecisionTreeRegressor(min_samples_leaf=8)
model.fit(X_train, y_train)
plt.figure(figsize=(10, 8), dpi=150)
plot_tree(model, feature_names=X.columns)
# %%

plt.savefig(start.MAIN_DIR + "results/tree.pdf")


# %%
from sklearn.metrics import mean_squared_error

predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions)
test = X_test
test["predictions"] = predictions
test["growth"] = y_test
test["error"] = test.predictions - test.growth
# %%
test["predictions"] = predictions
mod = smf.ols(
    formula="growth ~ predictions",
    data=test,
)
res = mod.fit()
print(res.summary())

# %%
tses_cutoff_percentile = stats.percentileofscore(df.tses_is, 6.375)
0.81 / df.growth.std()
0.43 / df.growth.std()
score0_cutoff_percentile = stats.percentileofscore(df.score0, 4.5)
stress_cutoff_percentile = stats.percentileofscore(df.das_stress, 0.64)
stress_cutoff_percentile = stats.percentileofscore(df.das_stress, 0.786)

1.9 / df.growth.std()

rmse / df.growth.std()

# %%
