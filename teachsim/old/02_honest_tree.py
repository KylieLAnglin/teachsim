# %%
# %%
import pandas as pd
import numpy as np
from . import start


import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split


sns.set_style("white")
sns.set(font="Times")
# %%
FILE_NAME = "feedback_analysis_withpre_post_survey_wide.dta"
SEED = 6
SEED = 8
MIN_SAMPLES = 10
TEST_SIZE = 32  # 36

PREDICTORS_RAW = ["das_depression", "dass_total", "tses_is", "treat"]
PREDICTORS_RAW = ["score1", "treat", "tses_is", "neo_o", "dass_total"]

# %%
df = pd.read_stata(
    start.RAW_DATA_DIR + FILE_NAME,
)
df["treat"] = df["treat"].map({"No Coaching": 0, "Coaching": 1})
df["growth"] = df.score2 - df.score1

predictors_percentiles = []
for predictor in PREDICTORS_RAW:
    new_predictor = predictor + "_p"
    df[new_predictor] = df[predictor].rank(pct=True)
    predictors_percentiles.append(new_predictor)

predictors = predictors_percentiles + ["treat"]
# %%
df = df.dropna(subset=predictors)
df = df.dropna(subset=["growth"])
df = df.set_index("id")

# %%
df["first_growth"] = df.score1 - df.score0
df.first_growth.mean()

# %%
df["big_growth"] = df.score2 - df.score0
df.big_growth.mean()

# %%
plt.hist(df.growth, bins=10, color="lightgray")
plt.xlabel("Change in Feedback Quality")
plt.ylabel("Number of Participants")
plt.yticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])

# %%
plt.boxplot(df.growth)
# %%

y = df.growth
X = df[predictors]

X_train = X
y_train = y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)


model = DecisionTreeRegressor(
    min_samples_leaf=MIN_SAMPLES, random_state=SEED, max_depth=2
)

model.fit(X_train, y_train)

# %%
plt.figure(figsize=(10, 8), dpi=150)
plot_tree(model, feature_names=X.columns, node_ids=True, max_depth=5)
# plt.savefig(start.MAIN_DIR + "results/tree.pdf")

# %%
testing = X_test.merge(y_test, left_index=True, right_index=True)
testing["predicted_growth"] = model.predict(X=X_test)
# testing["leaf"] = testing.predicted_growth.round(3).map(
#     {-0.588: 1, 0.429: 2, 1.667: 3, 0.824: 4}
# )
# testing["leaf"] = testing.predicted_growth.round(3).map(
#     {-0.861: 1, 0.333: 2, 1.656: 3, 0.679: 4}
# )
# testing["leaf"] = testing.predicted_growth.round(3).map(
#     {-0.600: 1, 0.353: 2, 1.867: 3, 0.885: 4}
# )
testing["leaf"] = testing.predicted_growth.round(3).map(
    {-0.600: 1, 0.278: 2, 1.762: 3, 0.65: 4}
)
testing.leaf.value_counts()

testing[testing.leaf == 1].growth.mean()
testing[testing.leaf == 2].growth.mean()
testing[testing.leaf == 3].growth.mean()
testing[testing.leaf == 4].growth.mean()


# %%
print("Kathleen:", model.apply(X_train.loc[43].values.reshape(1, -1)))
print("Victoria:", model.apply(X_train.loc[105].values.reshape(1, -1)))
print("Liz:", model.apply(X_train.loc[41].values.reshape(1, -1)))
print("Alex:", model.apply(X_train.loc[57].values.reshape(1, -1)))


# %%
df["leaves"] = model.apply(X_train)
df["leaf"] = 0
df["leaf"] = np.where(df.leaves == 2, 1, df.leaf)
df["leaf"] = np.where(df.leaves == 3, 2, df.leaf)
df["leaf"] = np.where(df.leaves == 5, 3, df.leaf)
df["leaf"] = np.where(df.leaves == 6, 4, df.leaf)
df.leaf.value_counts()

# %%

df[df.leaf == 1].growth.mean()
df.loc[43].growth.mean()

for predictor in predictors_percentiles:
    print(predictor)
    print("Overall mean : " + str(df[df.leaf == 1][predictor].mean().round(2)))
    print("Kathleen" + str(df.loc[43][predictor].mean().round(2)))
    print(" ")
# %%
LEAF = 2
CASE = 105
df[df.leaf == LEAF].growth.mean()
df.loc[CASE].growth.mean()

for predictor in predictors_percentiles:
    print(predictor)
    print("Overall mean : " + str(df[df.leaf == LEAF][predictor].mean().round(2)))
    print("Victoria" + str(df.loc[CASE][predictor].mean().round(2)))
    print(" ")
# %%
LEAF = 3
CASE = 41
df[df.leaf == LEAF].growth.mean()
df.loc[CASE].growth.mean()

for predictor in predictors_percentiles:
    print(predictor)
    print("Overall mean : " + str(df[df.leaf == LEAF][predictor].mean().round(2)))
    print("Liz" + str(df.loc[CASE][predictor].mean().round(2)))
    print(" ")
# %%
LEAF = 4
CASE = 57
df[df.leaf == LEAF].growth.mean()
df.loc[CASE].growth.mean()

for predictor in predictors_percentiles:
    print(predictor)
    print("Overall mean : " + str(df[df.leaf == LEAF][predictor].mean().round(2)))
    print("Alex" + str(df.loc[CASE][predictor].mean().round(2)))
    print(" ")
# %%
