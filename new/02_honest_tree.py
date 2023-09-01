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
SEED = 8

PREDICTORS = ["treat", "neo_e", "tses_is"]
# %%
df = pd.read_stata(
    start.RAW_DATA_DIR + FILE_NAME,
)
df["growth"] = df.score2 - df.score1
df["treat"] = df["treat"].map({"No Coaching": 0, "Coaching": 1})


# %%
df = df.dropna(subset=PREDICTORS)
df = df.dropna(subset=["growth"])
df = df.set_index("id")
len(df)
# %%

# %%
plt.hist(df.growth, bins=10, color="lightgray")
plt.xlabel("Change in Feedback Quality")
plt.ylabel("Number of Participants")
plt.yticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])

# %%
plt.boxplot(df.growth)
# %%

y = df.growth
X = df[PREDICTORS]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=round(len(df) / 3), random_state=SEED
)

model = DecisionTreeRegressor(min_samples_leaf=5, random_state=SEED, max_depth=2)
model.fit(X_train, y_train)

# %%
plt.figure(figsize=(10, 8), dpi=150)
plot_tree(model, feature_names=X.columns, node_ids=True, max_depth=5)
# plt.savefig(start.MAIN_DIR + "results/tree.pdf")

# %%
testing = X_test.merge(y_test, left_index=True, right_index=True)
testing["predicted_growth"] = model.predict(X=X_test)
testing["leaf"] = testing.predicted_growth.round(3).map(
    {-0.25: 1, 0.368: 2, 0.786: 3, 1.625: 4}
)
testing.leaf.value_counts()

testing[testing.leaf == 1].growth.mean()
testing[testing.leaf == 2].growth.mean()
testing[testing.leaf == 3].growth.mean()
testing[testing.leaf == 4].growth.mean()

# %%
df["predicted_growth"] = model.predict(df[PREDICTORS])
print("Kathleen:", model.apply(df[PREDICTORS].loc[43].values.reshape(1, -1)))
print("Victoria:", model.apply(df[PREDICTORS].loc[105].values.reshape(1, -1)))
print("Alex:", model.apply(df[PREDICTORS].loc[57].values.reshape(1, -1)))
print("Liz:", model.apply(df[PREDICTORS].loc[41].values.reshape(1, -1)))

# %%
