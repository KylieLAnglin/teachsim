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


PREDICTORS = ["treat", "neo_e", "tses_is"]
df = pd.read_csv(start.MAIN_DIR + "data/data_split.csv", index_col="id")
len(df)

# %%
df = df.dropna(subset=PREDICTORS)
df = df.dropna(subset=["growth"])
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

train = df[df.training == "train"]
test = df[df.training == "test"]

X_train = train[PREDICTORS]
y_train = train.growth
X_test = train[PREDICTORS]
y_test = test.growth

model = DecisionTreeRegressor(min_samples_leaf=5, random_state=start.SEED, max_depth=2)
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
