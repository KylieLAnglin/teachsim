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

# %%
FILE_NAME = "feedback_analysis_withpre_post_survey_wide.dta"
SEED = 6
TEST_SIZE = 30
# %%
df = pd.read_stata(
    start.RAW_DATA_DIR + FILE_NAME,
)

df.head()

df["treat"] = df["treat"].map({"No Coaching": 0, "Coaching": 1})

# %%


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
    "score0",
]

# predictors = [
#     "score1",
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
    X, y, test_size=TEST_SIZE, random_state=SEED
)
len(y_train)
len(y_test)

# X_train = df[predictors]
# y_train = df.growth
# %%
model = DecisionTreeRegressor(min_samples_leaf=10)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
plt.figure(figsize=(10, 8), dpi=150)
plot_tree(model, feature_names=X.columns)
# %%

plt.savefig(start.MAIN_DIR + "results/tree.pdf")

# %%
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, predictions)

# %%
df_test = X_test.merge(y_test, left_index=True, right_index=True)
df_test["predictions"] = predictions
mod = smf.ols(
    formula="growth ~ predictions",
    data=df_test,
)
res = mod.fit()
print(res.summary())

# %%
model.tree_.node_count
n_nodes = model.tree_.node_count
children_left = model.tree_.children_left
children_right = model.tree_.children_right
feature = model.tree_.feature
threshold = model.tree_.threshold

# %%

len(df_test)
# %%

mod = smf.ols(formula="growth ~ treat", data=df_test)
res = mod.fit()
print(res.summary(()))

print(df_test[df_test.treat == 0].growth.mean())
print(len(df_test[df_test.treat == 0].growth))

print(df_test[df_test.treat == 1].growth.mean())
print(len(df_test[df_test.treat == 1].growth))

# %%
left = df_test[df_test.treat == 0]

predictor = X_train.columns[feature[1]]
predictor_threshold = threshold[1]
print(predictor, str(predictor_threshold))
left["predictor_cutoff"] = np.where(left[predictor] > predictor_threshold, 1, 0)

mod = smf.ols(formula="growth ~ predictor_cutoff", data=left)
res = mod.fit()
print(res.summary(()))

print(left[left.predictor_cutoff == 0].growth.mean())
print(len(left[left.predictor_cutoff == 0].growth))

print(left[left.predictor_cutoff == 1].growth.mean())
print(len(left[left.predictor_cutoff == 1].growth))
left[left.predictor_cutoff == 1]
# %%

right = df_test[df_test.treat == 1]

predictor = X_train.columns[feature[4]]
predictor_threshold = threshold[4]
print(predictor, str(predictor_threshold))
right["predictor_cutoff"] = np.where(right[predictor] > predictor_threshold, 1, 0)

mod = smf.ols(formula="growth ~ predictor_cutoff", data=right)
res = mod.fit()
print(res.summary(()))


print(right[right.predictor_cutoff == 0].growth.mean())
print(len(right[right.predictor_cutoff == 0].growth))

print(right[right.predictor_cutoff == 1].growth.mean())
print(len(right[right.predictor_cutoff == 1].growth))

# %%
right_left = right[right.predictor_cutoff == 0]

predictor = X_train.columns[feature[5]]
predictor_threshold = threshold[5]
print(predictor, str(predictor_threshold))

right_left["predictor_cutoff"] = np.where(
    right_left[predictor] > predictor_threshold, 1, 0
)

mod = smf.ols(formula="growth ~ predictor_cutoff", data=right_left)
res = mod.fit()
res.summary(())

print(right_left[right_left.predictor_cutoff == 0].growth.mean())
print(len(right_left[right_left.predictor_cutoff == 0].growth))

print(right_left[right_left.predictor_cutoff == 1].growth.mean())
print(len(right_left[right_left.predictor_cutoff == 1].growth))


# %%
# right_right = right[right.predictor_cutoff == 1]

# predictor = X_train.columns[feature[6]]
# predictor_threshold = threshold[6]
# print(predictor, str(predictor_threshold))

# right_right["predictor_cutoff"] = np.where(
#     right_right[predictor] > predictor_threshold, 1, 0
# )

# mod = smf.ols(formula="growth ~ predictor_cutoff", data=right_right)
# res = mod.fit()
# res.summary(())

# print(right_right[right_right.predictor_cutoff == 0].growth.mean())
# print(len(right_right[right_right.predictor_cutoff == 0].growth))

# print(right_right[right_right.predictor_cutoff == 1].growth.mean())
# print(len(right_right[right_right.predictor_cutoff == 1].growth))

# %%

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
while len(stack) > 0:
    # `pop` ensures each node is only visited once
    node_id, depth = stack.pop()
    node_depth[node_id] = depth

    # If the left and right child of a node is not the same we have a split
    # node
    is_split_node = children_left[node_id] != children_right[node_id]
    # If a split node, append left and right children and depth to `stack`
    # so we can loop through them
    if is_split_node:
        stack.append((children_left[node_id], depth + 1))
        stack.append((children_right[node_id], depth + 1))
    else:
        is_leaves[node_id] = True

print(
    "The binary tree structure has {n} nodes and has "
    "the following tree structure:\n".format(n=n_nodes)
)
for i in range(n_nodes):
    if is_leaves[i]:
        print(
            "{space}node={node} is a leaf node.".format(
                space=node_depth[i] * "\t", node=i
            )
        )
    else:
        print(
            "{space}node={node} is a split node: "
            "go to node {left} if X[:, {feature}] <= {threshold} "
            "else to node {right}.".format(
                space=node_depth[i] * "\t",
                node=i,
                left=children_left[i],
                feature=feature[i],
                threshold=threshold[i],
                right=children_right[i],
            )
        )


# %%
