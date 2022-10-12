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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=44)
len(y_train)
len(y_test)

# X_train = df[predictors]
# y_train = df.growth
# %%
model = DecisionTreeRegressor(min_samples_leaf=10)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# %%
model.tree_.node_count
n_nodes = model.tree_.node_count
children_left = model.tree_.children_left
children_right = model.tree_.children_right
feature = model.tree_.feature
threshold = model.tree_.threshold

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


# %%
# %%

plt.figure(figsize=(10, 8), dpi=150)
plot_tree(model, feature_names=X.columns)
# %%

plt.savefig(start.MAIN_DIR + "results/tree.pdf")

# %%
df_test = X_test.merge(y_test, left_index=True, right_index=True)
# %%

mod = smf.ols(formula="growth ~ treat", data=df_test)
res = mod.fit()
res.summary(())

print(df_test[df_test.treat == 0].growth.mean())
print(len(df_test[df_test.treat == 0].growth))

print(df_test[df_test.treat == 1].growth.mean())
print(len(df_test[df_test.treat == 1].growth))

# %%
left = df_test[df_test.treat == 0]
left["tses_is_cutoff"] = np.where(left.tses_is <= 6.562, 0, 1)
# left["tses_is_cutoff"] = np.where(left.tses_is <= 6.562, 0, 1)

mod = smf.ols(formula="growth ~ tses_is_cutoff", data=left)
res = mod.fit()
res.summary(())

print(left[left.tses_is_cutoff == 0].growth.mean())
print(len(left[left.tses_is_cutoff == 0].growth))

print(left[left.tses_is_cutoff == 1].growth.mean())
print(len(left[left.tses_is_cutoff == 1].growth))
# %%
right = df_test[df_test.treat == 1]
right["cutoff"] = np.where(right.neo_n <= 2.39, 0, 1)
# right["cutoff"] = np.where(right.neo_n <= 2.375, 0, 1)

mod = smf.ols(formula="growth ~ cutoff", data=right)
res = mod.fit()
res.summary(())

print(right[right.cutoff == 0].growth.mean())
print(len(right[right.cutoff == 0].growth))

print(right[right.cutoff == 1].growth.mean())
print(len(right[right.cutoff == 1].growth))

# %%
right_right = right[right.treat == 1]
right_right["cutoff"] = np.where(right_right.tses_is <= 6.062, 0, 1)
# right_right["cutoff"] = np.where(right_right.tses_is <= 6.062, 0, 1)

mod = smf.ols(formula="growth ~ cutoff", data=right_right)
res = mod.fit()
res.summary(())

print(right_right[right_right.cutoff == 0].growth.mean())
print(len(right_right[right_right.cutoff == 0].growth))

print(right_right[right_right.cutoff == 1].growth.mean())
print(len(right_right[right_right.cutoff == 1].growth))

# %%
