# %%
# %%
import pandas as pd
import numpy as np
import start

from sklearn.model_selection import train_test_split

# %%
FILE_NAME = "feedback_analysis_withpre_post_survey_wide.dta"


df = pd.read_stata(start.RAW_DATA_DIR + FILE_NAME, index_col="id")
df["growth"] = df.score2 - df.score1
df["treat"] = df["treat"].map({"No Coaching": 0, "Coaching": 1})
df.sample()
# %%
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
    "score0",
    "score1",
]

X = df[PREDICTORS]
y = df["growth"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=round(len(df) / 3), random_state=start.SEED
)

df["training"] = np.where(df.index.isin(list(X_train.index)), "train", "test")
df[df.training == "train"].growth.mean()

df.to_csv(start.MAIN_DIR + "data/data_split.csv", index=True)
# %%

# %%
