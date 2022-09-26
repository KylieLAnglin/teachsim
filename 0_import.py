# %%
import pandas as pd
from library import start

# %%
df = pd.read_stata(
    start.RAW_DATA_DIR + "feedback_analysis_withpre_post_survey_coach_exit.dta",
)

df.head()
