import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


def clean_column_name(name): return re.sub(r'[^a-zA-Z]', '', name)

def compute_two_way_anova(df, numeric_vars, group1, group2):
    results = []
    df = df.rename(columns={col: clean_column_name(col) for col in df.columns})
    group1_clean = clean_column_name(group1)
    group2_clean = clean_column_name(group2)

    for var in numeric_vars:
        var_clean = clean_column_name(var)
        formula = f"{var_clean} ~ C({group1_clean}) * C({group2_clean})"
        try:
            model = smf.ols(formula, data=df).fit()
            anova_table = anova_lm(model, typ=2)
            for source, row in anova_table.iterrows():
                p_val = row["PR(>F)"]
                interpretation = "Significant" if p_val < 0.05 else "Not Significant"
                if source == "Residual":
                    interpretation = "-"

                results.append({
                    "Variable": var,
                    "Source": source,
                    "Sum Sq": row["sum_sq"],
                    "df": row["df"],
                    "F-Value": row["F"],
                    "p-Value": p_val,
                    "Significant (α<0.05)": interpretation
                })
        
        except Exception as e:
            print(f"ANOVA failed for variable '{var}': {e}")
    results = pd.DataFrame(results)
    return results.fillna(' ')