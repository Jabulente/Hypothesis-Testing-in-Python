from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
import numpy as np
import re

def rename(text): return re.sub(r'[^a-zA-Z]', "", text)

def compute_one_way_anova(df, numerical_columns, group_cols):
    results = []
    original_group_cols = group_cols[:]
    group_cols = [rename(col) for col in group_cols]
    df = df.rename(columns={col: rename(col) for col in df.columns})
    
    for original_group, group in zip(original_group_cols, group_cols):
        for col in numerical_columns:
            column_name = rename(col)  
            formula = f"{column_name} ~ C({group})" 
            model = smf.ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            for source, row in anova_table.iterrows():
                p_value = row["PR(>F)"]
                interpretation = "Significant" if p_value < 0.05 else "No significant"
                if source == "Residual": interpretation = "-"
                
                results.append({
                    "Group": original_group,
                    "Variable": col,
                    "Source": source,
                    "Sum Sq": row["sum_sq"],
                    "df": row["df"],
                    "F-Value": row["F"],
                    "p-Value": p_value,
                    "Significant (α<0.05)": interpretation
                })
    results = pd.DataFrame(results)
    return results.fillna(' ')