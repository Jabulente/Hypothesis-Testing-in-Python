# Import Required Libraries
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
import numpy as np
import re

# Clean columns names to avoid error whene fiting models
def rename(text): return re.sub(r'[^a-zA-Z]', "", text)

# Perform Anova Test over Multiple Groups and Variables and Return The results in One Formated DataFrame
def One_way_anova(data, Metrics, group_cols):
    results = []
    original_group_cols = group_cols[:]
    group_cols = [rename(col) for col in group_cols]
    data = data.rename(columns={col: rename(col) for col in data.columns})
    
    for original_group, group in zip(original_group_cols, group_cols):
        for col in Metrics:
            column_name = rename(col)  
            formula = f"{column_name} ~ C({group})" 
            model = smf.ols(formula, data=data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            for source, row in anova_table.iterrows():
                p_value = row["PR(>F)"]
                interpretation = "Significant" if p_value < 0.05 else "No significant"
                if source == "Residual":
                    interpretation = "-"
                
                results.append({
                    "Variable": col,
                    #"Factor": original_group,  # Use original name here
                    "Source": source,
                    "Sum Sq": row["sum_sq"],
                    "df": row["df"],
                    "F-Value": row["F"],
                    "p-Value": p_value,
                    "Interpretation": interpretation
                })

    return pd.DataFrame(results)

# Importing Clean and Structured Dataset
filepath = "../Datasets/Eggplant Fusarium Fertilizer Data.csv"
df = pd.read_csv(filepath)

group_cols = ['Fertilizer'] # A list of group or Factors
Metrics = ['Infection Severity (%)', 'Wilt index', 'Plant height (cm)', 'Days to wilt symptoms', 'Survival rate (%)', 'Disease incidence (%)']
Anova_results = One_way_anova(df, Metrics, group_cols) # Perfome ANOVA Test
display(Anova_results) # Display Results