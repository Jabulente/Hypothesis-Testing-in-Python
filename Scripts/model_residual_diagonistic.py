import statsmodels.formula.api as smf
from scipy.stats import shapiro
import pandas as pd
import numpy as np
import re 

filepath = "Datasets/Soil Health Dataset.csv"
soil_df = pd.read_csv(filepath)
print(soil_df)

def perform_groupwise_shapiro_wilk_test(dataframe, numerical_columns, group_column):
    def rename(text): return re.sub(r'[^a-zA-Z]', "", text)
    if len(numerical_columns) == 0: raise ValueError("No numerical columns provided for testing.")
    
    renamed_columns = {col: rename(col) for col in dataframe.columns}
    dataframe = dataframe.rename(columns=renamed_columns)
    renamed_group_column = rename(group_column)

    test_results = []
    for original_col in numerical_columns:
        renamed_col = rename(original_col)
        formula = f"{renamed_col} ~ C({renamed_group_column})"
        model = smf.ols(formula, data=dataframe).fit()
        residuals = model.resid

        stat, p_value = shapiro(residuals)
        test_results.append({
            'Group': group_column,
            'Variable': original_col,
            'Statistic': stat,
            'P-Value': p_value,
            'Normality': 'Normally Distributed' if p_value > 0.05 else 'Not Normally Distributed'
        })

    results_df = pd.DataFrame(test_results)
    return results_df

numerical_columns = soil_df.select_dtypes(include=[np.number]).columns
shapiro_test_results = perform_groupwise_shapiro_wilk_test(soil_df, numerical_columns, group_column='Soil Texture')
print(shapiro_test_results)