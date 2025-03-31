from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

housing = fetch_california_housing()
housing_df = pd.DataFrame(housing.data, columns = housing.feature_names)

print(housing_df.columns())