# TASK 4 - FEATURE ENCODING & SCALING

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Step 1: Create Sample Dataset

data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor'],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Chennai', 'Mumbai'],
    'Age': [22, 25, 30, 35, 28],
    'Salary': [25000, 40000, 60000, 80000, 50000]
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# Step 2: Identify Categorical Variables

# Nominal Variables:
# Gender, City

# Ordinal Variable:
# Education

# Step 3: Label Encoding (Ordinal Data)

education_order = {
    'High School': 1,
    'Bachelor': 2,
    'Master': 3,
    'PhD': 4
}

df['Education_Encoded'] = df['Education'].map(education_order)

print("\nAfter Label Encoding:\n")
print(df)

# Step 4: One-Hot Encoding (Nominal Data)


df_encoded = pd.get_dummies(df,
                            columns=['Gender', 'City'],
                            drop_first=True)

print("\nAfter One-Hot Encoding:\n")
print(df_encoded)


# Step 5: Feature Scaling


# Standardization
standard_scaler = StandardScaler()

df_encoded['Age_Standardized'] = standard_scaler.fit_transform(
    df_encoded[['Age']]
)

df_encoded['Salary_Standardized'] = standard_scaler.fit_transform(
    df_encoded[['Salary']]
)

# Normalization
minmax_scaler = MinMaxScaler()

df_encoded['Age_Normalized'] = minmax_scaler.fit_transform(
    df_encoded[['Age']]
)

df_encoded['Salary_Normalized'] = minmax_scaler.fit_transform(
    df_encoded[['Salary']]
)

print("\nAfter Scaling:\n")
print(df_encoded)

# Step 6: Visualization

# Plot for Age
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.hist(df['Age'], bins=5)
plt.title("Original Age")

plt.subplot(1,2,2)
plt.hist(df_encoded['Age_Standardized'], bins=5)
plt.title("Standardized Age")

plt.tight_layout()
plt.show()

# Plot for Salary
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.hist(df['Salary'], bins=5)
plt.title("Original Salary")

plt.subplot(1,2,2)
plt.hist(df_encoded['Salary_Normalized'], bins=5)
plt.title("Normalized Salary")

plt.tight_layout()
plt.show()
