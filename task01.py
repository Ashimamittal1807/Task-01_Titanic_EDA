import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())
print(df.info())
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column because it has too many missing values

df.drop("Cabin", axis=1, inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Check again
print("Duplicate Rows After Removal:", df.duplicated().sum())
print("\nData Types:")
print(df.dtypes)
# Box Plot for Age
plt.figure(figsize=(6,4))
sns.boxplot(x=df["Age"])
plt.title("Box Plot of Age")
plt.show()

# Remove Outliers using IQR

Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Age"] >= lower) & (df["Age"] <= upper)]

print("Dataset Shape After Removing Outliers:", df.shape)
plt.figure(figsize=(6,4))
plt.hist(df["Age"], bins=20, edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Bar Chart - Survival Count

plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

# Correlation Heatmap

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# Survival by Gender

plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

df.to_csv("Cleaned_Titanic.csv", index=False)

print("Cleaned dataset saved successfully!")
