# accident_analysis.py

import pandas as pd

# Load the sample dataset
df = pd.read_csv("data/US_Accidents_Sample.csv")

# Show basic info
print("✅ Data loaded")
print("Shape:", df.shape)
print("\nSample rows:")
print(df.head())
# Check for missing values
print("\n✅ Missing values per column:\n", df.isnull().sum())

# Drop rows with nulls in key columns for simplicity
df.dropna(subset=["Weather_Condition", "Start_Time", "Visibility(mi)", "Temperature(F)"], inplace=True)

print("\n✅ After dropping nulls, new shape:", df.shape)
import matplotlib.pyplot as plt
import seaborn as sns

# Convert Start_Time to datetime and extract hour
df["Start_Time"] = pd.to_datetime(df["Start_Time"])
df["Hour"] = df["Start_Time"].dt.hour

# Plot accident frequency by hour
plt.figure(figsize=(10, 5))
sns.countplot(x="Hour", data=df, palette="coolwarm")
plt.title("Accidents by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.savefig("images/accidents_by_hour.png", dpi=300)
plt.show()
# Top 10 weather conditions in accidents
plt.figure(figsize=(12, 6))
top_weather = df["Weather_Condition"].value_counts().nlargest(10).index
sns.countplot(data=df[df["Weather_Condition"].isin(top_weather)],
              x="Weather_Condition", order=top_weather, palette="muted")
plt.title("Top 10 Weather Conditions in Accidents")
plt.xticks(rotation=45)
plt.xlabel("Weather Condition")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.savefig("images/accidents_by_weather.png", dpi=300)
plt.show()
# Top 10 cities with most accidents
plt.figure(figsize=(12, 6))
top_cities = df["City"].value_counts().nlargest(10)
sns.barplot(x=top_cities.index, y=top_cities.values, palette="flare")
plt.title("Top 10 Cities with Most Accidents")
plt.xlabel("City")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/top_cities_accidents.png", dpi=300)
plt.show()
# Correlation heatmap
plt.figure(figsize=(8, 6))
numerical_cols = ["Severity", "Temperature(F)", "Humidity(%)", "Visibility(mi)", "Wind_Speed(mph)"]
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap - Numerical Features")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=300)
plt.show()
