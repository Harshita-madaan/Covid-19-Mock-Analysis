import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("mock_covid_data_200_rows.csv")
print(df.head())
df["date"]=pd.to_datetime(df["date"])
print(df.isnull().sum())
print(df.info())
print(df.describe())

df.drop_duplicates(inplace=True)

#Top Countries by Total Cases
latest=df.groupby("location").tail(1)
top_cases=latest.sort_values("total_cases",ascending=False)
print(top_cases[["location","total_cases"]].head(10))

#Death Rate by Country
latest["death_rate"] = (latest["total_deaths"] / latest["total_cases"]) * 100
print(latest[["location", "death_rate"]].sort_values(by="death_rate", ascending=False).head(10))

#Trends Over Time(India)
india=df[df["location"]=="India"]
plt.figure(figsize=(10,5))
plt.plot(india["date"],india["new_cases"])
plt.title("Daily New Cases in India")
plt.xlabel("Date")
plt.ylabel("New Cases")
plt.grid(True)
plt.show()

#Multiple Countries Comparison
countries=["India","Brazil","USA"]
for country in countries:
    subset=df[df["location"]==country]
    plt.plot(subset["date"], subset["new_cases"], label=country)

plt.title("Daily New Cases Comparison")  
plt.legend()
plt.xlabel("Date")
plt.ylabel("New Cases")  
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Rolling Averages
df["new_cases_7day_avg"] = df.groupby("location")["new_cases"].transform(lambda x: x.rolling(7).mean())

#Top 5 countries by Total Cases
top5=top_cases.head(5)
location=top5["location"]
cases=top5["total_cases"]
plt.figure(figsize=(10,5))
plt.bar(location,cases,color="skyblue",edgecolor="black")
plt.xlabel("Total Cases")
plt.ylabel("Top 5 Countries by Total Cases")
plt.grid(axis='x',linestyle='--',alpha=0.7)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#Total Cases vs Death
plt.scatter(latest["total_cases"],latest["total_deaths"],alpha=0.7)
plt.xlabel("Total Cases")
plt.ylabel("Total Deaths")
plt.title("Cases Vs Deaths")
plt.grid(True)
plt.show()
