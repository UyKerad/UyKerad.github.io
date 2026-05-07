# Predicting Upsets in Professional Tennis Matches
### Spring 2026 Data Science Project | Contributors: Alexander Cui, Alex Luo, Dhruv Das, Vincent DePasquale, Darek Yu, Vineth Mova


## 2. Introduction
Professional tennis is a highly dynamic sport where match outcomes are influenced by a multitude of complex variables. While the ATP (Association of Tennis Professionals) ranking system provides a reliable metric of long-term player consistency, upsets, a scenario where a statistically inferior player defeats a higher-ranked opponent, do occur. This project uses a dataset of over 60,000 ATP tennis matches to build a machine learning classification model that predicts the likelihood of these ranking-based upsets. By integrating ranking differences, playing surface, match round (within a tournament), betting odds, and match format, this analysis seeks to uncover the statistics behind upsets.

We aim to answer the following question: What are the most significant pre-match factors that affect upsets?

This analysis bridges sports analytics and practical data science. In tennis, predicting upsets reveals factors that rankings may miss. It can help us understand how robust and accurate the betting market for tennis is.  

## 3. Data Curation
We are using a dataset from Kaggle which contains over 60,000 matches from ATP tennis matches. ATP is the top tier tour for professional men’s tennis, so the dataset is looking at the best players in the world. This means the ranking system is highly structured and updated consistently, so the ranking variable is more statistically reliable and interpretable. Furthermore, ATP players are all elite professionals, so there will be less random variance in performance, which means the effects of variables such as court surface and match round will have a higher relative importance. The dataset contains tournament name, type, location, the data and match round, whether it was indoor/outdoor, the surface type, the player names, ranking, and winner, odds for each player, as well as the score and match format.

### 3.1 Loading Data and Preprocessing
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import ttest_ind
import scipy
import re

# Load the dataset
df = pd.read_csv("atp_tennis.csv")

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names and types:")
print(df.dtypes)
df.head()
```
<img width="1620" height="610" alt="1a8ec84696514979862863151a5898e1" src="https://github.com/user-attachments/assets/6d8c4871-b944-4fc6-a472-680dfd1a99f5" />

```
PREPROCESSING GOES HERE
```
## 4. Exploratory Data Analysis
### 4.1 Does Court Surface Affect Upset Rate? (Chi-Squared Test)
```
contingency = pd.crosstab(valid["Surface"], valid["upset"])

chi2, p, dof, expected = scipy.stats.chi2_contingency(contingency)

print(f"Chi-squared statistic: {chi2:.4f}")
print(f"p-value:               {p:.6f}")
print(f"Degrees of freedom:    {dof}")

# Plot upset rate by surface
upset_rate = valid.groupby("Surface")["upset"].mean().reset_index()
upset_rate = upset_rate.sort_values("upset", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
colors = sns.color_palette("coolwarm", len(upset_rate))
bars = ax.bar(upset_rate["Surface"], upset_rate["upset"], color=colors, edgecolor='black', linewidth=0.7)

ax.set_title("Upset Rate by Court Surface", fontsize=14, fontweight='bold')
ax.set_ylabel("Proportion of Upsets")
ax.set_xlabel("Surface")
ax.set_ylim(0, 0.45)
ax.axhline(valid['upset'].mean(), color='gray', linestyle='--', label=f'Overall mean ({valid["upset"].mean():.3f})')
ax.legend()

for bar, val in zip(bars, upset_rate["upset"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}",
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("upset_by_surface.png", dpi=150)
plt.show()
```
<img width="888" height="544" alt="82266bcd76e1bd9d3584bc0338300c95" src="https://github.com/user-attachments/assets/f1324fc6-9dda-40dc-88bc-557b2c8b6110" />

### 4.2 Does Court Type (Indoor/Outdoor) Affect Match Length? (T-Test)

```
# keep only relevant columns
ct = df[["Court", "Score", "Best of"]]
ct = ct[ct["Best of"] == 3]

# drop missing values
ct = ct.dropna()

# function to compute match length
def compute_total_games(score_string):

    # if score is not a string, reject it
    if not isinstance(score_string, str):
        return None

    # remove leading/trailing spaces
    score_string = score_string.strip()

    total_games = 0

    # split into sets
    sets = score_string.split()

    for set_score in sets:

        # remove tiebreak info, like 7-6(5) -> 7-6
        set_score = re.sub(r"\(.*?\)", "", set_score)

        # only process things that look like x-y
        if "-" not in set_score:
            continue

        parts = set_score.split("-")

        # should have exactly 2 numbers
        if len(parts) != 2:
            continue

        try:
            games1 = int(parts[0])
            games2 = int(parts[1])
            total_games += games1 + games2
        except ValueError:
            return None

    # if nothing valid was counted, reject row
    if total_games == 0:
        return None

    return total_games

ct["total_games"] = ct["Score"].apply(compute_total_games)

ct = ct[ct["Court"].isin(["Indoor", "Outdoor"])]

indoor = ct[ct["Court"] == "Indoor"]["total_games"]
outdoor = ct[ct["Court"] == "Outdoor"]["total_games"]

stat, p_value = ttest_ind(indoor, outdoor, equal_var=False)  # t-test

print("t-statistic:", stat)
print("p-value:", p_value)

alpha = 0.05

if p_value < alpha:
    print("Reject H0: match length differs by court type")
else:
    print("Fail to reject H0: no significant difference")

print(ct.groupby("Court")["total_games"].agg(["count", "mean", "std"]))

plt.figure()
ct.boxplot(column="total_games", by="Court")
plt.title("Match Length (Total Games) by Court Type")
plt.suptitle("")
plt.xlabel("Court Type")
plt.ylabel("Total Games")
plt.show()
```
<img width="772" height="531" alt="4acf2a74432d38c950b3ef84e5e343f6" src="https://github.com/user-attachments/assets/a19d2c32-f242-4d71-adf0-67c79ad56cf0" />

### 4.3 Are there outliers in the rankings of people who defeat top-10 players? In other words, what are the biggest upsets?
```
#filter to matches where top 10 players played and lost
top_10_losses=valid[((valid['Rank_1']<=10)&(valid['Winner']==valid['Player_2']))|((valid['Rank_2']<= 10)&(valid['Winner']==valid['Player_1']))].copy()

#compile list of rankings of the winners in such matches
top_10_losses['Winner_Rank'] = top_10_losses.apply(lambda x: x['Rank_1'] if x['Winner'] == x['Player_1'] else x['Rank_2'],axis=1)

#create quartile range
Q1=top_10_losses['Winner_Rank'].quantile(0.25)
Q3=top_10_losses['Winner_Rank'].quantile(0.75)
IQR=Q3-Q1
outlier_threshold=Q3+1.5*IQR

#identify outliers
outliers = top_10_losses[top_10_losses['Winner_Rank']>outlier_threshold]
print(f"Statistically extreme outliers (Rank > {outlier_threshold:.0f}): {len(outliers)}")
display(outliers)

#indentifying the biggest outlier/upset
biggest_outlier = outliers.loc[outliers['Winner_Rank'].idxmax()]
display(biggest_outlier.to_frame().T)

#Create box plot
plt.figure(figsize=(12, 6))
top_10_losses.boxplot(column='Winner_Rank',vert=False)
plt.title("Outlier Analysis: Rankings of Players who Defeated a Top 10 Opponent", fontsize=14)
plt.xlabel("ATP Rank of the Winner")
plt.ylabel("Top 10 Losses")
plt.show()
```
<img width="1219" height="624" alt="ace5dbc099069447e55685d49b2e0277" src="https://github.com/user-attachments/assets/2e19a3a8-e913-46df-b8db-e0b584a974ae" />

## 5. Primary Analysis

## 6. Visualizations

## Conclusions

