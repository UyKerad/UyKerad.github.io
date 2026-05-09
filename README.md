# Predicting Upsets in Professional Tennis Matches
### Spring 2026 Data Science Project | Contributors: Alexander Cui, Alex Luo, Dhruv Das, Vincent DePasquale, Darek Yu, Vineeth Movva

## 1. Contributions
**A (Project idea).**  
    - Final project idea on using Random Forest was brainstormed by Alexander Cui. Rest of the team agreed and expanded on the idea.   
**B (Dataset Curation and Preprocessing).**  
    - Dataset was found by Darek Yu. Rest of the team agreed on utilizing the dataset.  
**C (Data Exploration and Summary Statistics).**     
    - First Data Exploration on Court Surface v.s. Upset Rate was done by Darek Yu.  
    - Third Data Exploration was done by Alexander Cui.  
    - Dhruv Das did the second Data Exploration regarding the outliers in the rankings and the Visualization.   
    - Alex Luo worked on bug checking/correcting + conclusions for first and second data exploration.
**D (ML Algorithm Design/Development).**  
    - Feature Engineering and the new columns generated was done by Darek Yu.  
**E (ML Algorithm Training and Test Data Analysis).**    
    - Model training, testing (Logistic Regression, Random Forest, Gradient Boosting) was done by Vineeth Movva.
    
**F (Visualization, Result Analysis, Conclusion).** 
    - Model evaluation and grouped bar chart were done by Alex Luo
    - Feature important plot was done by Dhruv Das.
    
**G (Final Tutorial Report Creation).**  
    - Alexander Cui wrote the Introduction and Data Curation portions of this assignment.  
    - Website is hosted and was formatted by Darek Yu.   

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
```
Dataset shape: (65884, 17)
Tournament     object
Date           object
Series         object
Court          object
Surface        object
Round          object
Best of         int64
Player_1       object
Player_2       object
Winner         object
Rank_1          int64
Rank_2          int64
Pts_1           int64
Pts_2           int64
Odd_1         float64
Odd_2         float64
Score          object
dtype: object
index,Tournament,Date,Series,Court,Surface,Round,Best of,Player_1,Player_2,Winner,Rank_1,Rank_2,Pts_1,Pts_2,Odd_1,Odd_2,Score
0,Australian Hardcourt Championships,2000-01-03,International,Outdoor,Hard,1st Round,3,Dosedel S.,Ljubicic I.,Dosedel S.,63,77,-1,-1,-1.0,-1.0,6-4 6-2
1,Australian Hardcourt Championships,2000-01-03,International,Outdoor,Hard,1st Round,3,Clement A.,Enqvist T.,Enqvist T.,56,5,-1,-1,-1.0,-1.0,3-6 3-6
2,Australian Hardcourt Championships,2000-01-03,International,Outdoor,Hard,1st Round,3,Escude N.,Baccanello P.,Escude N.,40,655,-1,-1,-1.0,-1.0,6-7 7-5 6-3
3,Australian Hardcourt Championships,2000-01-03,International,Outdoor,Hard,1st Round,3,Knippschild J.,Federer R.,Federer R.,87,65,-1,-1,-1.0,-1.0,1-6 4-6
4,Australian Hardcourt Championships,2000-01-03,International,Outdoor,Hard,1st Round,3,Fromberg R.,Woodbridge T.,Fromberg R.,81,198,-1,-1,-1.0,-1.0,7-6 5-7 6-4
```

```
# Keep only rows where both players have valid (positive) rankings
valid = df[(df['Rank_1'] > 0) & (df['Rank_2'] > 0)].copy()
print(f"Rows with valid rankings: {len(valid)} / {len(df)} ({len(valid)/len(df)*100:.1f}%)")

# Determine which player was favored (lower rank = better)
valid['p1_wins']    = (valid['Winner'] == valid['Player_1'])
valid['p1_favored'] = (valid['Rank_1'] < valid['Rank_2'])

# An upset is when the lower-ranked (higher number) player wins
valid['upset'] = (valid['p1_favored'] != valid['p1_wins']).astype(int)

print(f"\nOverall upset rate: {valid['upset'].mean():.3f}  ({valid['upset'].sum():,} upsets out of {len(valid):,} matches)")
print(f"\nClass balance (0 = expected result, 1 = upset):")
print(valid['upset'].value_counts())
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
### 5.1 Motivation
Our exploratory data analysis confirmed that variables linked the tournament context all correlate with upset probability. We now build a binary classification model to predict whether a given match will be an upset (1) or not (0).
We will compare three models:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
   
### 5.2 Feature Engineering
We will engineer the following features:
1. **log_rank_diff** - Log of absolute ranking gap between players; A higher value means more one-sided on paper
2. **log_favored_rank** - Log of the higher ranked player's rank; Captures the quality of the favorite
3. **implied_prob_fav** - Normalized implied probability from betting odds
4. **round_num**- Encoded tournament round (1st Round -> Final)
5. **series_tier** - Tournament prestige tier (International -> Grand Slam)
6. **Best_of** - Match format (3 or 5 sets)
7. **Surface** - One-hot encoded court surface
   
```
# Work with rows that have valid odds
odds_valid = valid[(valid['Odd_1'] > 0) & (valid['Odd_2'] > 0)].copy()
print(f"Matches with valid betting odds: {len(odds_valid):,}")

# Rank features
odds_valid['rank_diff'] = abs(odds_valid['Rank_1'] - odds_valid['Rank_2'])
odds_valid['log_rank_diff'] = np.log1p(odds_valid['rank_diff'])
odds_valid['favored_rank'] = odds_valid[['Rank_1', 'Rank_2']].min(axis=1)
odds_valid['log_favored_rank'] = np.log1p(odds_valid['favored_rank'])

# Betting implied probability for the favorite (normalized to remove overround)
raw_fav  = odds_valid.apply(lambda r: 1/r['Odd_1'] if r['p1_favored'] else 1/r['Odd_2'], axis=1)
raw_dog  = odds_valid.apply(lambda r: 1/r['Odd_2'] if r['p1_favored'] else 1/r['Odd_1'], axis=1)
overround = raw_fav + raw_dog
odds_valid['implied_prob_fav'] = raw_fav / overround

# Encode round as ordinal
round_order = {'1st Round': 1, '2nd Round': 2, '3rd Round': 3, '4th Round': 4,
               'Round Robin': 3, 'Quarterfinals': 5, 'Semifinals': 6, 'The Final': 7}
odds_valid['round_num'] = odds_valid['Round'].map(round_order).fillna(3).astype(int)

# Encode tournament tier
series_order = {'International': 1, 'International Gold': 2, 'ATP250': 2,
                'ATP500': 3, 'Masters': 4, 'Masters 1000': 4,
                'Masters Cup': 5, 'Grand Slam': 5}
odds_valid['series_tier'] = odds_valid['Series'].map(series_order).fillna(2).astype(int)

feature_cols = ['log_rank_diff', 'log_favored_rank', 'implied_prob_fav',
                'round_num', 'series_tier', 'Best of', 'Surface']

X = pd.get_dummies(odds_valid[feature_cols], columns=['Surface'])
y = odds_valid['upset']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")
```
```
Matches with valid betting odds: 62,090
Feature matrix shape: (62090, 10)
Features: ['log_rank_diff', 'log_favored_rank', 'implied_prob_fav', 'round_num', 'series_tier', 'Best of', 'Surface_Carpet', 'Surface_Clay', 'Surface_Grass', 'Surface_Hard']
```

### 5.3 Model Training
```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import numpy as np
# Train-test split (stratified to preserve upset rate)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train upset rate:", round(y_train.mean(), 4), "Test upset rate:", round(y_test.mean(), 4))
# Baseline: always predict no upset
y_pred_base = np.zeros(len(y_test), dtype=int)
print("\nBaseline (always predict no upset)")
print("Accuracy:", round(accuracy_score(y_test, y_pred_base), 4))
print("Balanced accuracy:", round(balanced_accuracy_score(y_test, y_pred_base), 4))
print("F1:", round(f1_score(y_test, y_pred_base, zero_division=0), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_base))
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}
def eval_model(name, model):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # map metrics to their values for later use
    results[name] = {
        "ROC-AUC": round(roc_auc_score(y_test, proba), 4),
        "PR-AUC": round(average_precision_score(y_test, proba), 4),
        "F1": round(f1_score(y_test, pred, zero_division=0), 4),
        "Balanced Acc": round(balanced_accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "Accuracy": round(accuracy_score(y_test, pred), 4),
    }
    
    print(f"\n{name}")
    for m, v in results[name].items():
        print(f"{m}: {v}")

for name, model in models.items():
    eval_model(name, model)
```

## 6. Visualization
```
# axes names
results_df = pd.DataFrame(results).T
metrics = results_df.columns.tolist()
model_names = results_df.index.tolist()

#set up plot metadata
x = np.arange(len(metrics))
colors = ["red", "green", "blue"]
width = 0.25
fig, ax = plt.subplots(figsize=(11, 6))

# create sick grouped bar chart
for i, name in enumerate(model_names):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, results_df.loc[name], width, label=name, color=colors[i])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=6)

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)
ax.set_title("Model Comparisons for Tennis Upset Prediction")
ax.legend(loc="upper right")
plt.show()
```
<img width="923" height="528" alt="image" src="https://github.com/user-attachments/assets/a663b33a-529f-4a1e-95b5-06a2e8e310b2" />

For our baseline where we always predict no upset, we see Accuracy = 0.6541 but Balanced Accuracy = 0.5000 and F1 = 0.0000. F1 is 0 because it never identifies an upset (Recall = 0), and accuracy is of course, just 1 - the upset rate of 0.3459. We see that Accuracy is an unreliable metric for this dataset since it is unbalanced.

Logistic Regression showed the strongest overall performance. It tied for the highest ROC-AUC (0.7186), scored the best Balanced Accuracy = 0.6561, F1 = 0.5667, and Recall = 0.6421. However, its Precision is the lowest at 0.5072.

Random Forest performed between the other two models. It had slightly higher accuracy (0.6655) than Logistic Regression but performed worse in ROC-AUC = 0.7064, PR-AUC = 0.5431 and  F1 = 0.5440. It's more conservative than Logistic Regression fewer predicted upsets, with lower Recall = 0.5769, but no corresponding gain in Precision = 0.5146.

Gradient Boosting showed the most extreme tradeoff of all of these models, having the highest Accuracy = 0.6945, PR-AUC = 0.5580, and Precision = 0.6018. However, it has by far the lowest Recall = 0.3448 and lowest F1 = 0.4384. This overall means that it has high confidence but behaves very conservatively.

Depending on the use case, logistic regression and gradient boosting are both solid contenders. For example, if you are using the model to place bets, you want to have high precision and don't really care about recall, making gradient boosting the best choice. For our purposes, we will choose logistic regression since it has the best balance of precision and recall, and, more importantly, the coefficients directly concern themselves with our question of with features are most important for predicting upsets.

**NEED GRAPH FINDING FEATURE IMPORTANCE!!!!!!!!!!!!!!!!!!!!!!**

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

lr_model = models["Logistic Regression"].named_steps['clf']
feature_names = X.columns

importance = lr_model.coef_[0]

feat_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
feat_importance = feat_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_importance, hue='Feature', palette="vlag", legend=False)
plt.axvline(x=0, color='black', lw=1)
plt.title('Most Significant Factors for ATP Upsets (Logistic Regression Coefficients)', fontsize=14)
plt.xlabel('Coefficient Magnitude (Directional Influence)')
plt.ylabel('Match Variable')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
```
<img width="944" height="548" alt="feature_importance" src="https://github.com/user-attachments/assets/a7a1cc66-3be3-4617-850f-73173b88a6bc" />


## 7. Conclusions

