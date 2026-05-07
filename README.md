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
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("atp_tennis.csv")

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names and types:")
print(df.dtypes)
df.head()
```

## 4. Exploratory Data Analysis

## 5. Primary Analysis

## 6. Visualizations

## Conclusions

