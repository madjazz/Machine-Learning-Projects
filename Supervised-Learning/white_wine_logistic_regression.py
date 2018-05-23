# Vanilla Logistic Regression with Train-Test Split and Cross-Validation
# ----------------------------------------------------------------------

# Import Packages
# ---------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score

# Import and Explore Data
# ---------------------------

df = pd.read_csv("/Users/Timo/Documents/Propulsion_Academy/Teaching/Batch_3/dswd-2018-04/05-Machine-Learning-2/00_Data/winequality-white.csv", sep=";")

print(df.info())

# Check Response Variable Distribution
# ------------------------------------

df["quality"].hist()

# Verdict: Classes are highly unbalanced
# Solution: Undersampling of minority classes since observations < 10000

# Check distributions of predictor variables
# ------------------------------------------

plt.figure()
plt.suptitle("Distributions of Values for each column")
for i in range(1, len(df.columns) - 1):
    plt.subplot(10, 1, i)
    ax = sns.distplot(df[df.columns[i]], kde=False, rug=False)
    ax.set_title(df.columns[i])
    ax.set_xlabel("")
plt.show()

# Verdict: The "chlorides" variable is unbalanced.
# Solution: Correct with log-transformation

df["chlorides"] = np.log10(df["chlorides"])
df["chlorides"].hist()

# Perform Cross Validation with Vanilla Logistic Regression
# ---------------------------------------------------------

X = df.drop("quality", axis=1)
y = df["quality"]

sm = SMOTE(random_state=42, k_neighbors=4)

X_over, y_over = sm.fit_sample(X.values, y.values)

scaler = StandardScaler()
scaler.fit(X_over)

X_scaled = scaler.transform(X_over)

model = LogisticRegression()

scores = cross_val_score(model, X_scaled, y_over, cv=10)
accuracy = np.mean(scores)

print(accuracy)

# Penalize Model
# --------------

l1_model = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
l1_scores = cross_val_score(l1_model, X_scaled, y_over, cv=10)
l1_accuracy = np.mean(l1_scores)
print(l1_accuracy)

l2_model = LogisticRegression(penalty="l2")
l2_scores = cross_val_score(l2_model, X_scaled, y_over, cv=10)
l2_accuracy = np.mean(l2_scores)
print(l2_accuracy)