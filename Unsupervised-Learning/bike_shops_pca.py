import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


# Load data into memory
# ---------------------

customers = pd.read_csv("customers_bikeshops.csv")
orders = pd.read_csv("orders.csv", index_col=0)
products = pd.read_csv("products_bikes.csv")

# Merge datasets
# --------------

# product.id = bike.id
# bikeshop.id = customer.id

orders_products = pd.merge(orders, products, left_on="product.id", right_on="bike.id")
bike_data = pd.merge(customers, orders_products, left_on="bikeshop.id", right_on="customer.id")

# Make correlation plot
# ---------------------

corr = bike_data.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

# No strong correlations between variables that are not identical detected.

# Group data
# ----------

# Group by bikeshop and bike model
cluster_data = bike_data.groupby(["bikeshop.name", "model"]).\
    agg({"quantity": "sum", "price": "sum"}).reset_index()

# Compute sales
cluster_data["sales"] = cluster_data["quantity"] * cluster_data["price"]

# Select columns of interest
cluster_data = cluster_data[["bikeshop.name", "model", "sales"]]

# Pivot data
cluster_data = cluster_data.pivot(index="bikeshop.name", columns="model", values="sales").reset_index()

# Fill NaNs with 0s (NaN indicates no sales)
cluster_data = cluster_data.fillna(0)

# Normalize Scores
# ----------------

# Normalize data so that mean = 0 and standard deviation = 1

X = cluster_data.drop(["bikeshop.name"], axis=1)
y = cluster_data["bikeshop.name"]

normalized_data = scale(X)
sns.heatmap(normalized_data)

# PCA
# ---

# Our goal with PCA here is to reduce the dimensionality
# of the data on the bike sales per bike shop

model = PCA().fit(normalized_data)

# Visualize explained variance ratio and cumulative explained variance ratio

plt.subplot(1, 2, 1)
plt.plot(range(len(model.explained_variance_ratio_)), model.explained_variance_ratio_)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Proportions of Variance Explained by each Component')


plt.subplot(1, 2, 2)
plt.plot(range(len(model.explained_variance_ratio_)), np.cumsum(model.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()

# We now have 5 components which capture over 90% of the variance.
