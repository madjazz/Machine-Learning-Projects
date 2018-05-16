import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import linkage, dendrogram

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

# Fill NaNs with 0s (NaN indicates no sale)
cluster_data = cluster_data.fillna(0)

# Normalize Scores
# ----------------

# Normalize data so that mean = 0 and standard deviation = 1

X = cluster_data.drop(["bikeshop.name"], axis=1)
y = cluster_data["bikeshop.name"]

normalized_data = scale(X)
sns.heatmap(normalized_data)

# Hierarchical Clustering
# -----------------------

# Since we are dealing with only numerical data
# Euclidean distance is an appropriate measure

model_complete = linkage(normalized_data, "complete")
model_average = linkage(normalized_data, "average")
model_single = linkage(normalized_data, "single")
model_centroid = linkage(normalized_data, "centroid")
model_median = linkage(normalized_data, "median")
model_weighted = linkage(normalized_data, "weighted")
model_ward = linkage(normalized_data, "ward")

model_list = [model_complete, model_average, model_single, model_centroid, model_median, model_weighted, model_ward]
model_labels = ["Complete Linkage", "Average Linkage", "Single Linkage", "Centroid Linkage",
                "Median Linkage", "Weighted Linkage", "Ward Linkage"]

# Plot dendrograms


def plot_dendrograms(labels, models):
    fig = plt.figure()
    fig.suptitle("Dendrograms ", fontsize=16)
    for i in range(1, len(labels)):
        ax = plt.subplot(7, 2, i)
        ax.set_title(labels[i-1])
        dendrogram(models[i-1])
    plt.show()


plot_dendrograms(model_labels, model_list)

# All of the dendrograms indicate the bike shop 10 forms a distinct cluster to the other shops.
# Removing that bike shop could make further clustering more granular.
