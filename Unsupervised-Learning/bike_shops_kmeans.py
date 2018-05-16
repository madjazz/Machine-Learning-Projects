import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

X = cluster_data.drop(["bikeshop.name"], axis=1)
y = cluster_data["bikeshop.name"]

normalizer = MinMaxScaler()
normalizer.fit(X)

normalized_data = normalizer.transform(X)
sns.heatmap(normalized_data)

# K-Means
# -------

# Elbow plot

distortions = []
K = range(1, 10)
for k in K:
    model = KMeans(n_clusters=k).fit(normalized_data)
    distortions.append(model.inertia_)

plt.plot(K, distortions)
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title('Optimal Number of Clusters (Scree Plot/Elbow Method)')
plt.show()

# The optimal number of clusters according to the elbow method is four clusters

# Silhouette scores

silhouette_scores = []

range_n_clusters = range(2, 30)

for n_clusters in range_n_clusters:

    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(normalized_data)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_scores.append(silhouette_score(normalized_data, cluster_labels))


plt.plot(range_n_clusters, silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Optimal Number of Clusters (Silhouette Method)')
plt.show()

# The Silhouette plot shows that score keeps decreasing
# with increasing the number of clusters and is highest at two clusters.
# Thus, the optimal number of clusters is two.
