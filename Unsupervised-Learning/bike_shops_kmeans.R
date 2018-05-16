library(tidyverse)
library(corrplot)
library(ggthemes)
library(cluster)

# Set working directory!

# Load data in memory
# -------------------

customers <- read.csv("customers_bikeshops.csv")
orders <- read.csv("orders.csv")
products <- read.csv("products_bikes.csv")

# Merge datasets
# --------------

orders_products <- inner_join(orders, products, by = c("product.id" = "bike.id"))
bike_data <- inner_join(customers, orders_products, by = c("bikeshop.id" = "customer.id"))

# Group and Reshape Data
# ----------------------

cluster_data <- group_by(bike_data, bikeshop.name, model) %>%
  summarize(quantity = sum(quantity), price = sum(price)) %>%
  mutate(sales = quantity * price) %>%
  select(bikeshop.name, model, sales) %>%
  dcast(... ~ model, value.var = "sales")

X <- select(cluster_data, -1)
y <- select(cluster_data, 1)

# Fill NAs with 0s (NAs imply zero sales)
X <- apply(X, 2, as.numeric)
X[is.na(X)] <- 0

# Normalize Data
# --------------

# Scale the data with MinMax method

normalize <- function(x) {
  x <- (x-min(x))/(max(x)-min(x))
  return(x)
}

normalized_data <- apply(X, 2, normalize)
corrplot(normalized_data)

# K-Means
# -------

set.seed(42)

# Elbow plot

n_clusters <- seq(2, 11, 1)
distortions <- sapply(n_clusters, function(x)(kmeans(normalized_data, x, iter = 50)$tot.withinss))
elbow_df <- data.frame("n_clusters" = n_clusters, "distortion" = distortions)

elbow_p <- ggplot(elbow_df, aes(x = n_clusters, y = distortion)) +
  geom_line() +
  scale_x_continuous(breaks = n_clusters) +
  ggtitle("Optimal Number of Clusters (Scree Plot/Elbow Method)") +
  xlab("Number of Clusters") +
  ylab('Distortion') +
  theme_tufte()

elbow_p

# Silhouette plot

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters

silhouette_score <- function(x) {
  data <- as.matrix(dist(normalized_data))
  labels <- kmeans(normalized_data, x, iter = 50)$cluster
  result <- mean(silhouette(labels, dmatrix = data)[,3])
  return(result)
}

silhouette_avgs <- sapply(n_clusters, function(x)(silhouette_score(x)))
silhouette_df <- data.frame("n_clusters" = n_clusters, "silhouette_score" = silhouette_avgs)

silhouette_p <- ggplot(silhouette_df, aes(x = n_clusters, y = silhouette_score)) +
  geom_line() +
  scale_x_continuous(breaks = n_clusters) +
  ggtitle("Optimal Number of Clusters (Silhouette Method)") +
  xlab("Number of Clusters") +
  ylab('Silhouette Score') +
  theme_tufte()

silhouette_p

# The Silhouette plot shows that score keeps decreasing
# with increasing the number of clusters and is highest at two clusters.
# Thus, the optimal number of clusters is two.