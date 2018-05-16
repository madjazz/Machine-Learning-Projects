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

# Scale the data with mean = 0 and sd = 1

normalized_data <- data.frame(apply(X, 2, scale))

# Hierarchical Clustering

# Since we are dealing with only numerical data
# Euclidean distance is an appropriate measure

set.seed(42)

diss <- dist(normalized_data)

model_ward <- hclust(diss, method = "ward.D")
model_ward2 <- hclust(diss, method = "ward.D2")
model_single <- hclust(diss, method = "single")
model_complete <- hclust(diss, method = "complete")
model_average <- hclust(diss, method = "average")
model_mcquitty <- hclust(diss, method = "mcquitty")
model_centroid <- hclust(diss, method = "median")
model_median <- hclust(diss, method = "centroid")

par(mfrow = c(3, 3)) 
plot(model_ward, main = "Ward.D Linkage")
plot(model_ward2, main = "Ward.D2 Linkage")
plot(model_single, main = "Single Linkage")
plot(model_complete, main = "Complete Linkage")
plot(model_average, main = "Average Linkage")
plot(model_mcquitty, main = "McQuitty Linkage")
plot(model_centroid, main = "Centroid Linkage")
plot(model_median, main = "Median Linkage")

# Similarly to the Python script we can observe that most of the stores
# have a similar sales distribution accross bike models and very few
# stores (such as store 11) have high sales across models. Thus,
# clustering shops within the cluster interval of 2-4 seems appropriate.
# Re-clustering after removing store 11 would be an interesting
# next step to differentiate better among shops.

