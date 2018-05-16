library(tidyverse)
library(corrplot)
library(ggthemes)
library(gridExtra)

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

# PCA
# ---

# Our goal with PCA here is to reduce the dimensionality
# of the data on the bike sales per bike shop.

# We set the scale parameter in the prcomp() function to true
# so that we don't have to do the transformations
# manually.

model <- prcomp(X, scale = TRUE)

# Visualize explained variance ratio and cumulative explained variance ratio

model.var <- model$sdev^2
explained_variance_ratio <- model.var / sum(model.var)

pca_df <- data.frame("n_components" = seq(ncol(model$x)), 
                     "explained_variance_ratio" = explained_variance_ratio,
                     "evr_cumulative" = cumsum(explained_variance_ratio))

explained_variance_p1 <- ggplot(pca_df, aes(x = n_components, y = explained_variance_ratio)) +
  geom_line() +
  scale_x_continuous(breaks = pca_df$n_components) +
  ggtitle("Proportions of Variance Explained by each Component") +
  xlab("Number of Components") +
  ylab('Explained Variance Ratio') +
  theme_tufte()

explained_variance_p2 <- ggplot(pca_df, aes(x = n_components, y = evr_cumulative)) +
  geom_line() +
  scale_x_continuous(breaks = pca_df$n_components) +
  ggtitle("Proportions of Variance Explained by each Component (Cumulative)") +
  xlab("Number of Components") +
  ylab('Cumulative Explained Variance Ratio') +
  theme_tufte()

grid.arrange(explained_variance_p1, explained_variance_p2)

# We now have 5 components which capture over 90% of the variance.

