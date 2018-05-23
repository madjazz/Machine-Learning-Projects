# Vanilla Logistic Regression with Train-Test Split and Cross-Validation
# ----------------------------------------------------------------------

setwd("~/Documents/Education/Machine-Learning-Projects/Supervised-Learning/")
set.seed(42)

# Load Libraries
# --------------

library(tidyverse)
library(ggthemes)
library(gridExtra)
library(caret)
library(nnet)

# Import and Explore Data
# -----------------------

df <- read.csv("winequality-white.csv", sep = ";")
str(df)

# Check Response Variable Distribution
# ------------------------------------

table(df$quality)
df$quality <- as.factor(df$quality)

# Verdict: Classes are highly unbalanced
# Solution: Undersampling of minority classes since observations < 10000

# Check distributions of predictor variables
# ------------------------------------------

plotter <- function(df, col) {
  
  title <- names(df)[col]
  
  p <- ggplot(df, aes(x = df[, col])) + 
    geom_histogram(stat="count") +
    ggtitle(title) +
    xlab("") +
    ylab("Count")
    theme_tufte()
  
  return(p)
}

plots <- lapply(seq(1:ncol(df)), function(x)(plotter(df, x)))
grid.arrange(grobs = plots, ncol = 2)

# Verdict: The "chlorides" variable is unbalanced.
# Solution: Correct with log-transformation

df$chlorides <- log10(df$chlorides)

plotter(df, 5)

# Normalize Data
# --------------

y <- df$quality
X <- select(df, -quality)
df_scaled <- data.frame(scale(X), y)

# Perform Cross Validation with Vanilla Logistic Regression
# ---------------------------------------------------------

idx <- createDataPartition(df_scaled$y, p = 0.7, list = FALSE)
training_data <- df_scaled[idx,]
testing_data <- df_scaled[-idx,]

cross_validation <- trainControl(method = "repeatedcv", # Cross-validation method
                                 number = 10, # Number of k-folds
                                 repeats = 10, # Number of repetition
                                 verboseIter = FALSE, # Display text
                                 sampling = "smote" # Balance classes
) 

model.fit <- train(y ~ ., # Specify model
                   data = training_data, # Use training data
                   method = "multinom", # Model: Multinomial logistic regression (nnet package)
                   trControl = cross_validation # Use pre-defined training control from above
) 

model.predict <- predict(model.fit, # Use trained model
                         newdata = testing_data # Pass test data to function
                         )

model.fit