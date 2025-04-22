# Load necessary libraries
library(caret)        # for data partitioning and metrics
library(e1071)        # for SVM
library(class)        # for KNN
library(randomForest) # for Random Forest
library(naivebayes)   # for Naive Bayes
library(rpart)        # for Decision Tree
library(ggplot2)      # for visualization
library(reshape2)     # for reshaping data

# Load dataset
dataset <- read.csv(file.choose())

# Clean and convert "Precip.Type" to binary (Rain = 1, else 0)
dataset <- na.omit(dataset)
dataset$Rain <- ifelse(dataset$Precip.Type == "rain", 1, 0)
dataset$Rain <- as.factor(dataset$Rain)

# Select relevant features + target
features <- c("Temperature..C.","Apparent.Temperature..C.", "Humidity",
              "Wind.Speed..km.h.", "Visibility..km.","Pressure..millibars.")
dataset <- dataset[, c(features, "Rain")]

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(dataset$Rain, p = 0.7, list = FALSE)
trainData <- dataset[trainIndex, ]
testData  <- dataset[-trainIndex, ]

# Function to calculate performance metrics
calculate_metrics <- function(true, pred) {
  cm <- confusionMatrix(pred, true)
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- 2 * ((precision * recall) / (precision + recall))
  error_rate <- 1 - accuracy
  return(c(accuracy, precision, recall, f1, error_rate))
}

# Initialize results dataframe
results <- data.frame(
  Model = c("KNN", "SVM", "Decision Tree", "Random Forest", "Naive Bayes"),
  Accuracy = NA, Precision = NA, Recall = NA, F1 = NA, ErrorRate = NA
)

# K-Nearest Neighbors
knn_pred <- knn(train = trainData[, -which(names(trainData) == "Rain")],
                test  = testData[, -which(names(testData) == "Rain")],
                cl    = trainData$Rain, k = 3)
results[1, 2:6] <- calculate_metrics(testData$Rain, knn_pred)

# Support Vector Machine
svm_model <- svm(Rain ~ ., data = trainData, kernel = "linear", cost = 1)
svm_pred <- predict(svm_model, testData)
results[2, 2:6] <- calculate_metrics(testData$Rain, svm_pred)

# Decision Tree
dt_model <- rpart(Rain ~ ., data = trainData, method = "class")
dt_pred <- predict(dt_model, testData, type = "class")
results[3, 2:6] <- calculate_metrics(testData$Rain, dt_pred)

# Random Forest
rf_model <- randomForest(Rain ~ ., data = trainData, ntree = 100)
rf_pred <- predict(rf_model, testData)
results[4, 2:6] <- calculate_metrics(testData$Rain, rf_pred)

# Naive Bayes
nb_model <- naive_bayes(Rain ~ ., data = trainData)
nb_pred <- predict(nb_model, testData)
results[5, 2:6] <- calculate_metrics(testData$Rain, nb_pred)

# Display results
print(results)

# Visualization (optional)
results_long <- melt(results, id.vars = "Model")
ggplot(results_long, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "stack") +
  labs(title = "Model Performance Comparison", x = "Model", y = "Metric Value") +
  theme_minimal()
