# Load Libraries
library(tidyverse)
library(caret)
library(corrplot)
library(ggplot2)
library(reshape2)
library(rsample)
library(ROSE)
library(Boruta)
library(FSelector)
library(glmnet)
library(pROC)
library(nnet)
library(xgboost)
library(naivebayes)
library(e1071)
library(gbm)
library(glmnet)
library(smotefamily)


# --------------------- 1. Load and Clean Data ---------------------

#  Step 1: Load Dataset
# Load dataset from the given file path
file_path <- "/Users/utkarshroy/Desktop/Spring 25/Lee projecr/U Roy/By prof./project_data.csv"
data <- read.csv(file_path, stringsAsFactors = FALSE)
table(data$Class)
# Display structure and summary of the dataset
str(data)
summary(data)

#  Step 2: Drop Columns with More Than 2000 Missing Values
# Columns with excessive missing values are removed as they might not contribute effectively to the analysis.
data_f <- data[, colSums(is.na(data)) <= 2000]
# Step 2.1: Remove rows with NA in columns that have <= 60 missing values
cols_less_than_60_na <- names(which(colSums(is.na(data_f)) <= 100 & colSums(is.na(data_f)) > 0))

# Drop rows that have any NA in those columns
if (length(cols_less_than_60_na) > 0) {
  data_f <- data_f[complete.cases(data_f[, cols_less_than_60_na]), ]
  print(paste("✅ Rows with NA in columns (<= 60 NA) removed. Remaining rows:", nrow(data_f)))
}

colSums(is.na(data_f))
#  Visualizing Missing Values Before Handling
missing_counts <- colSums(is.na(data))
missing_df <- data.frame(Feature = names(missing_counts), Count = missing_counts)
missing_df <- missing_df[order(-missing_df$Count), ]
table(data$Class)
table(data_f$Class)
#  Bar plot of missing values per column
ggplot(missing_df, aes(x = reorder(Feature, -Count), y = Count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Count of Missing Values by Column", x = "Column", y = "Count of Missing Values")

#  Pie chart of missing vs non-missing values (with percentages)
total_values <- prod(dim(data))
missing_total <- sum(is.na(data))
missing_prop <- data.frame(
  Category = c("Missing", "Non-Missing"),
  Count = c(missing_total, total_values - missing_total)
)

ggplot(missing_prop, aes(x = "", y = Count, fill = Category)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y") +
  theme_void() +
  labs(title = "Proportion of Missing vs Non-Missing Values") +
  scale_fill_manual(values = c("orange", "blue")) +
  geom_text(aes(label = paste0(round(Count / sum(Count) * 100, 2), "%")), 
            position = position_stack(vjust = 0.5))

#  Step 3: Identify Numerical and Categorical Columns
num_cols <- names(data_f)[sapply(data_f, is.numeric)]   # Numerical columns
cat_cols <- names(data_f)[sapply(data_f, is.character)] # Categorical variables

#  Convert Class Variable to Numeric (Binary Encoding: Yes = 1, No = 0)
if ("Class" %in% names(data_f)) {
  data_f$Class <- ifelse(data_f$Class == "Yes", 1, 0)
}

#  Step 4: Remove Serial Number Column (Not useful for analysis)
if ("SERIALNO" %in% names(data_f)) {
  data_f <- data_f[, !(names(data_f) %in% "SERIALNO")]
}

#  Note: No categorical encoding is needed, as the dataset is already in numerical format.
# Skipping encoding to prevent unintended modifications before correlation analysis.

#  Step 5: Define Proportion-Based Imputation Function
# This function replaces missing values in numerical columns based on their observed distribution.
impute_by_proportion <- function(df, column) {
  value_counts <- table(df[[column]], useNA = "no")
  total_count <- sum(value_counts)
  
  # If column has no valid values, use mode (most frequent value)
  if (total_count == 0 || length(value_counts) < 2) {
    mode_val <- as.numeric(names(sort(-table(df[[column]])))[1]) # Mode value
    df[[column]][is.na(df[[column]])] <- mode_val
    return(df[[column]])
  }
  
  probs <- value_counts / total_count
  
  df[[column]][is.na(df[[column]])] <- sample(
    as.numeric(names(probs)), sum(is.na(df[[column]])), replace = TRUE, prob = probs
  )
  return(df[[column]])
}

#  Step 6: Handle Special Cases (Explicit Zero Imputation)
if ("JWMNP" %in% colnames(data_f)) {
  data_f$JWMNP[is.na(data_f$JWMNP)] <- 0  # Remote workers → replace with 0 because it is zero hours
}
if ("WKWN" %in% colnames(data_f)) {
  data_f$WKWN[is.na(data_f$WKWN)] <- 0    # Too young or working less than 12 months
}

#  Step 7: Apply Proportion-Based Imputation for Other Numerical Variables
for (col in num_cols) {
  if (!col %in% c("JWMNP", "WKWN")) {  # Skip already set values
    data_f[[col]] <- as.numeric(impute_by_proportion(data_f, col))
  }
}

# Check if missing values remain
print(paste("Remaining Missing Values:", sum(is.na(data_f))))  # Should be 0

#  Step 8: Remove Zero-Variance Features
zero_var_cols <- nearZeroVar(data_f)  # Identify zero-variance features
if (length(zero_var_cols) > 0) {
  data_f <- data_f[, -zero_var_cols]
  print(paste("Removed zero-variance columns:", length(zero_var_cols)))
}

#  Step 9: Normalize Numerical Features (Scaling & Centering)
available_num_cols <- intersect(num_cols, names(data_f))
if (length(available_num_cols) > 0) {
  preProcess_values <- preProcess(data_f[, available_num_cols], method = c("center", "scale"))
  data_normalized <- predict(preProcess_values, data_f[, available_num_cols])
  data_f[available_num_cols] <- data_normalized
}

#  Step 10: Correlation Analysis (Feature Selection)
if (length(available_num_cols) > 1) {
  cor_matrix <- cor(data_f[, available_num_cols], use = "complete.obs")
  
  # Check for NaN or Inf values in correlation matrix
  if (!any(is.na(cor_matrix)) && !any(is.infinite(cor_matrix))) {
    # Generate a correlation heatmapno
    corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)
    
    # Drop highly correlated features (threshold = 0.8)
    high_corr_vars <- findCorrelation(cor_matrix, cutoff = 0.8, names = TRUE)
    if (length(high_corr_vars) > 0) {
      data_f <- data_f[, !(names(data_f) %in% high_corr_vars)]
    }
  } else {
    print("⚠ Warning: Correlation matrix contains undefined values, skipping feature selection.")
  }
}

# Bar Plots for Categorical Features
if (length(cat_cols) > 0) {
  for (col in cat_cols) {
    ggplot(data_f, aes_string(x = col)) +
      geom_bar(fill = "blue") +
      theme_minimal() +
      labs(title = paste("Bar Plot of", col), x = col, y = "Count") +
      theme(axis.text.x = element_text(angle = 90, hjust = 1))
  }
}

#  Step 11: Save the Preprocessed Dataset
write.csv(data_f, "/Users/utkarshroy/Desktop/Spring 25/Lee projecr/U Roy/project_data_cleaned.csv", row.names = FALSE)
print("✅ Preprocessing Completed: Data Saved Successfully!")

# Check final dataset dimensions
dim(data_f)

table(data_f$Class)
# ---------------------  Split + Undersample ---------------------

set.seed(123)
split <- initial_split(data_f, prop = 0.65, strata = "Class")
train <- training(split)
test <- testing(split)

write.csv(train, "/Users/utkarshroy/Desktop/Spring 25/Lee projecr/U Roy/initial_train.csv", row.names = FALSE)
write.csv(test, "/Users/utkarshroy/Desktop/Spring 25/Lee projecr/U Roy/initial_test.csv", row.names = FALSE)



# ✅ Read previously saved train and test splits
train <- read.csv("/Users/utkarshroy/Desktop/Spring 25/Lee projecr/U Roy/initial_train.csv")
test  <- read.csv("/Users/utkarshroy/Desktop/Spring 25/Lee projecr/U Roy/initial_test.csv")

# Confirm dimensions and class balance
dim(train)
dim(test)
table(train$Class)
table(test$Class)

table(train$Class)
table(test$Class)
set.seed(123)
undersampled_train <- ovun.sample(Class ~ ., data = train, method = "under", 
                                  N = min(table(train$Class)) * 2)$data
write.csv(undersampled_train, "/Users/utkarshroy/Desktop/Spring 25/Lee projecr/U Roy/train_undersampled.csv", row.names = FALSE)
table(undersampled_train$Class)

# ---------------------  Feature Selection ---------------------

# Boruta
set.seed(123)
boruta_result <- Boruta(Class ~ ., data = undersampled_train, doTrace = 2)
final_boruta <- getSelectedAttributes(boruta_result, withTentative = FALSE)
boruta_data <- undersampled_train[, c(final_boruta, "Class")]
table(boruta_data$Class)
# Info Gain
set.seed(123)
info_gain <- information.gain(Class ~ ., data = undersampled_train)
info_gain <- info_gain[order(-info_gain$attr_importance), , drop = FALSE]
top_features_ig <- rownames(info_gain[info_gain$attr_importance > 0, , drop = FALSE])
ig_data <- undersampled_train[, c(top_features_ig, "Class")]

# LASSO
x <- model.matrix(Class ~ ., undersampled_train)[, -1]
y <- undersampled_train$Class
set.seed(123)
lasso_cv <- cv.glmnet(x, y, alpha = 1, family = "binomial")
best_lambda <- lasso_cv$lambda.min
lasso_model <- glmnet(x, y, alpha = 1, lambda = best_lambda, family = "binomial")
selected_vars <- rownames(coef(lasso_model))[which(coef(lasso_model)[, 1] != 0)]
selected_vars <- setdiff(selected_vars, "(Intercept)")
lasso_data <- undersampled_train[, c(selected_vars, "Class")]

# --------------------- Evaluation Setup ---------------------
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
results_list <- list()
evaluate_model <- function(model, data, model_name = "Model") {
  preds <- predict(model, data)
  probs <- predict(model, data, type = "prob")
  cm <- confusionMatrix(preds, data$Class, positive = "Yes")
  print(paste("Confusion Matrix for", model_name))
  print(cm$table)
  
  TP <- cm$table[2,2]; TN <- cm$table[1,1]; FP <- cm$table[1,2]; FN <- cm$table[2,1]
  TPR_Yes <- TP / (TP + FN); FPR_Yes <- FP / (FP + TN)
  Precision_Yes <- TP / (TP + FP); Recall_Yes <- TPR_Yes
  F1_Yes <- 2 * ((Precision_Yes * Recall_Yes) / (Precision_Yes + Recall_Yes))
  
  TPR_No <- TN / (TN + FP); FPR_No <- FN / (FN + TP)
  Precision_No <- TN / (TN + FN); Recall_No <- TPR_No
  F1_No <- 2 * ((Precision_No * Recall_No) / (Precision_No + Recall_No))
  
  auc_val <- auc(roc(data$Class, probs$Yes))
  MCC <- as.numeric((TP * TN - FP * FN)) / sqrt(as.numeric((TP + FP)) * as.numeric((TP + FN)) * as.numeric((TN + FP)) * as.numeric((TN + FN)))
  Kappa <- cm$overall["Kappa"]
  
  perf <- data.frame(
    Model = rep(model_name, 3),
    Class = c("Class No", "Class Yes", "Wt. Average"),
    TPR = c(TPR_No, TPR_Yes, mean(c(TPR_No, TPR_Yes))),
    FPR = c(FPR_No, FPR_Yes, mean(c(FPR_No, FPR_Yes))),
    Precision = c(Precision_No, Precision_Yes, mean(c(Precision_No, Precision_Yes))),
    Recall = c(Recall_No, Recall_Yes, mean(c(Recall_No, Recall_Yes))),
    F_measure = c(F1_No, F1_Yes, mean(c(F1_No, F1_Yes))),
    ROC = rep(auc_val, 3),
    MCC = rep(MCC, 3),
    Kappa = rep(Kappa, 3),
    Status = rep("", 3)
  )
  #______
  if (!is.na(perf$TPR[1]) && !is.na(perf$TPR[2])) {
    if (perf$TPR[2] >= 0.84 && perf$TPR[1] >= 0.82) {
      perf$Status[1:2] <- "Extra Credit"
    } else if (perf$TPR[2] >= 0.81 && perf$TPR[1] >= 0.79) {
      perf$Status[1:2] <- "Meets Criteria"
    } else {
      perf$Status[1:2] <- "Below Criteria"
    }
  } else {
    perf$Status[1:2] <- "Below Criteria"
  }
  #_________
  results_list[[model_name]] <<- perf
  cat("\nPerformance Table for", model_name, "\n")
  print(perf)
}

# ---------------------  Train Classifiers (Boruta-based) ---------------------
#boruta_data$Class <- factor(boruta_data$Class, levels = c(0, 1), labels = c("No", "Yes"))
boruta_data_train <-  train[, c(final_boruta, "Class")]
boruta_data_train$Class <- factor(boruta_data_train$Class, levels = c(0, 1), labels = c("No", "Yes"))
boruta_test <- test[, c(final_boruta, "Class")]
boruta_test$Class <- factor(boruta_test$Class, levels = c(0, 1), labels = c("No", "Yes"))
test$Class <- factor(test$Class, levels = c(0, 1), labels = c("No", "Yes"))
table(boruta_data_train$Class)
table(boruta_test$Class)
table(test$Class)
table(train$Class)




# Logistic Regression

set.seed(123)
log_boruta <- train(
  Class ~ ., data = boruta_data_train,
  method = "glm",
  family = "binomial",
  trControl = ctrl, metric = "ROC"
)
evaluate_model(log_boruta, boruta_data_train, model_name = "Logistic Regression - Boruta (Train)")
evaluate_model(log_boruta, boruta_test, model_name = "Logistic Regression - Boruta (Test)")




#Elastic Net Regularization: GLMNet 

glmnet_grid <- expand.grid(
  alpha = seq(0, 1, length = 5),
  lambda = 10^seq(-3, 1, length = 5)
)
set.seed(123)
glmnet_boruta <- train(
  Class ~ ., data = boruta_data_train,
  method = "glmnet",
  trControl = ctrl,
  metric = "ROC",
  preProcess = c("center", "scale"),
  tuneGrid = glmnet_grid
)
evaluate_model(glmnet_boruta, boruta_data_train, model_name = "GLMNet - Boruta (Train)")
evaluate_model(glmnet_boruta, boruta_test, model_name = "GLMNet - Boruta (Test)")


# SVM
set.seed(123)
svm_boruta <- train(Class ~ ., data = boruta_data_train, method = "svmRadial",
                    trControl = ctrl, metric = "ROC", preProcess = c("center", "scale"))
evaluate_model(svm_boruta, boruta_data_train, model_name = "SVM - Boruta")
evaluate_model(svm_boruta, boruta_test, model_name = "Test SVM - Boruta")


# Neural Network
nnet_grid <- expand.grid(
  size = c(1, 3, 5),
  decay = c(0, 0.1, 0.5)
)
set.seed(123)
nnet_boruta <- train(
  Class ~ ., data = boruta_data_train,
  method = "nnet",
  trControl = ctrl,
  metric = "ROC",
  preProcess = c("center", "scale"),
  trace = FALSE,
  tuneGrid = nnet_grid
)
evaluate_model(nnet_boruta, boruta_data_train, model_name = "Neural Net - Boruta (Train)")
evaluate_model(nnet_boruta, boruta_test, model_name = "Neural Net - Boruta (Test)")




# Naive Bayes
set.seed(123)
nb_grid <- expand.grid(
  laplace = 0:1,
  usekernel = c(TRUE, FALSE),
  adjust = c(0.5, 1, 1.5)
)
set.seed(123)
nb_boruta <- train(
  Class ~ ., data = boruta_data_train,
  method = "naive_bayes",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = nb_grid
)
evaluate_model(nb_boruta, boruta_data_train, model_name = "Naive Bayes - Boruta (Train)")
evaluate_model(nb_boruta, boruta_test, model_name = "Naive Bayes - Boruta (Test)")


#KNN
knn_grid <- expand.grid(k = c(3, 5, 7, 9))
set.seed(123)
knn_boruta <- train(
  Class ~ ., data = boruta_data_train,
  method = "knn",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = knn_grid
)
evaluate_model(knn_boruta, boruta_data_train, model_name = "KNN - Boruta (Train)")
evaluate_model(knn_boruta, boruta_test, model_name = "KNN - Boruta (Test)")


# ---------------------  Train Classifiers (Info Gain) ---------------------

ig_data <-  train[, c(top_features_ig, "Class")]
# Ensure Class is a factor with correct levels
ig_data$Class <- factor(ig_data$Class, levels = c(0, 1), labels = c("No", "Yes"))
ig_test <- test[, c(top_features_ig, "Class")]
table(ig_test$Class)
table(test$Class)


# 1. Logistic Regression
set.seed(123)
log_ig <- train(
  Class ~ ., data = ig_data,
  method = "glm", family = "binomial",
  trControl = ctrl, metric = "ROC"
)
evaluate_model(log_ig, ig_data, model_name = "Logistic Regression - Info Gain (Train)")
evaluate_model(log_ig, ig_test, model_name = "Logistic Regression - Info Gain (Test)")


# 2. Elastic Net Regularization: GLMNet 
glmnet_grid <- expand.grid(
  alpha = seq(0, 1, length = 5),
  lambda = 10^seq(-2, 1, length = 5)
)
set.seed(123)
glmnet_ig <- train(
  Class ~ ., data = ig_data,
  method = "glmnet",
  trControl = ctrl,
  metric = "ROC",
  preProcess = c("center", "scale"),
  tuneGrid = glmnet_grid
)
evaluate_model(glmnet_ig, ig_data, model_name = "GLMNet - Info Gain (Train)")
evaluate_model(glmnet_ig, ig_test, model_name = "GLMNet - Info Gain (Test)")


# 3. Support Vector Machine (SVM)
set.seed(123)
svm_grid <- expand.grid(
  sigma = c(0.01, 0.015, 0.02),
  C = c(0.5, 1, 2)
)
set.seed(123)
svm_ig <- train(
  Class ~ ., data = ig_data,
  method = "svmRadial",
  trControl = ctrl,
  metric = "ROC",
  preProcess = c("center", "scale"),
  tuneGrid = svm_grid
)
evaluate_model(svm_ig, ig_data, model_name = "SVM - Info Gain (Train)")
evaluate_model(svm_ig, ig_test, model_name = "SVM - Info Gain (Test)")

# 4. Neural Network (NNet)
set.seed(123)
nnet_ig <- train(Class ~ ., data = ig_data, method = "nnet", trace = FALSE,
                 trControl = ctrl, metric = "ROC", preProcess = c("center", "scale"))
evaluate_model(nnet_ig, ig_data, model_name = "Neural Net - Info Gain")
evaluate_model(nnet_ig, ig_test, model_name = "Neural Net - Info Gain")



# 5. Naive Bayes
nb_grid <- expand.grid(
  laplace = 0:1,
  usekernel = c(TRUE, FALSE),
  adjust = c(0.5, 1, 1.5)
)
set.seed(123)
nb_ig <- train(
  Class ~ ., data = ig_data,
  method = "naive_bayes",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = nb_grid
)
evaluate_model(nb_ig, ig_data, model_name = "Naive Bayes - Info Gain (Train)")
evaluate_model(nb_ig, ig_test, model_name = "Naive Bayes - Info Gain (Test)")

# 6. KNN
knn_grid <- expand.grid(k = c(3, 5, 7, 9))
set.seed(123)
knn_ig <- train(
  Class ~ ., data = ig_data,
  method = "knn",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = knn_grid
)
evaluate_model(knn_ig, ig_data, model_name = "KNN - Info Gain (Train)")
evaluate_model(knn_ig, ig_test, model_name = "KNN - Info Gain (Test)")

# ---------------------  Train Classifiers (LASSO) ---------------------

# Ensure Class is a factor with correct levels
lasso_data <-  train[, c(selected_vars, "Class")]
lasso_data$Class <- factor(lasso_data$Class, levels = c(0, 1), labels = c("No", "Yes"))
lasso_test <- test[, c(selected_vars, "Class")]
table(lasso_data$Class)
table(test$Class)


# 1. Logistic Regression
set.seed(123)
log_lasso <- train(
  Class ~ ., data = lasso_data,
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  metric = "ROC"
)
evaluate_model(log_lasso, lasso_data, model_name = "Tuned Logistic Regression - LASSO (Train)")
evaluate_model(log_lasso, lasso_test, model_name = "Tuned Logistic Regression - LASSO (Test)")

# 2. Elastic Net Regularization: GLMNet 

set.seed(123)
glmnet_grid <- expand.grid(
  alpha = seq(0, 1, length.out = 5),
  lambda = 10^seq(-4, 1, length.out = 5)
)

set.seed(123)
glmnet_lasso <- train(
  Class ~ ., data = lasso_data,
  method = "glmnet",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = glmnet_grid,
  preProcess = c("center", "scale")
)
evaluate_model(glmnet_lasso, lasso_data, model_name = "Tuned GLMNet - LASSO (Train)")
evaluate_model(glmnet_lasso, lasso_test, model_name = "Tuned GLMNet - LASSO (Test)")

# 3. Support Vector Machine (SVM)
set.seed(123)
svm_grid <- expand.grid(
  sigma = c(0.01, 0.015, 0.02),
  C = c(0.5, 1, 2)
)

set.seed(123)
svm_lasso <- train(
  Class ~ ., data = lasso_data,
  method = "svmRadial",
  trControl = ctrl,
  metric = "ROC",
  preProcess = c("center", "scale"),
  tuneGrid = svm_grid
)
evaluate_model(svm_lasso, lasso_data, model_name = "Tuned SVM - LASSO (Train)")
evaluate_model(svm_lasso, lasso_test, model_name = "Tuned SVM - LASSO (Test)")

# 4. Neural Network (NNet) HERE
set.seed(123)
nnet_lasso <- train(Class ~ ., data = lasso_data, method = "nnet", trace = FALSE,
                    trControl = ctrl, metric = "ROC", preProcess = c("center", "scale"))
evaluate_model(nnet_lasso, lasso_data, model_name = "Neural Net - LASSO")
evaluate_model(nnet_lasso, lasso_test, model_name = "Neural Net - Lasso")



# 5. Naive Bayes
set.seed(123)
nb_grid <- expand.grid(
  laplace = 0:1,
  usekernel = c(TRUE, FALSE),
  adjust = c(0.5, 1, 1.5)
)

set.seed(123)
nb_lasso <- train(
  Class ~ ., data = lasso_data,
  method = "naive_bayes",
  trControl = ctrl,
  tuneGrid = nb_grid,
  metric = "ROC"
)
evaluate_model(nb_lasso, lasso_data, model_name = "Tuned Naive Bayes - LASSO (Train)")
evaluate_model(nb_lasso, lasso_test, model_name = "Tuned Naive Bayes - LASSO (Test)")

# 6. KNN
set.seed(123)
knn_grid <- expand.grid(
  k = seq(3, 11, 2)  # try odd values to avoid ties
)

set.seed(123)
knn_lasso <- train(
  Class ~ ., data = lasso_data,
  method = "knn",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = knn_grid
)
evaluate_model(knn_lasso, lasso_data, model_name = "Tuned KNN - LASSO (Train)")
evaluate_model(knn_lasso, lasso_test, model_name = "Tuned KNN - LASSO (Test)")

#---------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------- SMOTE Balancing -------------------------

# Install smotefamily if not already installed
if (!require(smotefamily)) install.packages("smotefamily")   



train$Class <- as.numeric(as.character(train$Class))

# Apply SMOTE with sufficient duplication to reach ~424 cases for the minority class.
minority_n <- sum(train$Class == 1)
target_n <- 424
dup_size <- round((target_n - minority_n) / minority_n, 1)  

set.seed(123)
smote_result <- SMOTE(
  X = train[, -which(names(train) == "Class")],
  target = train$Class,
  K = 4,
  dup_size = dup_size
)

# Combining original and synthetic data
smote_data <- smote_result$data

# Convert the class back to factor with labels
smote_data$Class <- as.factor(ifelse(smote_data$class == 1, "Yes", "No"))
smote_data$class <- NULL

# Sub-sampling to leave exactly the double of the minimum of each class
min_class_count <- min(table(smote_data$Class))

balanced_smote_data <- smote_data %>%
  group_by(Class) %>%
  sample_n(min_class_count, replace = FALSE) %>%
  ungroup()

# Check final class balance
table(balanced_smote_data$Class)

# Save to CSV
#write.csv(balanced_smote_data, "/Users/utkarshroy/Desktop/Spring 25/Lee projecr/U Roy/initial_test.csv", row.names = FALSE)

# ------------------------- Feature Selection -------------------------
# Boruta
set.seed(123)
boruta_result <- Boruta(Class ~ ., data = balanced_smote_data, doTrace = 2)
final_boruta_s <- getSelectedAttributes(boruta_result, withTentative = FALSE)
boruta_data_smote <- balanced_smote_data[, c(final_boruta_s, "Class")]

# Info Gain
set.seed(123)
info_gain <- information.gain(Class ~ ., data = balanced_smote_data)
info_gain <- info_gain[order(-info_gain$attr_importance), , drop = FALSE]
top_features_ig_s <- rownames(info_gain[info_gain$attr_importance > 0, , drop = FALSE])
info_gain_data_smote <- balanced_smote_data[, c(top_features_ig_s, "Class")]

# LASSO
x <- model.matrix(Class ~ ., balanced_smote_data)[, -1]
y <- balanced_smote_data$Class
set.seed(123)
lasso_cv <- cv.glmnet(x, y, alpha = 1, family = "binomial")
best_lambda <- lasso_cv$lambda.min
lasso_model <- glmnet(x, y, alpha = 1, lambda = best_lambda, family = "binomial")
selected_vars_s <- rownames(coef(lasso_model))[which(coef(lasso_model)[, 1] != 0)]
selected_vars_s <- setdiff(selected_vars_s, "(Intercept)")
lasso_data_smote <- balanced_smote_data[, c(selected_vars_s, "Class")]


# ------------------------- Prepare Data -------------------------
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# ------------------------- Boruta-based Classifiers (SMOTE) -------------------------
boruta_train_s <- train[, c(final_boruta_s, "Class")]
boruta_train_s$Class <- factor(boruta_train_s$Class, levels = c(0, 1), labels = c("No", "Yes"))
boruta_test_s <- test[, c(final_boruta_s, "Class")]
#test$Class <- factor(test$Class, levels = c(0, 1), labels = c("No", "Yes"))
table(boruta_train_s$Class)
table(boruta_test_s$Class)
table(test$Class)

# Logistic Regression
set.seed(123)
log_boruta <- train(
  Class ~ ., data = boruta_train_s,
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  metric = "ROC"
)
evaluate_model(log_boruta, boruta_train_s, model_name = "Tuned Logistic Regression - Boruta - SMOTE (Train)")
evaluate_model(log_boruta, boruta_test_s, model_name = "Tuned Logistic Regression - Boruta - SMOTE (Test)")


# Elastic Net Regularization: GLMNet 
set.seed(123)



glmnet_grid <- expand.grid(
  alpha = seq(0, 1, length = 5),
  lambda = 10^seq(-6, 0, length = 10)
)

set.seed(123)
glmnet_boruta_tuned <- train(
  Class ~ ., data = boruta_train_s,
  method = "glmnet",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = glmnet_grid,
  preProcess = c("center", "scale")
)

evaluate_model(glmnet_boruta_tuned, boruta_train_s, model_name = "Tuned GLMNet - Boruta - SMOTE (Train)")
evaluate_model(glmnet_boruta_tuned, boruta_test_s, model_name = "Tuned GLMNet - Boruta - SMOTE (Test)")


# SVM
library(kernlab)
sigma_est <- sigest(Class ~ ., data = boruta_train_s)
print(sigma_est)


svm_grid <- expand.grid(
  sigma = c(0.012, 0.017, 0.0295),
  C = c(0.25, 0.5, 100, 20, 40, 60, 100)  
)

set.seed(123)
svm_boruta_tuned <- train(
  Class ~ ., data = boruta_train_s,
  method = "svmRadial",
  trControl = ctrl,
  metric = "ROC",
  preProcess = c("center", "scale"),
  tuneGrid = svm_grid
)

evaluate_model(svm_boruta_tuned, boruta_train_s, model_name = "SVM - Boruta - SMOTE")
evaluate_model(svm_boruta_tuned, boruta_test_s, model_name = "Test SVM - Boruta SMOTE")


# Neural Network
nnet_grid <- expand.grid(
  size = c(1, 3, 5),
  decay = c(0.1, 0.5, 1, 2)
)

set.seed(123)
nnet_boruta_tuned <- train(
  Class ~ ., data = boruta_train_s,
  method = "nnet",
  trControl = ctrl,
  tuneGrid = nnet_grid,
  metric = "ROC",
  preProcess = c("center", "scale"),
  trace = FALSE
)

evaluate_model(nnet_boruta_tuned, boruta_train_s, model_name = "Tuned Neural Net - Boruta - SMOTE (Train)")
evaluate_model(nnet_boruta_tuned, boruta_test_s, model_name = "Tuned Neural Net - Boruta - SMOTE (Test)")

# Naive Bayes
set.seed(123)
nb_boruta <- train(Class ~ ., data = boruta_train_s, method = "naive_bayes",
                   trControl = ctrl, metric = "ROC")
evaluate_model(nb_boruta, boruta_train_s, model_name = "Naive Bayes - Boruta - SMOTE")
evaluate_model(nb_ig, boruta_test_s, model_name = "Naive Bayes - Boruta - SMOTE")

#KNN
set.seed(123)
knn_grid <- expand.grid(k = c(3, 5, 7, 9, 11))

set.seed(123)
knn_boruta_tuned <- train(
  Class ~ ., data = boruta_train_s,
  method = "knn",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = knn_grid
)

evaluate_model(knn_boruta_tuned, boruta_train_s, model_name = "Tuned KNN - Boruta - SMOTE (Train)")
evaluate_model(knn_boruta_tuned, boruta_test_s, model_name = "Tuned KNN - Boruta - SMOTE (Test)")


# ------------------------- Info Gain-based Classifiers (SMOTE) -------------------------
ig_train_s <- train[, c(top_features_ig_s, "Class")]
ig_train_s$Class <- factor(ig_train_s$Class, levels = c(0, 1), labels = c("No", "Yes"))
ig_test <- test[, c(top_features_ig_s, "Class")]
table(ig_test$Class)
table(test$Class)

# 1. Logistic Regression
set.seed(123)
log_ig <- train(
  Class ~ ., data = ig_train_s,
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  metric = "ROC"
)
evaluate_model(log_ig, ig_train_s, model_name = "Tuned Logistic Regression - Info Gain - SMOTE (Train)")
evaluate_model(log_ig, ig_test, model_name = "Tuned Logistic Regression - Info Gain - SMOTE (Test)")


# 2. Elastic Net Regularization: GLMNet 

set.seed(123)
glmnet_grid <- expand.grid(
  alpha = seq(0, 1, length = 5),
  lambda = 10^seq(-6, 0, length = 10)
)
set.seed(123)
glmnet_ig_tuned <- train(
  Class ~ ., data = ig_train_s,
  method = "glmnet",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = glmnet_grid,
  preProcess = c("center", "scale")
)
evaluate_model(glmnet_ig_tuned, ig_train_s, model_name = "Tuned GLMNet - Info Gain - SMOTE (Train)")
evaluate_model(glmnet_ig_tuned, ig_test, model_name = "Tuned GLMNet - Info Gain - SMOTE (Test)")

# 3. Support Vector Machine (SVM)
set.seed(123)
library(kernlab)
sigma_est <- as.numeric(sigest(Class ~ ., data = ig_train_s))
svm_grid <- expand.grid(
  sigma = sigma_est,
  C = c(0.25, 0.5, 1, 5, 10)
)
set.seed(123)
svm_ig_tuned <- train(
  Class ~ ., data = ig_train_s,
  method = "svmRadial",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = svm_grid,
  preProcess = c("center", "scale")
)
evaluate_model(svm_ig_tuned, ig_train_s, model_name = "Tuned SVM - Info Gain - SMOTE (Train)")
evaluate_model(svm_ig_tuned, ig_test, model_name = "Tuned SVM - Info Gain - SMOTE (Test)")

# 4. Neural Network (NNet)
set.seed(123)
nnet_grid_ig <- expand.grid(
  size = c(1, 3, 5),
  decay = c(0.1, 0.5, 1, 2)
)
set.seed(123)
nnet_ig_tuned <- train(
  Class ~ ., data = ig_train_s,
  method = "nnet",
  trControl = ctrl,
  tuneGrid = nnet_grid_ig,
  metric = "ROC",
  preProcess = c("center", "scale"),
  trace = FALSE
)
evaluate_model(nnet_ig_tuned, ig_train_s, model_name = "Tuned Neural Net - Info Gain - SMOTE (Train)")
evaluate_model(nnet_ig_tuned, ig_test, model_name = "Tuned Neural Net - Info Gain - SMOTE (Test)")

# 5. Naive Bayes
set.seed(123)
tuned_nb_grid <- expand.grid(
  laplace = 0:1,
  usekernel = c(TRUE, FALSE),
  adjust = c(0.5, 1, 1.5)
)
set.seed(123)
nb_ig_tuned <- train(
  Class ~ ., data = ig_train_s,
  method = "naive_bayes",
  trControl = ctrl,
  tuneGrid = tuned_nb_grid,
  metric = "ROC"
)
evaluate_model(nb_ig_tuned, ig_train_s, model_name = "Tuned Naive Bayes - Info Gain - SMOTE (Train)")
evaluate_model(nb_ig_tuned, ig_test, model_name = "Tuned Naive Bayes - Info Gain - SMOTE (Test)")

# 6. KNN
set.seed(123)
knn_grid <- expand.grid(k = c(3, 5, 7, 9, 11))
set.seed(123)
knn_ig_tuned <- train(
  Class ~ ., data = ig_train_s,
  method = "knn",
  trControl = ctrl,
  tuneGrid = knn_grid,
  metric = "ROC"
)
evaluate_model(knn_ig_tuned, ig_train_s, model_name = "Tuned KNN - Info Gain - SMOTE (Train)")
evaluate_model(knn_ig_tuned, ig_test, model_name = "Tuned KNN - Info Gain - SMOTE (Test)")

# ------------------------- LASSO-based Classifiers (SMOTE) -------------------------
lasso_train_s <- train[, c(selected_vars_s, "Class")]
lasso_train_s$Class <- factor(lasso_train_s$Class, levels = c(0, 1), labels = c("No", "Yes"))
lasso_test <- test[, c(selected_vars_s, "Class")]
table(lasso_test$Class)
table(test$Class)

# 1. Logistic Regression
set.seed(123)
log_lasso <- train(
  Class ~ ., data = lasso_train_s,
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  metric = "ROC"
)
evaluate_model(log_lasso, lasso_train_s, model_name = "Tuned Logistic Regression - LASSO - SMOTE")
evaluate_model(log_lasso, lasso_test, model_name = "Tuned Logistic Regression - LASSO - SMOTE (Test)")

# 2. Elastic Net Regularization: GLMNet 

set.seed(123)
glmnet_lasso <- train(Class ~ ., data = lasso_train_s,
                      method = "glmnet",
                      trControl = ctrl,
                      metric = "ROC",
                      preProcess = c("center", "scale"))  

evaluate_model(glmnet_lasso, lasso_train_s, model_name = "GLMNet - LASSO - SMOTE")
evaluate_model(glmnet_lasso, lasso_test, model_name = "GLMNet - Lasso - SMOTE")

# 3. Support Vector Machine (SVM)
set.seed(123)
svm_lasso <- train(Class ~ ., data = lasso_train_s, method = "svmRadial",
                   trControl = ctrl, metric = "ROC", preProcess = c("center", "scale"))
evaluate_model(svm_lasso, lasso_train_s, model_name = "SVM - LASSO - SMOTE")
evaluate_model(svm_lasso, lasso_test, model_name = "Test SVM - Lasso - SMOTE")

# 4. Neural Network (NNet) 
nnet_grid_v2 <- expand.grid(
  size = c(1, 2, 3),           # hyper parameters
  decay = c(0.1, 0.5, 1, 2)    # hyper parameters
)

set.seed(123)
nnet_lasso_tuned_v2 <- train(Class ~ ., data = lasso_train_s,
                             method = "nnet",
                             trControl = ctrl,
                             tuneGrid = nnet_grid_v2,
                             metric = "ROC",
                             preProcess = c("center", "scale"),
                             trace = FALSE)

evaluate_model(nnet_lasso_tuned_v2, lasso_train_s, model_name = "Neural Net (tuned v2) - LASSO - SMOTE")
evaluate_model(nnet_lasso_tuned_v2, lasso_test, model_name = "Neural Net - Lasso - SMOTE")


# 5. Naive Bayes
set.seed(123)
nb_lasso <- train(Class ~ ., data = lasso_train_s, method = "naive_bayes",
                  trControl = ctrl, metric = "ROC")
evaluate_model(nb_lasso, lasso_train_s, model_name = "Naive Bayes - LASSO - SMOTE")
evaluate_model(nb_lasso, lasso_test, model_name = "Naive Bayes - Lasso - SMOTE")


# 6. KNN
set.seed(123)
knn_grid <- expand.grid(k = c(3, 5, 7, 9, 11))
set.seed(123)
knn_lasso_tuned <- train(
  Class ~ ., data = lasso_train_s,
  method = "knn",
  tuneGrid = knn_grid,
  trControl = ctrl,
  metric = "ROC"
)
evaluate_model(knn_lasso_tuned, lasso_train_s, model_name = "Tuned KNN - LASSO - SMOTE")
evaluate_model(knn_lasso_tuned, lasso_test, model_name = "Tuned KNN - LASSO - SMOTE (Test)")





write.csv(results_list, "/Users/utkarshroy/Desktop/Spring 25/Lee projecr/U Roy/result_list.csv", row.names = FALSE)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# 🔍Select Top 2 Test Models Based on TPR for Class = "Yes"
# ranks models by True Positive Rate (TPR) to identify the most effective classifiers
# at correctly predicting the minority class in the test set.


#SVM- Boruta-SMOTE
evaluate_model(svm_boruta_tuned, boruta_test_s, model_name = "Test SVM - Boruta SMOTE")


# SVM- LASSO- SMOTE
evaluate_model(svm_lasso, lasso_test, model_name = "Test SVM - Lasso - SMOTE")

#KNN-Infor-Gain-SMOTE
evaluate_model(knn_ig_tuned, ig_test, model_name = "Tuned KNN - Info Gain - SMOTE (Test)")
