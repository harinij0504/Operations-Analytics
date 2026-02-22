# ============================================================================
# MODEL TRAINING
# Supply Chain On-Time Delivery Prediction Model
# Logistic Regression
# ============================================================================

library(tidyr)
library(caret)

# ============================================================================
# 1. LOAD PREPARED DATA
# ============================================================================

load("prepared_data.RData")

cat("Data loaded:\n")
cat("  Training set:", nrow(train_data), "orders\n")
cat("  Validation set:", nrow(val_data), "orders\n")
cat("  Test set:", nrow(test_data), "orders\n\n")

# ============================================================================
# 2. BUILD LOGISTIC REGRESSION MODEL
# ============================================================================

# Convert target to factor (required for logistic regression)
train_data$Std_On_Time_Delivery <- as.factor(train_data$Std_On_Time_Delivery)
val_data$Std_On_Time_Delivery <- as.factor(val_data$Std_On_Time_Delivery)
test_data$Std_On_Time_Delivery <- as.factor(test_data$Std_On_Time_Delivery)

# Define formula (exclude target variable from predictors)
formula <- Std_On_Time_Delivery ~ .

# Train logistic regression model using glm
model <- glm(
  formula,
  data = train_data,
  family = binomial(link = "logit"),
  control = glm.control(maxit = 100)
)

cat("Model Summary:\n")
print(summary(model))

# ============================================================================
# 3. EXTRACT COEFFICIENTS
# ============================================================================

coefficients_df <- as.data.frame(summary(model)$coefficients)
coefficients_df <- coefficients_df %>%
  mutate(
    Feature = rownames(coefficients_df),
    Coefficient = `Estimate`,
    AbsValue = abs(`Estimate`),
    P_Value = `Pr(>|z|)`,
    Significant = case_when(
      `Pr(>|z|)` < 0.001 ~ "***",
      `Pr(>|z|)` < 0.01 ~ "**",
      `Pr(>|z|)` < 0.05 ~ "*",
      TRUE ~ ""
    )
  ) %>%
  select(Feature, Coefficient, AbsValue, P_Value, Significant) %>%
  arrange(desc(AbsValue))

cat("\n")
cat("Top 10 Most Important Predictors:\n")
print(head(coefficients_df, 10))

cat("\nKey Insights:\n")
cat("  Strongest positive (On-Time protector):",
    coefficients_df$Feature[coefficients_df$Coefficient > 0][1],
    "(+", round(coefficients_df$Coefficient[coefficients_df$Coefficient > 0][1], 2), ")\n")
cat("  Strongest negative (Late predictor):",
    coefficients_df$Feature[coefficients_df$Coefficient < 0][1],
    "(", round(coefficients_df$Coefficient[coefficients_df$Coefficient < 0][1], 2), ")\n")
cat("  Statistically significant features:", 
    sum(coefficients_df$P_Value < 0.05), "\n\n")

# ============================================================================
# 4. MAKE PREDICTIONS ON VALIDATION SET
# ============================================================================

# Predict probabilities
val_pred_prob <- predict(model, newdata = val_data, type = "response")

# Convert probabilities to binary predictions (0.5 threshold)
val_pred_class <- ifelse(val_pred_prob > 0.5, 0, 1)
val_pred_class <- as.factor(val_pred_class)

cat("Validation Set Performance:\n")
cm_val <- confusionMatrix(val_pred_class, val_data$Std_On_Time_Delivery)
print(cm_val)

# ============================================================================
# 5. MAKE PREDICTIONS ON TEST SET
# ============================================================================

# Predict probabilities
test_pred_prob <- predict(model, newdata = test_data, type = "response")

# Convert probabilities to binary predictions (0.5 threshold)
test_pred_class <- ifelse(test_pred_prob > 0.5, 0, 1)
test_pred_class <- as.factor(test_pred_class)

cat("\n")
cat("Test Set Performance:\n")
cm_test <- confusionMatrix(test_pred_class, test_data$Std_On_Time_Delivery)
print(cm_test)

# Extract confusion matrix values
tn <- cm_test$table[1, 1]
fp <- cm_test$table[1, 2]
fn <- cm_test$table[2, 1]
tp <- cm_test$table[2, 2]

cat("\n")
cat("Confusion Matrix Breakdown:\n")
cat("  True Negatives (TN):", tn, "- Correctly predicted on-time\n")
cat("  True Positives (TP):", tp, "- Correctly predicted late\n")
cat("  False Positives (FP):", fp, "- Incorrectly predicted late\n")
cat("  False Negatives (FN):", fn, "- Incorrectly predicted on-time\n\n")

# ============================================================================
# 6. CALCULATE KEY METRICS
# ============================================================================

accuracy <- cm_test$overall['Accuracy']
sensitivity <- cm_test$byClass['Sensitivity']
specificity <- cm_test$byClass['Specificity']
precision <- cm_test$byClass['Pos Pred Value']

cat("Key Performance Metrics:\n")
cat("  Accuracy:", round(accuracy * 100, 2), "%\n")
cat("  Sensitivity (Recall):", round(sensitivity * 100, 2), "%\n")
cat("  Specificity:", round(specificity * 100, 2), "%\n")
cat("  Precision:", round(precision * 100, 2), "%\n")
cat("  False Alarm Rate:", round((fp / (tn + fp)) * 100, 2), "%\n\n")

# ============================================================================
# 7. SAVE MODEL
# ============================================================================

save(model, file = "trained_model.RData")

cat("Model training complete!\n")
cat("Trained model saved to: trained_model.RData\n")
