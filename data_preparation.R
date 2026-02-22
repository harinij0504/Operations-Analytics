# ============================================================================
# DATA PREPARATION
# Supply Chain On-Time Delivery Prediction Model
# ============================================================================

# Load libraries
library(tidyverse)
library(caret)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

# Load the raw data
data <- readxl::read_excel("SCM_Disruption_data.xlsx")

# Display basic info
cat("Dataset Overview:\n")
cat("  Rows:", nrow(data), "\n")
cat("  Columns:", ncol(data), "\n")
cat("  Missing values:", sum(is.na(data)), "\n\n")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================

# Check target variable distribution
cat("Target Variable Distribution:\n")
print(table(data$Std_On_Time_Delivery))
cat("  On-time rate:", round(sum(data$Std_On_Time_Delivery == 1) / nrow(data) * 100, 1), "%\n")
cat("  Late rate:", round(sum(data$Std_On_Time_Delivery == 0) / nrow(data) * 100, 1), "%\n\n")

# Check disruption impact
cat("Disruption Impact:\n")
with_disruption <- data %>% filter(Std_Has_Disruption == 1)
without_disruption <- data %>% filter(Std_Has_Disruption == 0)

late_with <- sum(with_disruption$Std_On_Time_Delivery == 0) / nrow(with_disruption) * 100
late_without <- sum(without_disruption$Std_On_Time_Delivery == 0) / nrow(without_disruption) * 100

cat("  Late rate WITH disruption:", round(late_with, 1), "%\n")
cat("  Late rate WITHOUT disruption:", round(late_without, 1), "%\n")
cat("  Difference:", round(late_with - late_without, 1), "percentage points\n\n")

# ============================================================================
# 3. DATA CLEANING
# ============================================================================

# Remove any rows with complete missing data
data <- data[rowSums(is.na(data)) < ncol(data), ]

cat("After cleaning:\n")
cat("  Rows:", nrow(data), "\n")
cat("  Missing values:", sum(is.na(data)), "\n\n")

# ============================================================================
# 4. IDENTIFY NUMERIC AND CATEGORICAL VARIABLES
# ============================================================================

numeric_vars <- c(
  "Scheduled_Lead_Time_Days",
  "Base_Lead_Time_Days",
  "Order_Weight_Kg",
  "Shipping_Cost_USD",
  "Geopolitical_Risk_Index",
  "Weather_Severity_Index"
)

categorical_vars <- c(
  "Route_Type",
  "Transportation_Mode",
  "Product_Category",
  "Mitigation_Action_Taken",
  "Disruption_Event"
)

cat("Numeric variables:", length(numeric_vars), "\n")
cat("Categorical variables:", length(categorical_vars), "\n\n")

# ============================================================================
# 5. FEATURE ENGINEERING
# ============================================================================

# Create new features
data <- data %>%
  mutate(
    # Lead time buffer (safety margin)
    Lead_Time_Buffer = Scheduled_Lead_Time_Days - Base_Lead_Time_Days,
    
    # Cost efficiency
    Shipping_Cost_per_Kg = Shipping_Cost_USD / Order_Weight_Kg,
    
    # High risk environment flag
    High_Risk_Environment = ifelse(
      Geopolitical_Risk_Index > 0.6 | Weather_Severity_Index > 7, 
      1, 0
    )
  )

cat("New features created:\n")
cat("  Lead_Time_Buffer: range [", 
    min(data$Lead_Time_Buffer, na.rm = TRUE), 
    ", ", 
    max(data$Lead_Time_Buffer, na.rm = TRUE), 
    "]\n", sep = "")
cat("  Shipping_Cost_per_Kg: range [", 
    round(min(data$Shipping_Cost_per_Kg, na.rm = TRUE), 4), 
    ", ", 
    round(max(data$Shipping_Cost_per_Kg, na.rm = TRUE), 4), 
    "]\n", sep = "")
cat("  High_Risk_Environment: proportion =", 
    round(sum(data$High_Risk_Environment) / nrow(data) * 100, 1), "%\n\n")

# ============================================================================
# 6. HANDLE CATEGORICAL VARIABLES
# ============================================================================

# One-hot encoding for categorical variables
data <- data %>%
  mutate(
    Route_Type = as.factor(Route_Type),
    Transportation_Mode = as.factor(Transportation_Mode),
    Product_Category = as.factor(Product_Category),
    Mitigation_Action_Taken = as.factor(Mitigation_Action_Taken),
    Disruption_Event = as.factor(Disruption_Event)
  )

# Create dummy variables
dummies <- dummyVars(
  ~Route_Type + Transportation_Mode + Product_Category + Mitigation_Action_Taken,
  data = data
)

data_dummies <- predict(dummies, newdata = data) %>% as.data.frame()

# Combine with original numeric data
data_final <- cbind(
  data %>% select(
    Std_On_Time_Delivery,
    Scheduled_Lead_Time_Days,
    Base_Lead_Time_Days,
    Order_Weight_Kg,
    Shipping_Cost_USD,
    Geopolitical_Risk_Index,
    Weather_Severity_Index,
    Std_Has_Disruption,
    Lead_Time_Buffer,
    Shipping_Cost_per_Kg,
    High_Risk_Environment
  ),
  data_dummies
)

cat("Final dataset for modeling:\n")
cat("  Rows:", nrow(data_final), "\n")
cat("  Columns:", ncol(data_final), "\n\n")

# ============================================================================
# 7. STANDARDIZATION (Min-Max normalization to 0-1 range)
# ============================================================================

# Identify columns to standardize (all numeric except target)
cols_to_std <- names(data_final)[names(data_final) != "Std_On_Time_Delivery"]

# Apply Min-Max normalization
preProcess_obj <- preProcess(
  data_final[, cols_to_std],
  method = c("range")  # Min-Max normalization
)

data_std <- predict(preProcess_obj, data_final[, cols_to_std])
data_std$Std_On_Time_Delivery <- data_final$Std_On_Time_Delivery

# Check standardization
cat("Standardization check (sample variables):\n")
cat("  Scheduled_Lead_Time_Days: range [", 
    round(min(data_std$Scheduled_Lead_Time_Days), 2), 
    ", ", 
    round(max(data_std$Scheduled_Lead_Time_Days), 2), 
    "]\n", sep = "")
cat("  Shipping_Cost_USD: range [", 
    round(min(data_std$Shipping_Cost_USD), 2), 
    ", ", 
    round(max(data_std$Shipping_Cost_USD), 2), 
    "]\n\n", sep = "")

# ============================================================================
# 8. CREATE TRAIN/VALIDATION/TEST SPLIT
# ============================================================================

set.seed(42)

# 40% train, 30% validation, 30% test
train_idx <- createDataPartition(
  data_std$Std_On_Time_Delivery,
  p = 0.4,
  list = FALSE
)

train_data <- data_std[train_idx, ]
remaining_data <- data_std[-train_idx, ]

val_idx <- createDataPartition(
  remaining_data$Std_On_Time_Delivery,
  p = 0.5,
  list = FALSE
)

val_data <- remaining_data[val_idx, ]
test_data <- remaining_data[-val_idx, ]

cat("Data Split:\n")
cat("  Training set:", nrow(train_data), "orders (", 
    round(nrow(train_data) / nrow(data_std) * 100), "%)\n", sep = "")
cat("  Validation set:", nrow(val_data), "orders (", 
    round(nrow(val_data) / nrow(data_std) * 100), "%)\n", sep = "")
cat("  Test set:", nrow(test_data), "orders (", 
    round(nrow(test_data) / nrow(data_std) * 100), "%)\n\n", sep = "")

# ============================================================================
# 9. SAVE PREPARED DATA
# ============================================================================

# Save for model building
save(train_data, val_data, test_data, file = "prepared_data.RData")

cat("Data preparation complete!\n")
cat("Prepared data saved to: prepared_data.RData\n")
