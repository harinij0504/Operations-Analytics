# Supply Chain On-Time Delivery Prediction Model

## Project Overview

This project builds a logistic regression model to predict which international shipments will arrive late at booking time (2-3 weeks before shipment). The model achieves **98.7% accuracy** in identifying high-risk orders, enabling proactive customer communication and selective prevention strategies.

## Problem Statement

- **Challenge:** 12.9% of shipments (1,290 per 10,000) arrive late, costing $7,493 per late delivery
- **Root Cause:** Disruption events (port closures, geopolitical conflicts, severe weather) create a 95.3 percentage-point increase in late rates
- **Goal:** Identify high-risk orders at booking time to enable cost-justified prevention decisions

## Model Performance

- **Accuracy:** 98.70%
- **Sensitivity:** 94.29% (catches 363 of 385 late orders)
- **Specificity:** 99.35% (correctly identifies 2,598 of 2,615 on-time orders)
- **False Alarm Rate:** 0.65%

## Financial Impact

- **Prevention Cost:** $6,078 per re-routed order
- **Late Delivery Loss:** $7,493 per late order
- **Profit per Prevented Order:** $1,415
- **Annual Value:** $9.8 million (with 10,000 orders/month)

## Top Predictors (by coefficient strength)

1. **Scheduled Lead Time** (+88.11) - STRONGEST: Conservative promises protect delivery
2. **Base Lead Time** (-82.03) - Complex routes inherently risky
3. **Has Disruption** (-11.18) - Disruption events strongly predict delays

## Dataset

- **Total Orders:** 10,000 international shipments
- **Variables:** 42 (after preparation)
- **Data Split:** 40% training, 30% validation, 30% test
- **Target Variable:** On_Time_Delivery (binary: 1=on-time, 0=late)

## Files in This Repository

- `data_preparation.R` - Data cleaning, standardization, feature engineering
- `model_training.R` - Logistic regression model building
- `model_evaluation.R` - Performance evaluation and financial analysis
- `README.md` - This file

## Usage

1. **Prepare Data:**
   ```R
   source("data_preparation.R")
   ```

2. **Train Model:**
   ```R
   source("model_training.R")
   ```

3. **Evaluate Results:**
   ```R
   source("model_evaluation.R")
   ```

## Requirements

- R 3.6+
- tidyr
- caret
- ggplot2

## Recommendations

1. Deploy model to flag high-risk orders at booking time
2. Implement proactive customer communication (zero cost, high value)
3. Selectively re-route high-value orders (profitable)
4. Monitor model accuracy monthly and retrain quarterly

## Ethical Considerations

- Model outputs are probabilistic, not guarantees
- Audit quarterly for fairness across routes and customer types
- Preserve human decision-making authority
- Invest in real-time disruption data collection

## Author

Harini Vallal J
MSc Operations and Supply Chain Management
Trinity Business School

## License

MIT License - See LICENSE file for details
