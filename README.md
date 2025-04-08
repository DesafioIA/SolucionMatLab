# Advanced Diabetes Prediction Model for ICU Patients

## Description
A comprehensive diabetes prediction system developed for the Engineering for the Americas Virtual Hackathon 2025, using the WiDS Datathon 2021 dataset. This model employs advanced machine learning techniques to predict diabetes risk in patients newly admitted to intensive care units with high accuracy.

## Dataset Details
- **Source**: WiDS Datathon 2021
- **Size**: 130,157 rows, 181 columns
- **Class Distribution**: 21.63% patients with diabetes, 78.37% without diabetes
- **Missing Data**: Several critical variables had missing values (h1_temp_max: 22.8%, wbc_apache: 22.6%)

## Key Predictive Variables
- **Glucose measurements**: d1_glucose_max (r=0.4007), glucose_apache (r=0.3544)
- **Body metrics**: BMI (r=0.1690), weight (r=0.1555)  
- **Kidney function**: d1_bun_max (r=0.1470), d1_creatinine_max (r=0.1279)
- **Age**: Higher median age in diabetic patients (66.00 vs 63.00)

## Medical Insights
- Diabetic patients showed significantly higher glucose levels (median: 214.00 vs 138.00)
- BMI differences between groups (median: 30.15 with diabetes vs 26.99 without)
- Elevated creatinine levels in diabetic patients (median: 1.20 vs 0.96)

## Models Evaluated
- **Random Forest**: Accuracy 0.9248, F1-Score 0.8217, AUC-ROC 0.9597
- **Gradient Boosting**: Accuracy 0.8513, F1-Score 0.4759, AUC-ROC 0.9825  
- **RUSBoost**: Accuracy 0.2671, F1-Score 0.3712, AUC-ROC 0.9649
- **Neural Network**: Accuracy 0.8515, F1-Score 0.6679, AUC-ROC 0.8949
- **Advanced Ensemble**: Accuracy 0.9372, F1-Score 0.8525, AUC-ROC 0.9740

## Cross-Validation Results
- Robust 5×10-fold cross-validation for the Advanced Ensemble model
- **Accuracy**: 0.7159 (±0.0057)
- **Precision**: 0.4314 (±0.0049)
- **Sensitivity**: 0.9849 (±0.0030)
- **F1-Score**: 0.6000 (±0.0046)
- **AUC-ROC**: 0.9653 (±0.0020)

## Technical Implementation
- MATLAB implementation with customized preprocessing pipeline
- Optimized decision thresholds for each model (e.g., RF: 0.35, GB: 0.10)
- Advanced ensemble weighting based on model performance

This project demonstrates how machine learning can effectively support clinical decision-making in ICU settings, with particular strength in identifying patients at risk for diabetes using readily available clinical measurements.
