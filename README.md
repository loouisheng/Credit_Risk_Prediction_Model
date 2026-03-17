# Credit-Risk-Prediction-Model
A machine learning project for data science.

## Project Overview
This project aims to identify high-risk loan applicants and automate the loan approval workflow for financial institutions. Using the "Give Me Some Credit" dataset (150,000+ records), this project tackles real-world data challenges including severe class imbalance, missing data mechanics, and non-linear feature relationships. By implementing optimized XGBoost and SMOTE, the system achieves a high AUC-ROC, providing actionable insights for credit risk mitigation.

## Exploratory Data Analysis (EDA): 
Beyond basic statistics, I performed deep-dive analysis into feature distributions:

* **Target Imbalance:** Identified that only ~6.7% of samples are positive (delinquent), necessitating robust sampling strategies.<p align="center"><img width="567" height="455" alt="delinquency" src="https://github.com/user-attachments/assets/b5509ff6-313a-42ce-bc76-c9fe58116150" /> </p>

* **Outlier Detection:** Utilized Interquartile Range (IQR) and histogram to filter for features like DebtRatio and RevolvingUtilizationOfUnsecuredLines to prevent model bias.
<p align="center"><img width="1489" height="1189" alt="outliers" src="https://github.com/user-attachments/assets/03951e0c-ab0c-4050-98aa-65fdc22aa935" /></p>

* **Correlation Analysis:** Utilized heatmaps to analyze feature correlations and identified the high correlation between delinquency history across different time windows (30-59 days vs 90+ days). This helps us to better understand what are the important features for the model training.
<p align="center"><img width="1206" height="1126" alt="corr" src="https://github.com/user-attachments/assets/c16b7c5f-4900-431e-b89f-a59028f05b50" /></p>

## Data Cleaning:
* **Missing Value Imputation:** Instead of simple deletion, I utilized Median Imputation for MonthlyIncome and NumberOfDependents to preserve data density.
* **Feature Scaling:** While tree-based models are robust to scale, features were analyzed for skewness; extreme values in MonthlyIncome and DebtRatio were processed using Log Transformation. Applied Log1p transformation to these features to handle zero values and normalize distribution for improved model convergence.

## Handling Unbalanced Data:
* **SMOTE:** Generated synthetic examples for the minority class in the high-dimensional feature space.
* **Cost-Sensitive Learning**: Instead of modifying the dataset (which can lead to overfitting or loss of information), I modified the Loss Function, implementing the calculation of the ratio of negative to positive samples.This forces the model to prioritize the "Recall" of delinquent borrowers, which is the primary KPI in credit risk.

The reason why I used two approaches to handle unbalanced data is because I noticed the SMOTE solution's result was not optimistic. Therefore, I used cost-sensitive learning approach to try enhacning the model performance, and it work better than SMOTE approach.

## Model Architecture and Hyperparameter Tuning:
I utilized XGBoost for its exceptional handling of tabular data and built-in support for missing values.

```Python
xgb = XGBClassifier(
    n_estimators=130, 
    learning_rate=0.05, 
    max_depth=5, 
    scale_pos_weight=15,
    random_state=42
)
```
* **Learning Rate ($0.05$):** A lower rate combined with sufficient estimators to ensure smooth convergence.

* **Max Depth ($5$):** Balanced complexity to capture non-linear interactions without overfitting to specific noise in the training set.

The optimal hyperparameters were found by **GridSearch** combined with manual fine-tuning, as automated searches occasionally lead to overfitting on training data.

## Result
### Evaluation Metrics

In unbalanced credit scoring, Accuracy is a misleading metric. We focus on:
| Metric | Score | Financial Meaning |
| :--- | :--- | :--- |
| ROC-AUC | 0.8643 | Superior ability to rank-order customers by risk. |
| Recall | 0.79 | Successfully captures 79% of all actual defaulters, significantly reducing credit loss.. |
| Precision | 0.20 | Acceptable trade-off in cost-sensitive learning; prioritizes "safety first" over manual review overhead. |

**Note on Precision:** A Precision of 0.20 is highly competitive given the baseline default rate of ~6.7%. It indicates that the model is 3x more effective than random selection, providing a robust filter for secondary manual credit reviews.

### Feature Importance

<p align="center"><img width="914" height="590" alt="features_importance" src="https://github.com/user-attachments/assets/2ba11bd5-fc42-453c-b3e9-456106c7d385" /></p>

* **Revolving Utilization of Unsecured Lines (Top Driver):** It is the strongest predictor of default. High utilization suggests a borrower is "maxing out" their credit cards, indicating a liquidity crunch and immediate financial distress.
* **Age**: Younger borrowers tend to have more volatile income and less credit history (thin files), whereas older borrowers typically possess higher financial stability and wealth accumulation.
* **DebtRatio & MonthlyIncome:** A high DebtRatio combined with low MonthlyIncome indicates that a large portion of earnings is already committed to debt servicing, leaving little margin for financial shocks.
* **Historical Delinquency (30-59 Days Past Due):** Short-term delinquency acts as a Leading Indicator. It is often the first sign of a deteriorating credit profile before a full default occurs.
