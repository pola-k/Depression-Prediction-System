# Depression Prediction Using Machine Learning

This project is a solution to the [Kaggle Playground Series - Season 4, Episode 11](https://www.kaggle.com/competitions/playground-series-s4e11) competition. It focuses on building a machine learning pipeline that accurately predicts signs of mental health conditions using survey data.

## Objective

To build a highly accurate model that can help identify individuals potentially experiencing mental health issues—supporting timely diagnosis and intervention.


## Data Preprocessing & Deep Cleaning

A robust data cleaning and transformation pipeline was implemented to ensure quality inputs for machine learning models:

* **Exploratory Data Analysis**: Drew insights to guide feature selection and engineering.
* **Missing Value Handling**:

  * Categorical: Imputed using **KNN Imputer**.
  * Numerical: Filled with `0` or the **mean** of the column based on feature behavior.
*  **Encoding**: Applied **One-Hot Encoding** to multi-class categorical features like `Gender`, `Sleep Duration`, and `Dietary Habits`.
* **Age Binning**: Created age groups (`Teenagers`, `Young Adults`, `Adults`, `Middle-aged and Older`) while retaining the original `Age` column.
*  **Data Type Consistency**: Ensured all features were in model-compatible formats.
* **Feature Scaling**: Normalized numerical values using **StandardScaler** to improve convergence during training.

These steps dramatically improved data usability, reduced noise, and enhanced model performance.


## Models Evaluated

A broad range of classification models were tested to benchmark performance:

* Logistic Regression
* RandomForestClassifier
* ExtraTreesClassifier
* BaggingClassifier
* GradientBoostingClassifier
* LinearSVC
* SVC
* DecisionTreeClassifier
* XGBClassifier
* GaussianNB
* AdaBoostClassifier
* LightGBMClassifier
* **CatBoostClassifier** 
* **VotingClassifier** (ensemble of: Logistic Regression, CatBoost, LightGBM, XGBoost)


## Hyperparameter Tuning

The top 4 models used in the Voting Classifier underwent fine-tuning via **Optuna** with **5-fold Stratified Cross-Validation**, optimizing for **accuracy**. This ensured the models were well-calibrated to the dataset without overfitting.


## Best Model

Among all, **CatBoostClassifier** emerged as the best performer with an **Accuracy of 94%**
* Robustness to categorical features
* Minimal preprocessing requirement
* Fast training with excellent generalization


## Evaluation Metrics

Performance was measured using:

* Accuracy
* Precision
* Recall
* F1-Score

Additionally, the UI displays a confidence level with each prediction using a dynamic **progress bar**, offering users a transparent view of prediction certainty.

## Conclusion

This project combines meticulous data cleaning, rigorous model testing, and ensemble learning to achieve strong predictive performance. By leveraging CatBoost and an optimized Voting Classifier, it demonstrates the power of machine learning in sensitive domains like mental health.
