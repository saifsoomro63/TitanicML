# Titanic - Machine Learning from Disaster (Kaggle Challenge)

## 📌 Project Overview
This repository contains my comprehensive solution to the classic **Titanic: Machine Learning from Disaster** competition on Kaggle. The objective is to predict passenger survival ("what sorts of people were more likely to survive?") using passenger data such as age, gender, socio-economic class, and family size.

To systematically tackle this tabular classification problem and explore different facets of machine learning, I built, tuned, and evaluated **7 distinct prediction models**. This iterative process allowed me to compare baseline algorithms against advanced ensemble techniques and optimize feature engineering for performance.

## 📊 Dataset Profile
The project utilizes passenger data split into two subsets:
* **`train.csv`**: Details of 891 passengers, including the ground truth (`Survived`: 0 = No, 1 = Yes).
* **`test.csv`**: Features for 418 passengers without survival labels, used to evaluate model accuracy on unseen data.

**Key Features Explored:**
* `Pclass` (Socio-economic class)
* `Sex` & `Age`
* `SibSp` & `Parch` (Family relation variables)
* `Fare` (Ticket pricing)
* `Embarked` (Port of embarkation)
* `Cabin` (Cabin number)

## 🛠️ Data Pipeline & Feature Engineering
Before feeding data into the models, a robust data preprocessing pipeline was developed to avoid data leakage:
1.  **Missing Value Imputation:** * Imputed missing `Age` values using median values grouped by `Pclass` and `Sex`.
    * Filled missing `Embarked` values with the mode.
    * Handled missing `Fare` data using the overall median.
2.  **Feature Extraction:**
    * **Title Extraction:** Extracted titles (Mr., Mrs., Miss, Master, Dr., etc.) from the `Name` column to identify social standing and refine age/gender categorization.
    * **Family Size:** Combined `SibSp` and `Parch` to form `FamilySize` and categorized passengers as `IsAlone` vs. part of a family.
3.  **Data Encoding & Scaling:**
    * One-Hot Encoding applied to categorical variables (`Sex`, `Embarked`, `Title`).
    * Standardized numerical features (`Age`, `Fare`) using `StandardScaler` to optimize distance-based and gradient-descent algorithms.

## 🤖 The 7 Prediction Models
I implemented a spectrum of algorithms ranging from linear baselines to advanced ensemble architectures to benchmark performance. 

1.  **Logistic Regression:** Served as our statistical baseline model.
2.  **K-Nearest Neighbors (KNN):** Used to capture local data patterns via distance metrics.
3.  **Support Vector Machine (SVM):** Implemented to test non-linear boundaries using the RBF kernel.
4.  **Naive Bayes:** A probabilistic classifier utilized for baseline contrast.
5.  **Decision Tree:** A non-parametric model used to visualize rule-based classification paths.
6.  **Random Forest:** An ensemble bagging technique implemented to reduce variance and decision tree overfitting.
7.  **Gradient Boosting (e.g., XGBoost / LightGBM / CatBoost):** *[Specify which one you used here]* An iterative boosting algorithm tailored for peak predictive accuracy.

*Hyperparameters for ensemble models were fine-tuned using `GridSearchCV` / `RandomizedSearchCV` across a stratified k-fold cross-validation.*

## 📈 Performance Summary & Results
Models were evaluated locally using **Stratified 5-Fold Cross-Validation Accuracy** and verified against the Kaggle public leaderboard.

| Model | CV Accuracy (%) | Kaggle Score | Key Observations |
| :--- | :---: | :---: | :--- |
| Logistic Regression | `0.XX` | `0.75+` | High interpretability; struggle with non-linear feature interactions. |
| K-Nearest Neighbors | `0.XX` | `0.75+` | Sensitive to feature scaling, but performed decently after standardization. |
| Support Vector Machine | `0.XX` | `0.75+` | Handled non-linear boundaries well with RBF kernel tuning. |
| Naive Bayes | `0.XX` | `0.75+` | Fast baseline, but hindered by the independence assumption. |
| Decision Tree | `0.XX` | `0.75+` | Prone to overfitting without depth regularizations. |
| Random Forest | `0.XX` | `0.75+` | Drastic improvement; feature importance showed `Sex` and `Title` dominated. |
| **Gradient Boosting (Best)** | **`0.XX`** | **`0.75+`** | Provided the strongest predictive power and generalized best to test set. |


### 🔍 Key Takeaways
* **Feature Engineering > Complex Models:** Engineering features like passenger `Title` and `FamilySize` yielded a more significant accuracy boost than shifting between base algorithms.
* **The Power of Ensembles:** Tree-based ensemble methods (Random Forest & Gradient Boosting) significantly outperformed single estimators by effectively mapping structural interactions within tabular data.
