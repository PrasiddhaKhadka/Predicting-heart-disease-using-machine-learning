# ❤️ Predicting Heart Disease Using Machine Learning  

This project explores how to leverage **Python-based machine learning** and **data science libraries** to build a predictive model that determines whether an individual is likely to have **heart disease** based on their medical attributes.  

The goal is to apply a structured **data science workflow** to transform raw medical data into actionable insights and predictive power.  

---

## 🔍 Approach  

To ensure a clear and systematic process, the following steps were followed:  

1. **📝 Problem Definition**  
   Define the challenge: *Can we predict whether a patient has heart disease given their medical features?*  

2. **📊 Data Collection & Understanding**  
   Explore the dataset, identify patterns, handle missing values, and prepare it for analysis.

   The original data came from the Cleavland data from the UCI Machine Learning Repository.

   https://archive.ics.uci.edu/ml/datasets/heart+Disease

   There is also a version of it available on Kaggle. https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset


4. **📈 Evaluation**  
   Determine the success metric (e.g., accuracy, precision, recall, F1-score) to evaluate model performance.
   >If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we can pursue the         project.

6. **⚙️ Features**  
   Select and engineer the most relevant attributes that contribute to prediction.

   Create data dictionary


    ### 📌 Dataset Features  

    Below are the key features in the dataset along with their descriptions:  
    
    1. **age** – Age in years.  
    
    2. **sex** – Biological sex (1 = male, 0 = female).  
    
    3. **cp** – Chest pain type:  
       - 0: Typical angina → chest pain related to reduced blood supply to the heart  
       - 1: Atypical angina → chest pain not related to the heart  
       - 2: Non-anginal pain → often due to esophageal spasms (non-heart related)  
       - 3: Asymptomatic → chest pain not showing signs of disease  
    
    4. **trestbps** – Resting blood pressure (in mm Hg upon hospital admission).  
       - Normal: ~120 mm Hg  
       - Concern: ≥130–140 mm Hg  
    
    5. **chol** – Serum cholesterol in mg/dl.  
       - Formula: LDL + HDL + 0.2 × triglycerides  
       - Concern: >200 mg/dl  
    
    6. **fbs** – Fasting blood sugar (>120 mg/dl).  
       - 1 = True  
       - 0 = False  
       - Note: >126 mg/dl indicates diabetes  
    
    7. **restecg** – Resting electrocardiographic results:  
       - 0: Normal (nothing to note)  
       - 1: ST-T wave abnormality → from mild to severe irregularities in heart rhythm  
       - 2: Possible or definite left ventricular hypertrophy → enlargement of the heart’s main pumping chamber  
    
    8. **thalach** – Maximum heart rate achieved.  
    
    9. **exang** – Exercise-induced angina (1 = Yes, 0 = No).  
    
    10. **oldpeak** – ST depression induced by exercise relative to rest.  
        - Higher values indicate higher stress on the heart.  
    
    11. **slope** – Slope of the peak exercise ST segment:  
        - 0: Upsloping → better heart rate with exercise (rare)  
        - 1: Flatsloping → minimal change (typical healthy heart)  
        - 2: Downsloping → signs of an unhealthy heart  
    
    12. **ca** – Number of major vessels (0–3) colored by fluoroscopy.  
        - Colored = blood flow visible (healthy)  
        - More vessels colored = better circulation  
    
    13. **thal** – Thallium stress test result:  
        - 1, 3: Normal  
        - 6: Fixed defect (previous defect but currently stable)  
        - 7: Reversible defect (poor blood movement during exercise)  
    
    14. **target** – Presence of heart disease (predicted attribute).  
        - 1 = Yes (disease present)  
        - 0 = No (no disease)  

        


8. **🤖 Modelling**  
   Apply different machine learning algorithms, train the models, and compare their performance.  

9. **🧪 Experimentation & Optimization**  
   Tune hyperparameters, improve performance, and document insights from experimentation.  

---

✨ With this workflow, the project aims not only to build a robust predictive model but also to showcase best practices in **data preprocessing, model selection, and evaluation** in machine learning.  


# 🛠️ Preparing the Tools  

For this project, we will leverage key Python libraries for **data analysis, visualization, and numerical computation**:  

- **pandas** → for data loading, cleaning, and manipulation  
- **NumPy** → for efficient numerical operations and mathematical computations  
- **Matplotlib** → for creating insightful visualizations and plots  

These tools form the backbone of our data science workflow, enabling us to preprocess the dataset, explore patterns, and build a strong foundation for machine learning.  


```python
# Importing all the necessary tools

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# magic command in jupyter notebook(showing the plot inside the notebook)
%matplotlib inline

# Model from scikitlearn 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.metrics import RocCurveDisplay
```

## Load Data

```python
df = pd.read_csv('heart-disease.csv')
df.shape
```

## 🔎 Data Exploration (Exploratory Data Analysis – EDA)  

The purpose of **Exploratory Data Analysis (EDA)** is to develop a deeper understanding of the dataset and become familiar with the domain before building models. This step helps identify patterns, anomalies, and insights that guide the overall workflow.  

Key questions we aim to answer during EDA include:  

1. **❓ What problem are we trying to solve?**  
   - Define the research question and align it with the dataset’s scope.  

2. **📊 What type of data do we have?**  
   - Identify numerical, categorical, and ordinal variables.  
   - Decide how to process and treat each type.  

3. **⚠️ What’s missing from the data?**  
   - Detect missing values and decide on strategies (imputation, removal, or replacement).  

4. **📉 Where are the outliers?**  
   - Locate unusual values that may distort analysis.  
   - Understand their impact and whether they should be kept, corrected, or removed.  

5. **🛠️ How can features be engineered?**  
   - Add, modify, or remove features to improve the dataset’s predictive power.  
   - Create domain-specific features that may boost model performance.  

---

✨ By answering these questions, EDA ensures the dataset is **clean, well-understood, and ready** for the modelling phase.  


```python
df.head()
```

```python
df.tail()
```

```python
# Checking the empty or null value in our dataset
df.isna().sum()
```

```python
# Checking out how many of each class is there 
df['target'].value_counts()
```

```python
# plotting the value of our target 
df['target'].value_counts().plot(kind='bar',color=['salmon','Green'],title='Target Counts')
plt.show()
```

```python
# information of our dataset 
df.info()
```

```python
df.describe()
```

## Heart Disease Frequence according to Sex

```python
df['sex'].value_counts()
```

```python
# Comparing the target column with sex column

pd.crosstab(df.target,df.sex)
```

```python
# Creating a plot of the crosstab

pd.crosstab(df.target,df.sex).plot(
    kind='bar',
    figsize=(10,6),
    color=['salmon','lightblue'],
)
plt.title('Heart Diseases Frequence for Sex')
plt.xlabel("0= No Diseases, 1 = Diseases")
plt.ylabel('Count')
plt.legend(['Female','Male'])
plt.xticks(rotation=0)
plt.show()
```

### Age vs Max heart rate for heart disease

```python
# creating a figure
plt.figure(figsize=(10,6))

# scatter with positive examples
#df.age[df.target == 1] => people with age that have positive heart disease
plt.scatter(x= df.age[df.target == 1],
           y=df.thalach[df.target == 1],
           color=['salmon'])

# scatter with negative examples
plt.scatter(x= df.age[df.target == 0],
           y = df.thalach[df.target == 0],
           color = ['lightblue']
           )

# customizing the chart
plt.title('Heart Disease in function of Age and Max Heart Rate')
plt.xlabel('Age')
plt.ylabel('Thalach / Max heart rate')
plt.legend(['Disease','No Disease'])
plt.axhline(y=df.thalach.mean(),color='grey',linestyle=':')

# show 
plt.show()
```

```python
# Checking the distribution of age column with histogram

df.age.plot(kind='hist')
plt.show()
```

### ❤️ Feature: Chest Pain Type (`cp`)

This feature describes the **type of chest pain** experienced by the patient:

- 🟥 **0: Typical Angina** → Chest pain related to **reduced blood supply** to the heart.  
- 🟧 **1: Atypical Angina** → Chest pain **not directly related** to the heart.  
- 🟨 **2: Non-Anginal Pain** → Often caused by issues such as **esophageal spasms** (non-heart related).  
- 🟩 **3: Asymptomatic** → No typical chest pain symptoms, but may still indicate underlying disease.  


```python
pd.crosstab(df.cp,df.target)
```

```python
# Making the cross tab visual
pd.crosstab(df.cp,df.target).plot(
    kind='bar',
    figsize=(10,6),
    color=['salmon','lightblue']
)

# customizing
plt.title('Heart Disease Frequency Per Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.ylabel('Amount')
plt.legend(['No Disease','Disease'])


plt.show()
```

```python
df.head()
```

```python
# Make a correlation matrix
df.corr()
```

```python
# Let's make our co-relation matrix a little prettier

corr_matrix = df.corr()

fig, ax = plt.subplots(figsize=(10, 6)) 

ax = sns.heatmap(
    corr_matrix,
    annot=True,
    linewidths=0.5,
    fmt='.2f',
    #colormap
    cmap='YlGnBu'            
)

plt.show()

```

## Modelling

```python
df.head()
```

```python
# Splitting the data into X(Independent Features) and y(Dependent Features)

X = df.drop('target',axis=1)
y = df['target']

```

```python
# Splitting the data into train and test sets
np.random.seed(42)

# Split it into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size=0.2)

```

```python
X_train
```

```python
y_train
```

### 🚀 Building and Evaluating Machine Learning Models  

Now that we’ve successfully **split our dataset** into **training** and **test sets**, it’s time to move on to the next step: **model building**.  

- 🏋️ **Training Set** → Used to **train** the model (learn patterns).  
- 🧪 **Test Set** → Used to **evaluate** the model (check how well it generalizes).  

We’ll experiment with **three different machine learning algorithms** to compare their performance:  

1. 📊 **Logistic Regression**  
2. 🤝 **K-Nearest Neighbours (KNN) Classifier**  
3. 🌲 **Random Forest Classifier**  

By testing these models, we’ll be able to identify which one performs best on our dataset.  


```python
# Put models in a dictionary
models = {"Logistic Regression": LogisticRegression(max_iter=1000),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of differetn Scikit-Learn machine learning models
    X_train : training data (no labels)
    X_test : testing data (no labels)
    y_train : training labels
    y_test : test labels
    """
    # Set random seed
    np.random.seed(42)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores
```

```python
model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)

model_scores
```

### Model Comparison

```python
model_compare = pd.DataFrame(model_scores,index=['Score']).T
model_compare
```

```python
model_compare.plot(
    kind='bar',
    title = 'Model Score',
    color = ['lightgreen']
)
plt.xlabel('Tested Models')
plt.ylabel('Scores')
plt.xticks(rotation=0)
plt.show()
```

# 🔎 Next Steps After Building a Baseline Model

Now that we’ve got a **baseline model**, it’s important to remember:  
➡️ A model’s **first predictions** aren’t always the ones we should base our next steps on.  

To improve and properly evaluate our model, we’ll look at the following:

---

## ⚙️ Model Improvement
- **Hyperparameter Tuning** → Optimize parameters to boost performance.  
- **Feature Importance** → Identify which features contribute the most.  

---

## 📊 Model Evaluation
- **Confusion Matrix** → Visualize predictions vs. actual values.  
- **Cross-Validation** → Ensure results are consistent across different subsets of data.  

---

## 🎯 Performance Metrics
- **Precision** → Of the predicted positives, how many are correct?  
- **Recall** → Of all actual positives


### ⚙️ Model Improvement (Hyperparameter tuning)

```python
# 1. Using KNN

train_scores = []
test_scores = []

# Creating a list of different values for n_neigbhours
neighbours = range(1,21)

# Setting up an instance 
knn = KNeighborsClassifier()

# Loop through different n_neigbours
for i in neighbours:
    knn.set_params(n_neighbors = i)

    # Fitting the algorithm
    knn.fit(X_train,y_train)

    # Update the training score list 
    train_scores.append(knn.score(X_train,y_train))

    # Update the testing score list 
    test_scores.append(knn.score(X_test,y_test))
    
```

```python
train_scores
```

```python
test_scores
```

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(neighbours, train_scores, marker='o', linestyle='-', label='Train Score', color='royalblue')
plt.plot(neighbours, test_scores, marker='s', linestyle='--', label='Test Score', color='darkorange')

# Title & labels
plt.title('KNN Performance: Train vs Test Scores', fontsize=14, fontweight='bold')
plt.xlabel('Number of Neighbours (K)', fontsize=12)
plt.ylabel('Model Accuracy Score', fontsize=12)

# Grid & legend
plt.grid(alpha=0.3)
plt.legend(fontsize=11, loc='best')

# Highlight maximum test score
best_k = neighbours[test_scores.index(max(test_scores))]
best_score = max(test_scores)
plt.scatter(best_k, best_score, color='red', s=100, label=f'Best Test Score ({best_score*100:.2f}%)')
plt.legend()

# Show result
print(f"✅ Maximum KNN Score on the test data: {best_score*100:.2f}% (at K={best_k})")
plt.show()

```

# 🎯 Hyperparameter Tuning with RandomizedSearchCV

To improve our models beyond the baseline, we’ll perform **hyperparameter tuning** using **RandomizedSearchCV**.  

We’ll focus on tuning the following models:  

- 📊 **Logistic Regression** (`LogisticRegression()`)  
- 🌲 **Random Forest Classifier** (`RandomForestClassifier()`)  

This process helps us:  
- 🔧 Find the **best set of parameters**  
- 📈 Improve model **accuracy and generalization**  
- ⏱️ Save time compared to GridSearchCV by sampling a random subset of parameters  


1 📊 **Logistic Regression** (`LogisticRegression()`)  

```python
# Creating a hyperparameters for logisticRegression
# np.logspace() -> generates value from 10^-4 to 10^4
log_reg_grid = {"C":np.logspace(-4,4,20),
               "solver":['liblinear']
               }

# Creating a hyperparameters for RandomForestClassifier
rf_grid = {
    "n_estimators":np.arange(10,1000,50),
    "max_depth":[None,3,5,10],
    "min_samples_split":np.arange(2,20,2),
    "min_samples_leaf":np.arange(1,20,2)
}
```

```python
# Tunning the logistic Regression
np.random.seed(42)

# Randomized Search CV
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                               param_distributions=log_reg_grid,
                               cv=5,
                               n_iter=20,
                               verbose=True)

# Fitting random hyperparameter search model for logisticRegression
rs_log_reg.fit(X_train,y_train)
```

```python
rs_log_reg.best_params_
```

```python
rs_log_reg.score(X_test,y_test)
```

2 🌲 **Random Forest Classifier** (`RandomForestClassifier()`)  


```python
# Tunning the Random Forest Classifier
np.random.seed(42)

# Randomized Search CV
rs_random_forest = RandomizedSearchCV(RandomForestClassifier(),
                               param_distributions=rf_grid,
                               cv=5,
                               n_iter=20,
                               verbose=True)

# Fitting random hyperparameter search model for logisticRegression
rs_random_forest.fit(X_train,y_train)
```

```python
rs_random_forest.best_params_
```

```python
rs_random_forest.score(X_test,y_test)
```

# 🎯 Hyperparameter Tuning with GridSearchCV

Since our **Logistic Regression** model has achieved the **best performance so far**, the next step is to **further optimize its hyperparameters** using **GridSearchCV**.  

**Why GridSearchCV?**  
- Exhaustively searches over a specified parameter grid  
- Finds the **best combination of hyperparameters** for maximum model performance  
- Ensures robust evaluation with **cross-validation**  

This process helps us push our baseline model even further and ensures that we are using the most effective configuration for our dataset.

1 📊 **Logistic Regression** (`LogisticRegression()`)  

```python
# Different hyperparameters for our logistics regression model 
lg_reg_grid = {
    'C':np.logspace(-4,4,30),
    'solver':['liblinear']
}

# Setting up grid hyperparameter search for logistic regression
gs_log_reg = GridSearchCV(LogisticRegression(),
                         lg_reg_grid,
                         cv=5,
                         verbose=True)

# Fitting the grid hyperparameter search model 
gs_log_reg.fit(X_train,y_train)
```

```python
gs_log_reg.best_params_
```

```python
gs_log_reg.score(X_test,y_test)
```

# 📊 Evaluating Our Tuned Machine Learning Classifier

Once our model has been **tuned**, it’s important to evaluate its performance **beyond just accuracy**.  

Key evaluation tools include:  

- **ROC Curve & AUC Score** → Assess the trade-off between True Positive Rate and False Positive Rate  
- **Confusion Matrix** → Visualize true vs. predicted labels  
- **Classification Report** → Includes Precision, Recall, and F1-score for each class  
- **Precision** → Proportion of predicted positives that are actually correct  
- **Recall** → Proportion of actual positives that are correctly identified  
- **F1-Score** → Harmonic mean of Precision and Recall  
- **Cross-Validation** → Ensures consistent performance across multiple subsets of the data  

---

## 🔹 Making Predictions

To make meaningful comparisons and properly evaluate our **trained model**, the first step is to **generate predictions** on the test set.  

This provides the foundation for computing all the metrics listed above and understanding how well the model generalizes to unseen data.


```python
# Making predictions with tuned model 

y_preds= gs_log_reg.predict(X_test)
y_preds
```

```python
y_preds.shape
```

```python
y_test
```

```python
# plot ROC curve and calculate AUC metric

RocCurveDisplay.from_estimator(gs_log_reg,X_test,y_test)
plt.show()
```

```python
# Confusion matrix 

print(confusion_matrix(y_test,y_pred=y_preds))
```

```python
sns.set(font_scale = 1.5)

def plot_conf_matrix(y_test,y_preds):
    """
    Plots a nice looking confusion matrix using Seaborn's heatmap()
    """
    fig, ax = plt.subplots(figsize = (3,3))
    ax = sns.heatmap(
        confusion_matrix(y_test,y_preds),
        annot=True,
        cbar=False
    ) 
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    
    plt.show()

plot_conf_matrix(y_test, y_preds)
```

```python
print(classification_report(y_test, y_preds))
```

### Calculate the evaluation metrics using cross-validation

We're going to calculate accuracy, precision, recall and f1-score of our model using cross-validation and to do so we'll be using `cross_val_score()`

```python
# Check best hyperparameters
gs_log_reg.best_params_
```

```python
# Create a new classifier with best parameters
clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")
```

```python
# Cross-validated accuracy
cv_acc = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="accuracy")
cv_acc
```

```python
cv_acc = np.mean(cv_acc)
cv_acc
```

```python
# Cross-validated precision
cv_precision = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="precision")
cv_precision=np.mean(cv_precision)
cv_precision
```

```python
# Cross-validated recall
cv_recall = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="recall")
cv_recall = np.mean(cv_recall)
cv_recall
```

```python
# Cross-validated f1-score
cv_f1 = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="f1")
cv_f1 = np.mean(cv_f1)
cv_f1
```

```python
# Visualize cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                           "Precision": cv_precision,
                           "Recall": cv_recall,
                           "F1": cv_f1},
                          index=[0])

cv_metrics.T.plot.bar(title="Cross-validated classification metrics",
                      legend=False);

plt.show()
```

## Feature Importance in Machine Learning Models

Feature importance helps us answer the question:  
**"Which features contributed the most to the model’s predictions, and how did they contribute?"**

The approach to determining feature importance depends on the type of machine learning model being used. Different models expose different methods or attributes for this purpose. A practical way to discover the correct approach is to search for:  
`<model name> feature importance`.

---

### Logistic Regression Feature Importance

For **Logistic Regression**, feature importance is typically derived from the model’s coefficients.  
- A **positive coefficient** indicates that as the feature increases, the likelihood of the positive class also increases.  
- A **negative coefficient** suggests the opposite effect.  
- The **magnitude of the coefficient** reflects the strength of the influence, though scaling of features should be considered for fair comparison.

In the following section, we will compute and visualize the feature importance for our tuned Logistic Regression model.


```python
gs_log_reg.best_params_
```

```python
clf = LogisticRegression(C=0.20433597178569418,
                        solver='liblinear')
clf.fit(X_train,y_train)
```

```python
# checking coef
clf.coef_
```

```python
df.head()
```

```python
# Matching the co-efficient of feature to the column
feature_dict = dict(zip(df.columns,list(clf.coef_[0])))
feature_dict
```

```python
# Visualize the feature importance
feature_df = pd.DataFrame(feature_dict,index=[0])
feature_df.T.plot.bar(title="Feature Importance",legend=False)
plt.show()
```

```python

```
