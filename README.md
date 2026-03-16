#  Titanic Survival Prediction (Machine Learning)

This project uses **Machine Learning with Python** to predict whether a passenger survived the **Titanic disaster** based on features such as age, sex, ticket class, and embarkation port.

The model is built using **Logistic Regression** and trained on the **Titanic dataset**. The program also includes **data preprocessing, visualization, and model evaluation**.

This project is a great introduction to **data science, machine learning pipelines, and classification problems**.

---

# Project Overview

The sinking of the Titanic is one of the most famous shipwrecks in history. Using passenger data, we can analyze patterns and build a machine learning model to predict survival outcomes.

The dataset contains information about passengers such as:

- Passenger class
- Gender
- Age
- Number of siblings/spouses aboard
- Number of parents/children aboard
- Ticket fare
- Port of embarkation

Using these features, the model learns patterns that influenced survival probability.

---

#  Features

- 📊 Data preprocessing and cleaning
- 🔎 Handling missing values
- 🔢 Converting categorical data into numerical format
- 🤖 Logistic Regression classification model
- 📈 Model accuracy evaluation
- 📉 Data visualization using Matplotlib
- 📊 Survival distribution plots

---

# 📂 Dataset

The dataset used in this project is the **Titanic training dataset**.

File required:
    train.csv

It contains passenger information and whether they survived.

Important columns include:

| Column | Description |
|------|-------------|
| Survived | Survival status (0 = No, 1 = Yes) |
| Pclass | Passenger class |
| Sex | Gender |
| Age | Passenger age |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Fare | Ticket fare |
| Embarked | Port of embarkation |

---

# 🧹 Data Preprocessing

Several preprocessing steps are applied before training the model.

### Removing unnecessary columns
   Name
   Ticket
   Cabin
   PassengerId

These columns do not significantly contribute to prediction.

---

### Handling Missing Values

- **Age** → Filled with the mean age  
- **Embarked** → Filled with the most frequent value

---

### Encoding Categorical Variables

Machine learning models require numeric input.

Encoding applied:

| Category | Encoding |
|--------|---------|
| male | 0 |
| female | 1 |
| S | 0 |
| C | 1 |
| Q | 2 |

---

#  Machine Learning Model

The project uses **Logistic Regression**, a popular algorithm for **binary classification problems**.

Why Logistic Regression?

- Simple and efficient
- Works well for classification
- Easy to interpret
- Fast training time

Model training:
 model = LogisticRegression(max_iter=2000)
 model.fit(X_train, y_train)

---

#  Train-Test Split

The dataset is divided into two parts:

| Dataset | Purpose |
|-------|--------|
| Training Data | Used to train the model |
| Testing Data | Used to evaluate the model |

Split configuration:
  test_size = 0.2
  random_state = 42

This means **80% training data and 20% testing data**.

---

# Model Evaluation

The model's performance is evaluated using **accuracy score**.
Accuracy = accuracy_score(y_test, y_pred)

Accuracy measures how many predictions were correct.

Example output:
   Accuracy: 0.80

This means the model predicts correctly about **80% of the time**.

---

#  Data Visualization

The project includes two visualizations.

### Survival Distribution

Shows how many passengers survived vs did not survive.
  data['Survived'].value_counts().plot(kind='bar')


This helps visualize the imbalance in the dataset.

---

### Survival by Gender

Displays survival counts for male and female passengers.


pd.crosstab(data['Sex'], data['Survived']).plot(kind='bar')


This plot demonstrates that **female passengers had a higher survival rate**.

---

#  Technologies Used

| Technology | Purpose |
|-----------|--------|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning |

---

#  Required Libraries

Install required libraries using pip:

```bash
pip install pandas numpy matplotlib scikit-learn


