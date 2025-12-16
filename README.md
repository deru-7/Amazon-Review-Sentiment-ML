# Amazon Review Sentiment Analysis (ML)

A **production-style Machine Learning project** that classifies Amazon product reviews into **Negative, Neutral, or Positive** sentiment using classical NLP and linear models.

This project focuses on **clean architecture, correct ML logic, and deployment readiness**, not just model accuracy.

---

## 🚀 Project Overview

* **Problem Type:** Multiclass Text Classification (3-class)
* **Input:** Raw Amazon product review (text)
* **Output:** Sentiment label → `Negative | Neutral | Positive`
* **Approach:** TF-IDF + Linear Models
* **Deployment:** Streamlit Web App

---

## 🧠 Sentiment Mapping Logic

Original Amazon ratings (`Score` column):

| Rating | Mapped Sentiment |
| ------ | ---------------- |
| 1, 2   | Negative (0)     |
| 3      | Neutral (1)      |
| 4, 5   | Positive (2)     |

This avoids fuzzy sentiment boundaries and reflects real-world interpretation.

---

## 📊 Dataset Details

* **Source:** Amazon Fine Food Reviews Dataset
* **Rows:** 568,454 reviews
* **Target Column:** `Score`
* **Text Column Used:** `Text`

Key preprocessing steps:

* Lowercasing
* Special character removal
* Extra whitespace normalization
* TF-IDF vectorization (20,000 features)

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:**

  * pandas, numpy
  * scikit-learn
  * re (regex)
  * pickle
* **Modeling:**

  * Logistic Regression
  * Linear SVM (LinearSVC)
* **Frontend:** Streamlit

---

## 🧪 Model Performance (3-Class)

### Logistic Regression

* **Accuracy:** ~88%
* **Macro F1:** ~0.69

### Linear SVM (Final Model)

* **Accuracy:** ~89%
* **Macro F1:** ~0.70

> ⚠️ Macro F1 is prioritized over accuracy due to strong class imbalance.

---

## 📈 Confusion Matrix (Example)

| Actual \ Predicted | Negative | Neutral | Positive |
| ------------------ | -------- | ------- | -------- |
| **Negative**       | 12,279   | 571     | 3,557    |
| **Neutral**        | 1,808    | 2,408   | 4,312    |
| **Positive**       | 1,835    | 676     | 86,245   |

---

## 🖥️ Streamlit App Features

* Clean text input
* Real-time sentiment prediction
* Color-coded results:

  * 🔴 Negative
  * 🟡 Neutral
  * 🟢 Positive
* Uses **saved TF-IDF vectorizer + trained model**

---




## ✅ Example Inputs

**Negative:**

> The product stopped working after a few days and feels cheaply made.

**Neutral:**

> The product works as described. Nothing special, but it does the job. Quality is average for the price.

**Positive:**

> Excellent quality and fast delivery. Completely satisfied.

---

## 🎯 Key Learnings

* Handling **class imbalance** correctly matters more than raw accuracy
* Neutral sentiment is the hardest class to model
* Classical ML still outperforms deep learning for medium-sized text problems
* Production readiness > fancy models

---

## 📌 Future Improvements

* Probability-based confidence display
* Class-weight tuning
* Error analysis dashboard
* REST API deployment

---

> This project is designed to demonstrate **real-world ML thinking**, not just model training.


