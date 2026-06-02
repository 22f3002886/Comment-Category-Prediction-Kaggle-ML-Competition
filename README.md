# Comment Category Prediction Challenge

## 🏆 Kaggle Competition | NLP · Multi-Class Text Classification

> A college-hosted Kaggle competition to predict the category of user comments across 4 classes using an ensemble of NLP and machine learning models.

---

## 📌 Overview

This project builds an end-to-end NLP pipeline that classifies user comments into one of 4 categories. The pipeline covers exploratory data analysis, text preprocessing, custom feature engineering, TF-IDF vectorization, model training with class-imbalance handling, hyperparameter tuning, and a final soft-voting ensemble — achieving **91.78% accuracy** and **82.92% Macro F1** on the validation set.

🔗 **Competition:** [Comment Category Prediction Challenge – Kaggle](https://www.kaggle.com/competitions/comment-category-prediction-challenge/overview)

---

## 📂 Project Structure

```
├── notebook-t12026.ipynb   # Full project notebook
├── submission.csv                      # Final competition submission
└── README.md
```

---

## 📊 Results

| Model | Accuracy | F1 (Weighted) | F1 (Macro) |
|---|---|---|---|
| ✅ **Voting Ensemble** *(final)* | **91.78%** | **91.87%** | **82.92%** |
| LightGBM | 91.65% | 91.73% | 82.43% |
| XGBoost | 90.23% | 90.65% | 79.34% |
| Logistic Regression | 89.19% | 89.84% | 78.12% |

> **Metric:** Macro F1-Score was used as the primary metric to handle class imbalance fairly across all categories.

---

## 🔍 Approach

### 1. Exploratory Data Analysis
- Visualized class (label) distribution to identify imbalance
- Plotted correlation heatmap across numeric features
- Analyzed comment length distribution across classes

### 2. Preprocessing
- **Text:** Lowercased, removed URLs, stripped special characters, collapsed whitespace
- **Numeric:** Filled missing values with median
- **Categorical** (`race`, `religion`, `gender`, `disability`): Filled with mode, encoded via `LabelEncoder`

### 3. Feature Engineering (12+ custom features)

| Category | Features |
|---|---|
| Text-based | `comment_length`, `word_count`, `avg_word_length`, `uppercase_ratio`, `exclamation_count`, `question_count` |
| Vote-based | `vote_diff`, `vote_total`, `vote_ratio` |
| Interaction | `emoticon_total`, `if_diff`, `if_product` |

### 4. Feature Representation
- **Word-level TF-IDF** — 1–3 n-grams, 5,000 features (`sublinear_tf=True`)
- **Character-level TF-IDF** — 2–5 char n-grams, 5,000 features (`analyzer='char_wb'`)
- **Scaled numeric features** — `StandardScaler` via sklearn `Pipeline`
- All three matrices horizontally stacked into a single sparse feature matrix (~10,000+ features)

### 5. Modeling & Tuning
- Trained **Logistic Regression**, **XGBoost**, and **LightGBM** as base models
- Handled class imbalance via `class_weight='balanced'` and `compute_sample_weight`
- Applied `RandomizedSearchCV` (cv=2, Macro F1) on the best-performing base model
- Combined all three into a **Soft Voting Ensemble** as the final model

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python | Core language |
| Jupyter Notebook | Development environment |
| pandas / NumPy | Data manipulation |
| scikit-learn | Pipelines, TF-IDF, LR, ensemble, metrics |
| XGBoost | Gradient boosting classifier |
| LightGBM | Fast histogram-based boosting |
| Matplotlib / Seaborn | Visualization |
| Kaggle | Competition platform & compute |

---

## 🚀 How to Use This Repository

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/comment-category-prediction.git
   ```

2. **Navigate to the project folder**
   ```bash
   cd comment-category-prediction
   ```

3. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
   ```

4. **Run the notebook**
   ```bash
   jupyter notebook 22f3002886-notebook-t12026.ipynb
   ```
   Execute all cells to reproduce the full pipeline and results.

> 💡 The notebook was originally run on Kaggle. If running locally, update the dataset paths in **Section 3 (Load Data)** to point to your local CSV files.

---

## 📬 Contact

- **Name:** Abhishek Kumar
- **LinkedIn:** [Your LinkedIn](https://www.linkedin.com/in/masterabhishek/)
- **GitHub:** [Your GitHub](https://github.com/22f3002886)
- **Kaggle:** [Your Kaggle](https://www.kaggle.com/abhiishek01)

Feel free to reach out for any questions or collaboration opportunities!

---

*This README provides a complete overview of the project including the approach, results, and instructions to reproduce the work.*
