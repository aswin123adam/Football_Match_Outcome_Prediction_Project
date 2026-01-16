# ⚽ Arsenal Football Match Outcome Prediction
A machine learning project to predict Arsenal FC match outcomes (Win/Draw/Loss) using historical team and player performance data spanning 6 years.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Models & Results](#models--results)
- [Key Features Used](#key-features-used)
- [Future Improvements](#future-improvements)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Team Members](#team-members)

---

## 🎯 Project Overview

This project aims to build predictive models for Arsenal FC match outcomes using machine learning techniques. We leverage both team-level and player-level statistics to understand which factors most significantly influence match results.

**Primary Objective:** Predict whether Arsenal will Win (W), Draw (D), or Lose (L) a given match.

**Secondary Objective:** Predict Expected Goals (xG) based on in-game performance metrics.

---

## 📊 Dataset Description

### Team Data (`Arsenal_team_data.xlsx`)
- **Records:** 219 matches
- **Time Period:** 2019 - 2025
- **Features:** 29 columns including goals, xG, shots, possession, formations, captaincy, and venue

### Player Data (`Arsenal_Player_data_for_6_years.xlsx`)
- **Records:** 538 player-match entries
- **Features:** 34 columns including individual stats (goals, assists, xG, passes, tackles, etc.)

---

## 🔬 Methodology

### 1. Data Preprocessing
- Loaded and cleaned both team and player datasets
- Handled missing values using `.dropna()`
- Encoded categorical variables:
  - Result: W=2, D=1, L=0
  - Venue: Home=1, Away=0
  - Formations: Label encoded

### 2. Feature Engineering
- Created aggregated match-level features from player data
- Calculated weighted pass accuracy based on minutes played
- Combined tackles and interceptions into defensive actions metric
- Extracted time-based features (hour, day of week)

### 3. Model Development
- Split data: 80% training, 20% testing
- Applied Stratified K-Fold Cross-Validation (5 folds)
- Trained Random Forest models for both classification and regression tasks

---

## 📈 Models & Results

### Model 1: Match Outcome Classification (Random Forest Classifier)

| Metric | Value |
|--------|-------|
| Cross-Validated Accuracy | ~59.8% |
| Test Accuracy | ~54.5% |

**Classification Report:**
```
              precision    recall  f1-score   support
           D       0.00      0.00      0.00         4
           L       0.43      0.21      0.29        14
           W       0.62      0.81      0.70        26
    accuracy                           0.55        44
```

### Model 2: Expected Goals (xG) Regression (Random Forest Regressor)

| Metric | Value |
|--------|-------|
| Mean Squared Error (MSE) | 0.274 |
| R-squared (R²) | 0.578 |

After adding defensive features (Tackles & Interceptions, Set-Piece Goals):
| Metric | Value |
|--------|-------|
| MSE | 0.266 |
| R² | 0.589 |

---

## 🔑 Key Features Used

### Team-Level Features
- Expected Goals (xG) & Expected Goals Against (xGA)
- Shots Total & Shots on Target
- Possession %
- Pass Completion %
- Venue (Home/Away)
- Formation Used
- Opponent Formation
- Tackles and Interceptions
- Set-Piece Goals Scored

### Player-Level Aggregated Features
- Total Goals & xG per match
- Total Minutes Played
- Key Passes (Shot Creating Actions)
- Weighted Pass Accuracy
- Combined Tackles + Interceptions

---

## 🚀 Future Improvements

To achieve higher prediction accuracy, we recommend the following enhancements:

### 1. **Data Augmentation & External Sources**
- Incorporate opponent team statistics (their xG, form, injuries)
- Add historical head-to-head records against specific opponents
- Include player injury/availability data
- Add manager tenure and tactical changes data
- Weather conditions at match time

### 2. **Advanced Feature Engineering**
- **Rolling Averages:** Calculate form over last 5/10 matches (goals scored, xG, points)
- **Momentum Features:** Win/loss streaks, points in last N games
- **Fatigue Index:** Days since last match, travel distance for away games
- **Key Player Impact:** Performance when captain/star players are present vs absent
- **Opponent Strength:** League position, recent form of opposition

### 3. **Model Improvements**
- **Class Imbalance Handling:** Use SMOTE or class weights (Draws are underrepresented)
- **Hyperparameter Tuning:** GridSearchCV or RandomizedSearchCV for optimal parameters
- **Ensemble Methods:** Combine multiple models (XGBoost, LightGBM, Neural Networks)
- **Gradient Boosting:** Try XGBoost or CatBoost which often outperform Random Forest

### 4. **Alternative Approaches**
- **Ordinal Classification:** Treat W/D/L as ordered outcomes (W > D > L)
- **Poisson Regression:** Model goals scored/conceded separately, then derive result
- **Time Series Models:** LSTM/GRU for sequence prediction using match history
- **Probabilistic Models:** Output probability distributions instead of single predictions

### 5. **Evaluation Enhancements**
- Use Log Loss instead of accuracy for probabilistic evaluation
- Implement Brier Score for calibration assessment
- Compare against baseline (e.g., betting odds, always predicting "Win")

### 6. **Code & Pipeline Improvements**
- Create modular Python scripts for reproducibility
- Implement MLflow or similar for experiment tracking
- Build a prediction pipeline for new incoming match data
- Add unit tests for data preprocessing functions

---

## 📁 Repository Structure

```
football-match-prediction/
├── data/
│   ├── Arsenal_team_data.xlsx          # Team-level match statistics
│   └── Arsenal_Player_data_for_6_years.xlsx  # Player-level statistics
├── notebooks/
│   └── Football_Prediction.ipynb       # Main analysis notebook
└── README.md                           # Project documentation
```

---

## 🖥️ How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/football-match-prediction.git
   cd football-match-prediction
   ```

2. Open the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/Football_Prediction.ipynb
   ```

3. Run all cells sequentially to reproduce the analysis.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - Machine learning models
- **Matplotlib & Seaborn** - Data visualization
- **Google Colab** - Development environment

---
## 📄 License

This project is for educational purposes as part of the MI6228 Analytics Live module.

---

## 📚 References

- Data sourced from FBref and official Premier League statistics
- Scikit-learn documentation: https://scikit-learn.org/
- Expected Goals (xG) methodology: https://fbref.com/en/expected-goals-model-explained/