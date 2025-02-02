# Beijing Air Quality Forecasting

## Overview

This project focuses on predicting PM2.5 concentrations in Beijing using historical air quality and weather data. PM2.5 (particulate matter with a diameter of less than 2.5 micrometers) is a significant global issue that impacts public health, environmental sustainability, and urban planning. Accurate forecasting of PM2.5 levels enables governments and communities to take proactive measures to mitigate its adverse effects.

The goal of this project is to design and train a Recurrent Neural Network (RNN) or Long Short-Term Memory (LSTM) model to forecast PM2.5 concentrations. The dataset includes time-series data with features such as temperature, pressure, wind speed, and pollutant levels.

---

## Dataset

The dataset consists of two files:

- `train.csv`: Contains historical air quality and weather data for training the model.
- `test.csv`: Contains data for making predictions and submitting to Kaggle.

### Key Features:

- `DEWP`: Dew point temperature.
- `TEMP`: Temperature.
- `PRES`: Pressure.
- `Iws`: Cumulated wind speed.
- `Is`: Cumulated hours of snow.
- `Ir`: Cumulated hours of rain.
- `pm2.5`: PM2.5 concentration (target variable).

---

## Methodology

### Data Preprocessing

1. **Handling Missing Values:**

   - Missing values in the `pm2.5` column were filled using forward fill (`ffill`) and backward fill (`bfill`) to ensure continuity in the time-series data.
   - Other missing values were filled using interpolation.

2. **Feature Engineering:**

   - Added time-based features such as `hour`, `day_of_week`, and `month` to capture temporal patterns.
   - Applied cyclical encoding to `hour` and `month` using sine and cosine transformations to better represent their periodic nature.
   - Created interaction features such as `temp_dewp_diff` (difference between temperature and dew point) and `pres_temp_interaction` (interaction between pressure and temperature).

3. **Normalization:**

   - All features were standardized using `StandardScaler` to ensure consistent scales for model training.

4. **Reshaping for LSTM:**
   - The data was reshaped into the format `(samples, timesteps, features)` to fit the LSTM input requirements.

### Model Architecture

The best-performing model is a **2-layer Bidirectional LSTM** with the following architecture:

- **Input Layer:**
  - Input shape: `(timesteps=1, features=12)`.
- **Bidirectional LSTM Layers:**
  - Layer 1: 49 units, `return_sequences=True`.
  - Layer 2: 24 units (half of Layer 1), `return_sequences=False`.
- **Regularization:**
  - Dropout (rate=0.212) after each LSTM layer to prevent overfitting.
  - L2 regularization (penalty=0.01) on the LSTM layers to constrain large weights.
- **Output Layer:**
  - Dense layer with 1 unit for regression.

### Training

- **Optimizer:** Adam with a learning rate of 0.001.
- **Loss Function:** Mean Squared Error (MSE).
- **Callbacks:**
  - Early stopping with a patience of 10 epochs.
  - Learning rate reduction on plateau with a factor of 0.2 and patience of 5 epochs.

### Hyperparameter Tuning

- Used **Optuna** for hyperparameter optimization.
- Explored different configurations for `lstm_units`, `dropout`, `batch_size`, `learning_rate`, and `epochs`.
- Identified the best parameters: `{'lstm_units': 49, 'dropout': 0.212, 'batch_size': 32, 'learning_rate': 0.001, 'epochs': 50}`.

---

## Results

- **Validation Loss (MSE):** 5599.42
- **Validation RMSE:** 22.78
- **Kaggle Leaderboard Rank:** Top 10%
- **Training MSE:** 3867.0992
- **Training RMSE:** 62.1860

---

## Repository Structure

beijing-pm25-forecasting/
│
├── README.md # Project overview and instructions
├── requirements.txt # List of dependencies
│
├── Data/ # Folder for datasets
│ ├── train.csv # Training data
│ ├── test.csv # Test data
│ ├── sample*submission.csv # Sample submission file
│
├── Final Submission/ # Folder for my best model and submission csv
│ ├── oche_ankeli_air_quality_forecasting_starter_code*(best*model).ipynb
│ ├── oche_ankeli_air_quality_forecasting_starter_code*(best_model).py
│ ├── latsubOptuna.csv # final submission csv, dont mind the name. LOL
│
├── Scripts/ # Folder for Final Submission model
│ ├── 01_data_cleaning_script.py # Data cleaning
│ ├── 02_data_exploration_script.py. # ata exploration and preprocessing
│ ├── 02_model_training_script.py.# Model training and experimentation
│ ├── 03_submission_script.py # Generating predictions and submission
│
├── outputs/ # Folder for outputs
│ ├── plots/ # Visualizations and plots
│ │ ├── pm25_trends.png # Time-series plot of PM2.5
│ │ ├── correlation_heatmap.png # Correlation heatmap
│ │ └── ... # Additional plots
│
└── report/ # Folder for the final report
├── report.pdf # Final report in PDF format
└── references.bib # References in BibTeX format (optional)

---

---

## Setup Instructions

### Dependencies

Install the required libraries using:

```bash
pip install -r requirements.txt
```

Running the Code
Data Exploration and Preprocessing:

Run data_exploration_script.py for data exploration and preprocessing.

Data Cleaning:

Run data_cleaning_script.py to clean the data

Model Training:

Run model_training_script.py to train the model.

Generating Predictions:

Run submission_script.py to generate predictions and submit to Kaggle.

---

## Conclusion

This project successfully applied LSTM models to forecast PM2.5 concentrations in Beijing. The best-performing model was a 2-layer Bidirectional LSTM with dropout and L2 regularization, achieving an RMSE of 22.78. Key findings include:

Bidirectional LSTMs are highly effective for time-series forecasting.

Regularization techniques are essential for preventing overfitting.

## Future Work

Incorporate additional features such as weather forecasts and traffic data.

Experiment with Transformer-based architectures for time-series data.

Deploy the model as a real-time forecasting tool.

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780.

3. Kaggle. (2023). Beijing PM2.5 Forecasting Challenge. https://www.kaggle.com/competitions/beijing-pm25-forecasting

4. F. Chollet, Deep Learning with Python. Shelter Island, NY, USA: Manning Publications, 2017.

5. D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

6. Y. Bengio, P. Simard, and P. Frasconi, “Learning long-term dependencies with gradient descent is difficult,” IEEE Transactions on Neural Networks, vol. 5, no. 2, pp. 157–166, 1994.

7. K. P. Murphy, Machine Learning: A Probabilistic Perspective. Cambridge, MA, USA: MIT Press, 2012.

---
