# 🌡️ Temperature Data Forecasting: Statistical Decomposition and LSTM

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-yellowgreen.svg)](https://scikit-learn.org/)
[![Champion MAPE](https://img.shields.io/badge/Champion%20Test%20MAPE-2.67%25-brightgreen.svg)]()

## 🚀 Project Summary

This project focuses on forecasting daily temperatures using a hybrid machine learning approach. We combine **Time-Series Decomposition (STL)** with a **Long Short-Term Memory (LSTM)** neural network.

A major challenge in weather forecasting is separating repeating seasonal patterns from complex climate trends. Our pipeline solves this by decomposing the data locally to prevent data leakage. We train the LSTM model strictly on the underlying trend, and then re-apply the seasonal patterns to reliably forecast temperatures 365 days into the future.

---

## ✨ Key Features & Code Architecture

* **Clean Helper Functions:** The main pipeline logic is organized into clean, reusable functions (`deseasonalize_data`, `build_lstm_model`, `generate_future_forecast`, and `reverse_scaling_and_seasonality`).
* **Automated Scenario Loops:** We evaluate three different Train/Test split ratios (**80/20**, **70/30**, and **60/40**). The code uses Python dictionaries (`results`, `model_data`, `trained_models`) to store data cleanly without overwriting variables.
* **Leak-Free Validation:** Seasonal patterns are extracted strictly inside the training sets to ensure our test evaluation remains 100% fair and accurate.
* **Rolling Future Forecasts:** The model uses its own predictions to feed back into the input sequence, allowing it to forecast a full year (365 days) into the future.
* **Side-by-Side Strategy Comparison:** The code generates and plots two different future forecasting strategies (`_train_season` vs. `_full_season`) on the same chart for direct comparison.

---

## 📁 Dataset & Preprocessing Pipeline

The dataset is sourced from **BMKG Stasiun Meteorologi Kemayoran** and is included directly in this repository as `data.csv`.

### Data Columns
- **`Tanggal`:** The date of the recorded weather (`YYYY-MM-DD`), used as the dataframe index.
- **`Temperatur rata-rata (°C)`:** The daily average temperature recorded in degrees Celsius.

### Preprocessing Steps
Before feeding data into the neural network, we apply three main steps:
1. **Exploratory Data Analysis (EDA):** We temporarily split the date into `Day`, `Month`, and `Year` columns to visualize historical temperature patterns using bar charts.
2. **Removing Seasonality:** We use multiplicative decomposition (`period=365`) to separate the seasonal waves from the main trend. The raw temperatures are divided by the seasonal wave to create clean, deseasonalized training data.
3. **Scaling and Windowing:** The data is scaled to a `0 to 1` range using `MinMaxScaler`. We then format it into sequence batches using a sliding window of **`look_back = 30`** days, preparing it perfectly for the LSTM layers.

---

## 📊 Model Evaluation & Performance

The models are trained using Mean Squared Error (MSE) loss and the Adam optimizer. To evaluate true performance, our test predictions are rescaled back to original Celsius values and re-multiplied by the seasonal wave.

| Scenario | Train MSE | Test MSE | Train MAE | Test MAE | Train MAPE | Test MAPE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **80/20** | 0.5121 | 1.1641 | 0.5677 | 0.8653 | 1.99% | 2.96% |
| **🏆 70/30** | **0.4487** | **0.9648** | **0.5141** | **0.7676** | **1.80%** | **2.67%** |
| **60/40** | 0.4557 | 1.1529 | 0.5148 | 0.8312 | 1.81% | 2.91% |

### 🏆 Champion Model Selection
Based on the metrics above, the **70/30 Split** is our official **Champion Model**:
1. **Best Accuracy:** It achieved the lowest overall error on unseen data, resulting in a **Test MAPE of 2.67%** and an average error of just **0.7676°C**.
2. **Perfect Context Balance:** The 80/20 split suffered from slight tail-end overfitting on a shorter test set, while the 60/40 split gave the model too little historical training data to learn long-term patterns effectively.

---

## 🚀 Future Forecasting Strategies (365 Days)

When forecasting into the unknown future, our final code plots two different paths to show a classic data science trade-off: **Visual Smoothness vs. Calendar Accuracy**.

* **Approach A: Training Split Seasonality (`future_predict_train`)**
  We reuse the seasonal pattern straight from the end of the training split. **Pros:** The forecast connects perfectly smoothly to the test set line on the chart. **Cons:** The peak calendar months might be slightly shifted out of phase.
* **Approach B: Full 3-Year Seasonality (`future_predict_full`)**
  We extract the seasonal pattern using the full averaged 3-year dataset. **Pros:** It filters out random historical weather noise and locks future forecasts perfectly to true real-world calendar months. **Cons:** It causes a minor, mathematically expected visual jump where the future forecast begins.

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
To execute the full forecasting pipeline locally, clone this repository to your local environment:
```bash
git clone [https://github.com/username/temperature-forecasting-hybrid.git](https://github.com/username/temperature-forecasting-hybrid.git)
cd temperature-forecasting-hybrid
```

### 2. Fetch the Dataset Directly (Optional)
If you are running experiments in an external cloud environment (such as Google Colab) and strictly need to fetch the core BMKG Kemayoran temperature dataset without cloning the entire codebase, execute the following command:

```bash
curl -L -o data.csv [https://github.com/username/temperature-forecasting-hybrid/raw/main/data.csv](https://github.com/username/temperature-forecasting-hybrid/raw/main/data.csv)
```

### 3. Install Environment Dependencies
Ensure your execution environment has the required scientific packages configured:

```bash
pip install tensorflow pandas numpy scikit-learn statsmodels matplotlib
```

### 4. Execute the Forecasting Pipeline
Launch the main execution notebook to reproduce all data scaling routines, leak-free validation loops, and comparative output visualizations:

```bash
jupyter notebook Final_Forecasting_Pipeline.ipynb
```

---

## 👥 Credits & Attribution

This project was originally developed as a Final Project assignment at Institut Teknologi Sepuluh Nopember (ITS) by **Group 3**:
* **Shabrina Nur Ihsani** (5026221002)
* **Nailah Azzahra** (5026221010)
* **Kayla Kirani Kusnadi** (5026221111)

### 🛠️ Production Refactoring & Optimization
To transform the initial working prototype into a scientifically rigorous, production-ready pipeline, **Shabrina Nur Ihsani** completed code refactoring and architectural upgrades:
* **Functional Pipeline Tidying:** Cleaned up unstructured procedural code into highly organized, modular helper functions for processing, modeling, and plotting.
* **Preventing Weight Inheritance (State Leakage):** Fixed a critical model evaluation flaw where multiple scenarios were trained sequentially on the same Keras instance. The loop now guarantees fresh weight initialization (`build_lstm_model`) for every split to ensure fair comparative metrics.
* **Leak-Free Decomposition Boundaries:** Eliminated input data leakage risks by ensuring that multiplicative time-series decomposition happens strictly inside isolated training boundaries.
* **Automated Validation Loops:** Replaced manual script execution with clean Python dictionary loops to seamlessly train and store independent datasets, scalers, and models (80/20, 70/30, 60/40) in one execution pass.
* **Bilingual Documentation:** Upgraded repository presentation and metrics reporting to international standards.

📖 **Read the Full Story:** Lessons learned behind this refactoring -> link
