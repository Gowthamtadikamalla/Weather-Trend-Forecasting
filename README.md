# Weather Trend Forecasting

## PM Accelerator Mission 

The **PM Accelerator**, founded by **Dr. Nancy Li**, is dedicated to **empowering individuals** at all stages of their product management careers. Their mission focuses on **providing comprehensive education and resources** to aspiring and current product managers, with a strong emphasis on **inclusivity and diversity**.

A standout initiative is **"PMA KIDS"**, a non-profit program offering **free product management education** to teenagers from underserved families. This initiative aims to **break financial barriers** and promote **educational fairness**, with an ambitious goal of establishing **200 schools worldwide** over the next 20 years. This effort fosters a **diverse and innovative landscape** in the tech industry.  

## Project Overview
This project focuses on analyzing historical weather data and building predictive models to forecast future temperature trends. It utilizes **Prophet, LSTM, and Fine-Tuned LSTM** models, along with various data analysis techniques, to derive meaningful insights into weather patterns.

---

## Data Processing & Analysis

### **1. Data Cleaning & Preprocessing**
- Handled missing values and outliers.
- Converted timestamps to proper formats.
- Normalized features for improved model performance.

### **2. Exploratory Data Analysis (EDA)**
- Analyzed temperature and precipitation trends over time.
- Visualized distributions of wind speed and atmospheric pressure.

---

## Forecasting Models

### **1. Prophet Model**
- A time-series forecasting model that captures seasonality and trends in weather data.

### **2. LSTM Model**
- A bidirectional LSTM model trained on sequential temperature data.
- Captures long-term dependencies for better forecasting.

### **3. Fine-Tuned LSTM Model (Best Performing)**
- Enhanced architecture with additional LSTM layers and dropout.
- Optimized hyperparameters for improved accuracy.

#### ** Model Evaluation Results**
| Model               | MAE   | RMSE  |  
|---------------------|------|------|  
| Prophet            | 0.0878 | 0.1123 |  
| LSTM               | 0.0985 | 0.1415 |  
| **Fine-Tuned LSTM** | **0.0601** | **0.0886** (Best) |  

---

## Advanced Analyses

### **Climate Trend Analysis**
- Examined long-term changes in average temperatures.

### **Environmental Impact**
- Studied the correlation between air quality and weather parameters.
- Analyzed the effect of pollutants such as CO, NO₂, SO₂, and PM2.5.

### **Feature Importance**
- Identified key factors influencing temperature predictions.

### **Spatial Analysis**
- Mapped global temperature variations using **Geopandas** and **Matplotlib**.

---

