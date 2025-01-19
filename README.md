# README: Heart Rate Prediction using Machine Learning

## Project Overview
This project demonstrates a pipeline for predicting heart rate (in BPM) based on features such as ML2, V5, and RR interval. Multiple regression models are trained and evaluated using Python, with detailed metrics provided for each model.

---

## Dataset
The dataset used in this project is stored in a file named `heartbeat_data.csv`. It contains the following columns:
- **ML2**: Signal measurement from lead ML2.
- **V5**: Signal measurement from lead V5.
- **RR Interval (ms)**: The time interval between successive R-wave peaks.
- **Heart Rate (BPM)**: The target variable.
- **Date**: A timestamp for each observation (removed during preprocessing).

---

## Workflow

### 1. **Data Preprocessing**
- The `Date` column was dropped as it was not relevant to prediction.
- Features (independent variables): `ML2`, `V5`, `RR Interval (ms)`
- Target (dependent variable): `Heart Rate (BPM)`

### 2. **Exploratory Data Analysis (EDA)**
- Histograms of features were plotted to understand their distributions.
- Correlation analysis was performed to identify relationships between features.

### 3. **Data Splitting**
- The dataset was split into training (90%) and testing (10%) subsets using `train_test_split`.

### 4. **Feature Scaling**
- StandardScaler was used to normalize the features to ensure uniform scaling.

### 5. **Model Training**
- Models implemented:
  - **Linear Regression**
  - **Lasso Regression**
  - **Ridge Regression**
  - **ElasticNet Regression**
- Each model was trained on the scaled training data and evaluated on both training and test sets.

### 6. **Evaluation Metrics**
- **Mean Absolute Error (MAE)**: Average absolute error between predicted and actual values.
- **R² Score**: Coefficient of determination. Indicates how well the model fits the data.

### 7. **New Data Prediction**
- The models were tested with new data points to predict heart rate.

---

## Results
For each regression model, the following metrics were computed:

1. **Linear Regression**:
   - MAE (Training):
   - R² (Training):
   - MAE (Testing):
   - R² (Testing):

2. **Lasso Regression**:
   - MAE (Training):
   - R² (Training):
   - MAE (Testing):
   - R² (Testing):

3. **Ridge Regression**:
   - MAE (Training):
   - R² (Training):
   - MAE (Testing):
   - R² (Testing):

4. **ElasticNet Regression**:
   - MAE (Training):
   - R² (Training):
   - MAE (Testing):
   - R² (Testing):

---

## Requirements
### Python Libraries
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

### Installation
Install the required libraries using the following command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Instructions for Use
1. Place the dataset (`heartbeat_data.csv`) in the same directory as the script.
2. Run the script to:
   - Train and evaluate models.
   - Predict heart rate for new data.
3. Modify the `new_data` array to input custom data points for prediction.

---

## Visualization
### Scatter Plots
Scatter plots were generated to compare actual and predicted values for test data. Linear relationships indicate better model performance.

### Box Plots
Box plots show the effect of scaling on features, ensuring uniform distribution for training.

---

## Future Improvements
1. **Dataset Quality**:
   - Use a larger and more diverse dataset for better generalization.
   - Ensure clean data without outliers or missing values.
2. **Model Optimization**:
   - Tune hyperparameters for better performance.
   - Implement cross-validation to evaluate model reliability.
3. **Feature Engineering**:
   - Explore additional features like age, activity level, and health conditions.

---

## Conclusion
This project demonstrates a complete pipeline for heart rate prediction using basic regression models. The code can be further extended to include advanced models and hyperparameter optimization for better accuracy.

---

## Contact
For questions or feedback, feel free to reach out:
- **Name**: Kumar Mohit
- **Email**: [kumarmohitsspn969@gmail.com](mailto:kumarmohitsspn969@gmail.com)
- **GitHub**: [github.com](https://github.com/)

