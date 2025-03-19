# Solar Power Generation Prediction

## Overview
This project predicts solar power generation using irradiation data and temporal features. The model is built using XGBoost optimized with Particle Swarm Optimization (PSO) to achieve high accuracy. The approach transforms limited monthly data from HOMER into detailed hourly synthetic data for comprehensive machine learning.

## Data Generation

The project addresses a common challenge in solar power prediction: limited data availability. While the HOMER simulator provided monthly average irradiation and generation values, the model requires hourly granularity for accurate predictions.

### Data Sources
- **Original Data**: Monthly average irradiation and generation values from HOMER simulator
- **Synthetic Data**: Hourly data generated using PVLib and custom modeling techniques

### Generation Process
1. **PVLib Integration**: Used PVLib's location-based modeling to generate realistic clear-sky irradiance values for the target location (Ghana coordinates: 5.9, -0.2)
2. **Scaling Technique**: Applied scaling factors to match the PVLib-generated values with the actual monthly averages from HOMER
3. **Generation Modeling**: Created synthetic generation values by applying:
   - Monthly efficiency patterns derived from HOMER data
   - Time-of-day efficiency adjustments (morning hours have better efficiency due to cooler panels)
   - System characteristics (threshold for minimum irradiance, saturation at high levels)
   - Realistic variations through controlled random factors

### Data Quality
The synthetic data preserves the key characteristics of real solar power systems:
- Zero irradiance and generation during nighttime hours
- Morning efficiency advantage due to cooler temperatures
- Midday efficiency reduction due to heat
- Daily and seasonal patterns consistent with the geographical location
- Realistic generation/irradiance ratios that match the HOMER system specifications

## Feature Engineering

To capture the complex relationship between irradiance and solar power generation, we engineered several features:

1. **Cyclical Time Features**: 
   - Sine and cosine transformations of hour, month, and day of year to preserve their cyclical nature

2. **Temporal Context**: 
   - Day of week, weekend indicator
   - Daylight and peak sun period indicators

3. **Irradiance Derivatives**: 
   - Squared irradiance to capture non-linear relationships
   - Rate of change in irradiance
   - Lagged values (previous hour, 24 hours ago)

4. **Interaction Features**: 
   - Combined irradiance and time features

## Model Development

### XGBoost with PSO Optimization
The solar power prediction model uses XGBoost optimized with Particle Swarm Optimization:

1. **XGBoost**: Selected for its ability to model non-linear relationships and handle the complex patterns in solar generation

2. **Particle Swarm Optimization**: Used to find the optimal hyperparameters:
   - 10 particles exploring 7-dimensional parameter space
   - Parameters optimized: max_depth, learning_rate, n_estimators, subsample, colsample_bytree, min_child_weight, gamma
   - 50 iterations to find optimal configuration
   - Objective: minimize mean squared error in cross-validation

3. **Evaluation Metrics**:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R² Score

### Model Pipeline
The complete machine learning pipeline includes:

1. Data preprocessing and feature engineering
2. PSO-based hyperparameter optimization
3. Model training with optimal parameters
4. Evaluation on test data
5. Feature importance analysis
6. Model persistence for deployment

## Usage

### Prediction Function
The model includes a ready-to-use prediction function that takes irradiance values and datetime information as inputs:

```python
def predict_generation(irradiance, datetime_index):
    """
    Predict solar generation given irradiance values and datetime index
    
    Args:
        irradiance: Array or Series of irradiance values
        datetime_index: Corresponding DatetimeIndex
        
    Returns:
        Array of predicted generation values
    """
    # Creates dataframe, engineers features, and returns predictions
    # See implementation in notebook for details
```

### Example
```python
sample_dates = pd.date_range(start="2025-01-01", periods=24, freq="h")
sample_irradiance = [0, 0, 0, 0, 0, 50, 200, 400, 600, 800, 900, 950, 
                     900, 800, 700, 500, 300, 100, 0, 0, 0, 0, 0, 0]

sample_predictions = predict_generation(sample_irradiance, sample_dates)
```

## Visualizations

The project includes several visualizations to understand the data and model performance:

1. **Daily Pattern Analysis**: Sample week showing irradiance and generation patterns
2. **Prediction Accuracy**: Scatter plot of predicted vs. actual generation values
3. **Feature Importance**: Bar chart showing the most influential features
4. **Sample Predictions**: Visualization of model predictions for new data

## Dependencies

- pandas
- numpy
- pvlib
- xgboost
- pyswarms
- scikit-learn
- matplotlib

## Future Improvements

- Integration with real-time weather forecasting data
- Extension to include temperature effects
- Comparison with other optimization techniques (genetic algorithms, Bayesian optimization)
- Web API for online predictions
