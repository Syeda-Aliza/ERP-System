from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

# Load the saved model
loaded_model = joblib.load('model.joblib')

app = Flask(__name__)

# Ensure the 'static' folder exists for saving images
if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/')
def index():
    return render_template('file.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Read the uploaded CSV file
    df = pd.read_csv(file, parse_dates=['TrnsDate'])
    
    # Ensure the date column is in datetime format
    df['TrnsDate'] = pd.to_datetime(df['TrnsDate'])
    df.set_index('TrnsDate', inplace=True)
    
    # Handle missing values in 'Amount'
    df['Amount'].fillna(df['Amount'].median(), inplace=True)
    
    # Step 1: Feature Engineering
    # Adding time-related features
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Weekday'] = df.index.weekday
    
    # Step 2: Decompose the time series (Trend, Seasonal, Residual)
    decomposition = seasonal_decompose(df['Amount'], model='additive', period=12)
    decomposition_plot_filename = 'static/decomposition_plot.png'
    decomposition.plot()
    plt.tight_layout()
    plt.savefig(decomposition_plot_filename)
    plt.close()

    # Step 3: ARIMA Forecast (For visualization and comparison)
    train_size = int(len(df) * 0.8)
    train, test = df['Amount'][:train_size], df['Amount'][train_size:]
       
    # Step 2: Fit ARIMA model (set your own order)
    # Start with order=(1, 1, 1) and adjust based on performance
    model = ARIMA(train, order=(1, 1, 1))
    fit = model.fit()

    # Step 3: Forecast
    forecast = fit.forecast(steps=len(test))

    # Step 4: Evaluate the ARIMA Model
    mse = mean_squared_error(test, forecast)
    print(f"Mean Squared Error: {mse}")
    
    # Visualize the forecast vs actual for ARIMA
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[:train_size], train, label='Training Data', color='purple', linewidth=2)
    plt.plot(df.index[train_size:], test, label='Actual Sales', color='blue', linewidth=2)
    #plt.plot(df.index[train_size:], forecast, label='Forecasted Sales', color='yellow', linestyle='--', linewidth=2)

    # Add gridlines, title, and labels
    plt.title('ARIMA Forecast vs Actual Sales', fontsize=16, fontweight='bold')
    #plt.xlabel('Date', fontsize=12)
    #plt.ylabel('Sales', fontsize=12)
    #plt.legend(loc='upper left')
    #plt.grid(True, which='both', linestyle='--', alpha=0.7)

    # Save the ARIMA forecast plot
    arima_plot_filename = 'static/arima_forecast_plot_appealing.png'
    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig(arima_plot_filename)

    # Visualize the amount data
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Amount'], label='Amount', color='black', linewidth=2)
    plt.title('Monthly Sales Data', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.legend()
    #plt.grid(True, which='both', linestyle='--', alpha=0.7)

    # Save the sales plot
    plot_filename = 'static/sales_data_plot_appealing.png'
    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig(plot_filename)
    

    #df.columns = df.columns.str.strip()  # Removes any leading/trailing spaces
    
    X = df.drop(['Amount'], axis=1)  # Features
    y = df['Amount']  # Target variable

    # Handle categorical features encoding
    categorical_features = X.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_encoded = encoder.fit_transform(X[categorical_features])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

    # Concatenate the encoded columns back into the original dataset
    X = X.drop(columns=categorical_features).reset_index(drop=True)
    X = pd.concat([X, X_encoded_df], axis=1)
    

    # Split the data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle missing values and scaling
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)  # Impute on training data
    X_test_imputed = imputer.transform(X_test)  # Apply same transformation to test data

    # Fit the scaler on the training data and apply scaling to both X_train and X_test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    
    
    # Train the KNN model
    knn_regressor = KNeighborsRegressor(n_neighbors=5)
    knn_regressor.fit(X_train_scaled, y_train)

    # Make predictions
    predictions = knn_regressor.predict(X_test_scaled)


    # Step 7: Evaluate the KNN Model
    r2 = r2_score(y_test, predictions)
    print(f"R² Score: {r2:.2f}")
    print(f"Predictions: {predictions[:5]}")  # Show the first 5 predictions

    # Prepare data for chart (labels, prediction values)
    prediction_data = {
        'labels': X_test.apply(lambda row: ', '.join(row.astype(str)), axis=1).tolist(),
        'values': predictions.tolist()
    }
    
    # Ensure the 'static' directory exists before saving plots
    #if not os.path.exists('static'):
        #os.makedirs('static')
    
    
    # Prepare prediction summary and confidence using the R² score
    prediction_summary = f"Sales are predicted with an R2 score of {r2:.2f}."
    prediction_confidence = f"Model Confidence (R2 Score): {r2:.2f}"

    results = {
        'summary': prediction_summary,
        'confidence': prediction_confidence,
        'chartData': prediction_data,
        'sales_plot': url_for('send_image', filename='sales_data_plot_appealing.png'),
        'decomposition_plot': url_for('send_image', filename='decomposition_plot.png'),
        'arima_forecast_plot': url_for('send_image', filename='arima_forecast_plot_appealing.png')
    }
    
    print(results)  # Log the results to the console for debugging
    return jsonify(results)

    
# Removed the item_predict endpoint
@app.route('/static/<filename>')
def send_image(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
