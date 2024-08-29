# Gold Price Predictor

This project is a web application that predicts gold prices using various machine learning models and provides backtesting functionality.

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/gold-price-predictor.git
   cd gold-price-predictor
   ```

2. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```
   python app.py
   ```

5. Access the application at `http://localhost:8080` in your web browser.

## Usage Guidelines

1. **Predict Gold Prices**: 
   - Fill in the form on the homepage with the required input features (Open, High, Low, Volume).
   - Click "Predict" to get the predicted gold price.

2. **Backtesting**:
   - Click the "Run Backtest" button to perform backtesting on all implemented models.
   - View the results, including MSE, Sharpe Ratio, and Total Return for each model.

3. **Visualization**:
   - Click on the "View Gold Price Chart" link to see a visualization of historical gold prices.

## Running Tests

To run the unit tests:

```
pytest tests/
```

## Project Structure

- `app.py`: Main Flask application
- `src/`: Source code for data fetching, model training, and backtesting
- `templates/`: HTML templates for the web interface
- `tests/`: Unit tests
- `config.json`: Configuration file for data fetching and model parameters

## Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.