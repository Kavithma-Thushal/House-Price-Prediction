from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Visualize predictions vs actual prices
    plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
    plt.scatter(X_test, y_pred, color='red', label='Predicted Prices')
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Size (in arbitrary units)")
    plt.ylabel("Price (in $)")
    plt.legend()
    plt.show()
