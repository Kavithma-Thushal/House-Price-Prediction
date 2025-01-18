from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Print model coefficients
    print(f"Coefficient: {model.coef_[0]}, Intercept: {model.intercept_}")

    return model, X_train, X_test, y_train, y_test
