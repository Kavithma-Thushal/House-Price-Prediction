from src.dataset import generate_dataset
from src.linear_regression import train_model
from src.model import evaluate_model

# Generate the dataset
X, y = generate_dataset()

# Train the model
model, X_train, X_test, y_train, y_test = train_model(X, y)

# Evaluate the model
evaluate_model(model, X_test, y_test)
