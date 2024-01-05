import numpy as np
import matplotlib.pyplot as plt
# import svm form sklearn
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_binary_data(input_ranges, n_samples):
    # Create binary data based on a simple condition (e.g., sum of conditions)
    conditions = np.random.uniform(low=input_ranges[:, 0], high=input_ranges[:, 1], size=(n_samples, len(input_ranges)))
    response = (np.sum(conditions, axis=1) > np.sum(input_ranges[:, 1] - input_ranges[:, 0]) / 2).astype(int)
    return conditions, response

def staircase_active_learning_binary(input_ranges, n_samples_total, n_initial_samples, model, n_iterations=5):
    n_variables = len(input_ranges)
    n_samples_per_iteration = (n_samples_total - n_initial_samples) // n_iterations

    # Initial random samples
    initial_conditions, initial_response = create_binary_data(input_ranges, n_initial_samples)
    conditions = initial_conditions.copy()
    response = initial_response.copy()

    for _ in range(n_iterations):
        # Train the model on the current set of conditions and response
        model.fit(conditions, response)
        # Use the staircase method to select new conditions
        new_conditions = []
        new_response = []
        for i in range(n_variables):
            # Create n new sets by adapting each condition level individually towards the opposite response
            for j in range(n_samples_per_iteration):
                modified_condition = conditions[j].copy()
                modified_condition[i] = input_ranges[i, 0] if response[j] == 1 else input_ranges[i, 1]
                new_conditions.append(modified_condition)
                new_response.append(1 - response[j])  # Opposite response

            # Also, create sets by combining conditions
            for j in range(n_samples_per_iteration):
                modified_condition = conditions[j].copy()
                modified_condition[i] = input_ranges[i, 0] if response[j] == 1 else input_ranges[i, 1]
                new_conditions.append(modified_condition)
                new_response.append(response[j])

        conditions = np.vstack((conditions, new_conditions))
        response = np.concatenate((response, new_response))

    return conditions, response

# Example usage
input_ranges = np.array([(0, 1), (0, 1)])  # Specify the input ranges for each condition
n_iterations = 50
n_samples_total = 100
n_initial_samples = 20

# Initialize a support vector machine classifier
model = svm.OneClassSVM()

for i in range(n_iterations):
    # Staircase active learning for binary classification
    staircase_conditions, staircase_response = staircase_active_learning_binary(
        input_ranges, n_samples_total, n_initial_samples, model)

    # Evaluate the model
    X_train, X_test, y_train, y_test = train_test_split(staircase_conditions, staircase_response, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    # Visualization (for 1D conditions)
    # plt.figure(figsize=(10, 6))
    # plt.scatter(staircase_conditions[:, 0], staircase_conditions[:, 1], c=staircase_response, cmap='viridis', marker='o', edgecolors='k', label='Samples')
    # plt.title('Staircase Active Learning for Binary Classification')
    # plt.xlabel('Condition 1')
    # plt.ylabel('Condition 2')
    # plt.legend()
    # plt.show()
