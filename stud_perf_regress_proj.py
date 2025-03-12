import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset from a local CSV file for analysis
data = pd.read_csv('StudentsPerformance.csv')

# Define features (independent variables) and target (dependent variable)
features = ['test preparation course', 'parental level of education']
target = 'math score'

# Prepare data by selecting features and target, then encode categorical variables
X = data[features]  # Extract feature columns
y = data[target]    # Extract target column
X = pd.get_dummies(X).astype(int)  # Convert categorical variables to dummy variables (0/1)
print("Encoded features preview:")  # Display first few rows to verify encoding
print(X.head())

# Initialize and train a linear regression model to predict math scores
model = LinearRegression()  # Create model instance
model.fit(X, y)  # Train model on the encoded features and target

# Print model coefficients and intercept to interpret feature impacts
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")
print(f"Intercept: {model.intercept_}")

# Prepare and predict math score for a new student with specific characteristics
new_data = pd.DataFrame({
    'test preparation course': ['completed'],
    'parental level of education': ["bachelor's degree"]
})  # Create a new data point
new_data = pd.get_dummies(new_data).astype(int).reindex(columns=X.columns, fill_value=0)  # Encode to match training data
prediction = model.predict(new_data)  # Predict math score
print(f"\nPredicted math score: {prediction[0]}")

# Generate visualizations to illustrate the analysis results

# Bar plot: Show average math scores by parental education level
edu_levels = data['parental level of education'].unique()  # Get unique education levels
avg_scores = [data[data['parental level of education'] == level]['math score'].mean() for level in edu_levels]  # Calculate averages
plt.bar(edu_levels, avg_scores, color='purple')  # Create bar plot
plt.xlabel('Parental Level of Education')  # Label x-axis
plt.ylabel('Average Math Score')  # Label y-axis
plt.title('Average Math Score by Parental Education')  # Set title
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.show()  # Display the plot

# Scatter plot: Show math scores by parental education and test prep status
plt.figure(figsize=(12, 6))  # Set figure size for better visibility
for prep in data['test preparation course'].unique():  # Loop through test prep categories
    subset = data[data['test preparation course'] == prep]  # Filter data by category
    plt.scatter(subset['parental level of education'], subset['math score'], label=prep, alpha=0.5)  # Plot points
plt.xlabel('Parental Level of Education')  # Label x-axis
plt.ylabel('Math Score')  # Label y-axis
plt.title('Math Scores by Parental Education and Test Prep')  # Set title
plt.xticks(range(len(data['parental level of education'].unique())), data['parental level of education'].unique(), rotation=45)  # Set x-axis labels
plt.legend()  # Add legend for test prep categories
plt.tight_layout()  # Adjust layout to prevent label overlap
plt.show()  # Display the plot
