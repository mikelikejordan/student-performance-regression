import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv('StudentsPerformance.csv')

# Define features and target
features = ['test preparation course', 'parental level of education']
target = 'math score'

# Prepare data
X = data[features]
y = data[target]
X = pd.get_dummies(X).astype(int)
print("Encoded features preview:")
print(X.head())

# Train model
model = LinearRegression()
model.fit(X, y)

# Print coefficients and intercept
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")
print(f"Intercept: {model.intercept_}")

# Predict for a new student
new_data = pd.DataFrame({
    'test preparation course': ['completed'],
    'parental level of education': ["bachelor's degree"]
})
new_data = pd.get_dummies(new_data).astype(int).reindex(columns=X.columns, fill_value=0)
prediction = model.predict(new_data)
print(f"\nPredicted math score: {prediction[0]}")

# Visualizations
# Bar plot
edu_levels = data['parental level of education'].unique()
avg_scores = [data[data['parental level of education'] == level]['math score'].mean() for level in edu_levels]
plt.bar(edu_levels, avg_scores, color='purple')
plt.xlabel('Parental Level of Education')
plt.ylabel('Average Math Score')
plt.title('Average Math Score by Parental Education')
plt.xticks(rotation=45)
plt.show()

# Scatter plot
plt.figure(figsize=(12, 6))
for prep in data['test preparation course'].unique():
    subset = data[data['test preparation course'] == prep]
    plt.scatter(subset['parental level of education'], subset['math score'], label=prep, alpha=0.5)
plt.xlabel('Parental Level of Education')
plt.ylabel('Math Score')
plt.title('Math Scores by Parental Education and Test Prep')
plt.xticks(range(len(data['parental level of education'].unique())), data['parental level of education'].unique(), rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
