import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("students_data.csv")

# Extract features and labels
X = df[['Assignment Score (%)', 'Project Score (%)',
        'Mid-Semester Exam (%)', 'Attendance (%)']].values
Y = df[['Final Grade (O = 1, Not O = 0)']].values

# Normalize features
X = X / 100.0

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# Build Neural Network
model = Sequential([
    Dense(8, activation='relu', input_shape=(4,)),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train, epochs=200, validation_data=(X_test, Y_test))

# Evaluate
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# User input
assignment = float(input("Assignment Score (%): "))
project = float(input("Project Score (%): "))
exam = float(input("Mid-Semester Exam (%): "))
attendance = float(input("Attendance (%): "))

# Prepare data
test_data = np.array([[assignment, project, exam, attendance]]) / 100.0

# Predict
prediction = model.predict(test_data)[0][0]
print(f"O Grade Probability: {prediction * 100:.2f}%")
print("Outcome:", "O Grade" if prediction >= 0.5 else "Not O Grade")
