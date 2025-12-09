# Neural Network Model for Predicting O Grade in Data Warehousing and Mining

This project develops a Feed-Forward Neural Network (FNN) to predict whether a student will achieve an **O Grade (Outstanding)** based on academic performance metrics such as assignment score, project score, mid-semester exam score, and attendance.  
The model is implemented using **TensorFlow/Keras**, trained on a dataset of 15 student records.

---

## 📝 Project Summary

This project explores how neural networks can be applied to academic analytics.  
A lightweight deep learning model is built to classify students into:

- **O Grade (1)**
- **Not O Grade (0)**

The model uses four key input parameters and learns patterns that influence high academic performance.  
It demonstrates how AI can support decision-making in educational contexts.

---

## 🔍 Features

- Feed-Forward Neural Network with two hidden layers  
- Normalization of student performance data  
- Train-test data split (80/20)  
- Prediction of O Grade probability  
- Interactive user input prediction  

---

## 📊 Dataset Overview

The dataset contains 15 student entries with:

- Assignment Score (%)
- Project Score (%)
- Mid-Semester Exam Score (%)
- Attendance (%)
- Final Grade (O = 1, Not O = 0)

---

## 🧠 Neural Network Architecture

- **Input Layer:** 4 neurons  
- **Hidden Layer 1:** 8 neurons, ReLU  
- **Hidden Layer 2:** 6 neurons, ReLU  
- **Output Layer:** 1 neuron, Sigmoid  
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Epochs:** 200  

---

## 🚀 Code Implementation

```python
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split 

df = pd.read_csv("students_data.csv")

X = df[['Assignment Score (%)', 'Project Score (%)', 
        'Mid-Semester Exam (%)', 'Attendance (%)']].values 
Y = df[['Final Grade (O = 1, Not O = 0)']].values 

X = X / 100.0 

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(8, activation='relu', input_shape=(4,)),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=200, validation_data=(X_test, Y_test))

loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

---

## 📈 Output

The model prints prediction accuracy and allows interactive testing with user inputs.

---

## ✔️ Conclusion

This project shows that even a simple neural network can effectively predict high academic performance based on structured student data.  
It provides a foundation for more advanced models in educational data mining.

---

## 👤 Author

Mytri J  
Data Warehousing & Mining — Neural Network Case Study  
