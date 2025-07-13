import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df=pd.read_csv("Pima_Indians_Diabetes_Dataset.csv")
print(df.info())
print(df.describe())
print(df.head())


print(df.isnull().sum)

sns.countplot(x='Outcome', data=df)
plt.title("Diabetes Distribution (0: No, 1: Yes)")
plt.show()

sns.pairplot(df, hue='Outcome')
plt.show()


X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

importance = model.coef_[0]
features = df.columns[:-1]  

plt.barh(features, importance)
plt.title("Feature Importance in Diabetes Prediction")
plt.show()
