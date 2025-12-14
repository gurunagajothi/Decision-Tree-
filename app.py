import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Iris Decision Tree App", layout="wide")

st.title("🌸 Iris Flower Classification using Decision Tree")

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

# Dataset preview
st.subheader("📊 Dataset Preview")
st.dataframe(X.head())

# Correlation heatmap
st.subheader("🔍 Feature Correlation Heatmap")
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm", ax=ax1)
st.pyplot(fig1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
clf = DecisionTreeClassifier(random_state=42, max_depth=4)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Model performance
st.subheader("📈 Model Performance")
st.write(f"**Accuracy:** {acc:.3f}")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion matrix
st.subheader("🧩 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig2, ax2 = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
    cmap="Blues",
    ax=ax2,
)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
st.pyplot(fig2)

# Decision Tree visualization
st.subheader("🌳 Decision Tree Visualization")
fig3, ax3 = plt.subplots(figsize=(14, 8))
plot_tree(
    clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    ax=ax3,
)
st.pyplot(fig3)

# User input for prediction
st.subheader("🔮 Predict New Flower")

sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.0)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.4)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.5)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

if st.button("Predict"):
    sample = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred_class = clf.predict(sample)[0]
    st.success(f"🌼 Predicted Flower: **{iris.target_names[pred_class]}**")
