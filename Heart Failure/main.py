import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Title
st.title('HEART FAILURE PREDICTION')

# Sidebar for classifier selection
classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('KNN', 'SVM', 'RandomForest', 'Logistic Regression', 'Naive Bayes', 'Decision Tree')
)

# Load dataset
df = pd.read_csv("hf.csv")
X, y = df.iloc[:, :-1], df['DEATH_EVENT']

# Display dataset information
st.write("### Head of Dataset:")
st.write(df.head())

st.write("### Description of Dataset:")
st.write(df.describe())

st.write("### Correlation Heatmap:")
fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

# Pairplot as a static image to improve performance
st.write(f"### Pairplot for {classifier_name}:")
fig_pair = sns.pairplot(df, hue="DEATH_EVENT", diag_kind='kde')
fig_pair.savefig("pairplot.png")
st.image("pairplot.png")

# Function to add parameters with unique keys
def add_parameter_ui(clf_name, unique_key_suffix=''):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider(f'C (SVM) {unique_key_suffix}', 0.01, 10.0, key=f'SVM_C_{unique_key_suffix}')
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider(f'K (KNN) {unique_key_suffix}', 1, 15, key=f'KNN_K_{unique_key_suffix}')
        params['K'] = K
    elif clf_name == 'RandomForest':
        max_depth = st.sidebar.slider(f'Max Depth (RF) {unique_key_suffix}', 2, 15, key=f'RF_max_depth_{unique_key_suffix}')
        n_estimators = st.sidebar.slider(f'N Estimators (RF) {unique_key_suffix}', 1, 100, key=f'RF_n_estimators_{unique_key_suffix}')
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    elif clf_name == 'Decision Tree':
        max_depth = st.sidebar.slider(f'Max Depth (DT) {unique_key_suffix}', 2, 15, key=f'DT_max_depth_{unique_key_suffix}')
        params['max_depth'] = max_depth
    return params

params = add_parameter_ui(classifier_name, unique_key_suffix="")

# Function to get the classifier
def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1234)
    elif clf_name == 'Logistic Regression':
        clf = LogisticRegression(max_iter=200)  # Increase max_iter to avoid convergence warnings
    elif clf_name == 'Naive Bayes':
        clf = GaussianNB()
    elif clf_name == 'Decision Tree':
        clf = DecisionTreeClassifier(max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Display classifier information
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = {acc:.2f}')

# Confusion matrix heatmap
st.write(f"### Heatmap of Confusion Matrix ({classifier_name}):")
conf_matrix = confusion_matrix(y_test, y_pred)
fig_conf, ax_conf = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Event', 'Event'], yticklabels=['No Event', 'Event'], ax=ax_conf)
plt.ylabel('Actual')
plt.xlabel('Predicted')
st.pyplot(fig_conf)

# Accuracy comparison graph for all classifiers
st.write("### Accuracy Graph for All Classifiers:")

classifiers = ['KNN', 'SVM', 'RandomForest', 'Logistic Regression', 'Naive Bayes', 'Decision Tree']
accuracies = []
conf_matrices = {}

for name in classifiers:
    if name == 'KNN':
        clf_temp = KNeighborsClassifier(n_neighbors=5)
    elif name == 'SVM':
        clf_temp = SVC(C=1.0)
    elif name == 'RandomForest':
        clf_temp = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=1234)
    elif name == 'Logistic Regression':
        clf_temp = LogisticRegression(max_iter=200)
    elif name == 'Naive Bayes':
        clf_temp = GaussianNB()
    elif name == 'Decision Tree':
        clf_temp = DecisionTreeClassifier(max_depth=5, random_state=1234)

    # Fit the model and predict
    clf_temp.fit(X_train, y_train)
    y_pred_temp = clf_temp.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred_temp))
    conf_matrices[name] = confusion_matrix(y_test, y_pred_temp)

fig_acc, ax_acc = plt.subplots(figsize=(8, 6))
sns.barplot(x=classifiers, y=accuracies, palette='viridis', ax=ax_acc)
plt.title('Accuracy of Different Classifiers')
plt.ylabel('Accuracy')
plt.xlabel('Classifier')
st.pyplot(fig_acc)

# Confusion matrix heatmap for all classifiers
st.write("### Heatmap of Confusion Matrix for All Classifiers:")

fig, axes = plt.subplots(3, 2, figsize=(12, 15))  # Adjust grid for 6 classifiers
axes = axes.flatten()

for idx, (name, conf_matrix) in enumerate(conf_matrices.items()):
    sns.heatmap(
        conf_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No Event', 'Event'], yticklabels=['No Event', 'Event'],
        ax=axes[idx]
    )
    axes[idx].set_title(f'Confusion Matrix for {name}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

# Adjust layout
plt.tight_layout()
st.pyplot(fig)

# Function to get accuracy vs parameter graph for each classifier
def plot_accuracy_vs_parameter(clf_name):
    st.write(f"### Graph for {clf_name}:")
    
    # Define parameter range based on classifier
    if clf_name == 'KNN':
        param_range = range(1, 21)  # n_neighbors: 1 to 20
        accuracies = []
        for k in param_range:
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
    elif clf_name == 'SVM':
        param_range = np.linspace(0.1, 10, 10)  # C: 0.1 to 10
        accuracies = []
        for c in param_range:
            clf = SVC(C=c)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
    elif clf_name == 'RandomForest':
        param_range = range(1, 101, 10)  # n_estimators: 1 to 100 (step 10)
        accuracies = []
        for n in param_range:
            clf = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=1234)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
    elif clf_name == 'Decision Tree':
        param_range = range(1, 21)  # max_depth: 1 to 20
        accuracies = []
        for depth in param_range:
            clf = DecisionTreeClassifier(max_depth=depth, random_state=1234)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
    elif clf_name == 'Logistic Regression':
        # Simulated graph for Naive Bayes: Accuracy vs Training Set Size
        param_range = range(10, len(X_train), 10)  # Increasing training data size
        accuracies = []
        for size in param_range:
            X_subset = X_train[:size]
            y_subset = y_train[:size]
            clf = GaussianNB()
            clf.fit(X_subset, y_subset)
            y_pred = clf.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
    elif clf_name == 'Naive Bayes':
    # Simulated graph for Naive Bayes: Accuracy vs Training Set Size
        param_range = range(10, len(X_train) + 1, 10)  # Increasing training data size in steps of 10
        accuracies = []
        for size in param_range:
            X_subset = X_train[:size]
            y_subset = y_train[:size]
            clf = GaussianNB()
            clf.fit(X_subset, y_subset)
            y_pred = clf.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))


    # Plot the graph
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(param_range, accuracies, marker='o', label=f'{clf_name}')
    plt.title(f'Accuracy vs Parameter ({clf_name})')
    plt.xlabel('Parameter Value')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    st.pyplot(fig)

# Display accuracy vs parameter graph
plot_accuracy_vs_parameter(classifier_name)