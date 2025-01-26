import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import io

st.title('Implementasi Algoritma Naive Bayes untuk Prediksi Rekomendasi Obat Berdasarkan Profil Pasien')

# Load dataset
with st.expander('Dataset'):
    data = pd.read_csv('Classification.csv')
    st.write(data)  # Display dataset

    st.success('Informasi Dataset')
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Display summary statistics
    st.success('Deskripsi Dataset')
    st.write(data.describe())

# Visualize the dataset
with st.expander('Visualisasi Data'):
    st.subheader('Distribusi Data per Fitur')

    # Plot distribution for numeric columns
    for column in ['Age', 'Na_to_K']:
        fig, ax = plt.subplots()
        sns.histplot(data[column], kde=True, color='blue', ax=ax)
        ax.set_title(f'Distribusi {column}')
        st.pyplot(fig)

    # Plot countplot for categorical columns
    for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
        fig, ax = plt.subplots()
        sns.countplot(x=data[column], palette='viridis', ax=ax)
        ax.set_title(f'Jumlah Data per Kategori {column}')
        st.pyplot(fig)

# Preprocess data and implement Naive Bayes
with st.expander('Klasifikasi Menggunakan Naive Bayes'):
    st.subheader('Preprocessing Data')

    # Map categorical variables to numeric
    data['Sex'] = data['Sex'].map({'F': 0, 'M': 1})
    data['BP'] = data['BP'].map({'LOW': 0, 'NORMAL': 1, 'HIGH': 2})
    data['Cholesterol'] = data['Cholesterol'].map({'NORMAL': 0, 'HIGH': 1})

    st.write("Data Setelah Encoding:")
    st.write(data.head())

    # Define features and target
    X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
    y = data['Drug']

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write(f"Jumlah Data Training: {len(X_train)}")
    st.write(f"Jumlah Data Testing: {len(X_test)}")

    # Train Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Display evaluation metrics
    st.subheader('Hasil Evaluasi Model')
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy: {accuracy * 100:.2f}%')

    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

    # Allow user to input new data for prediction
    st.subheader('Prediksi Obat untuk Data Baru')
    age = st.number_input('Usia', min_value=0, max_value=120, value=30)
    sex = st.selectbox('Jenis Kelamin', options=['F', 'M'])
    bp = st.selectbox('Tekanan Darah', options=['LOW', 'NORMAL', 'HIGH'])
    cholesterol = st.selectbox('Kolesterol', options=['NORMAL', 'HIGH'])
    na_to_k = st.number_input('Rasio Na ke K', min_value=0.0, max_value=50.0, value=15.0)

    # Map inputs to numerical values
    sex = 0 if sex == 'F' else 1
    bp = {'LOW': 0, 'NORMAL': 1, 'HIGH': 2}[bp]
    cholesterol = 0 if cholesterol == 'NORMAL' else 1

    # Predict using the trained model
    if st.button('Prediksi'):
        new_data = [[age, sex, bp, cholesterol, na_to_k]]
        prediction = model.predict(new_data)
        st.success(f'Rekomendasi Obat: {prediction[0]}')