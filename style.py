import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import io

# Set page config
st.set_page_config(
    page_title="Prediksi Obat dengan Naive Bayes",
    page_icon="💊",
    layout="wide"
)

# Title and description
st.title('💊 Prediksi Rekomendasi Obat dengan Algoritma Naive Bayes')
st.markdown("""
    **Aplikasi ini menggunakan algoritma Naive Bayes** untuk memprediksi jenis obat yang direkomendasikan
    berdasarkan profil pasien seperti usia, jenis kelamin, tekanan darah, kolesterol, dan rasio natrium ke kalium.
""")

# Load dataset
with st.sidebar.expander('📄 Dataset', expanded=True):
    data = pd.read_csv('Classification.csv')
    st.dataframe(data)

    st.success('Informasi Dataset')
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.markdown("**Deskripsi Statistik Dataset:**")
    st.write(data.describe())

# Visualize the dataset
st.header('📊 Visualisasi Data')
st.markdown("Berikut adalah visualisasi data yang terdapat dalam dataset.")

col1, col2 = st.columns(2)

# Plot distribution for numeric columns
with col1:
    for column in ['Age', 'Na_to_K']:
        fig, ax = plt.subplots()
        sns.histplot(data[column], kde=True, color='blue', ax=ax)
        ax.set_title(f'Distribusi {column}', fontsize=14)
        st.pyplot(fig)

# Plot countplot for categorical columns
with col2:
    for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
        fig, ax = plt.subplots()
        sns.countplot(x=data[column], palette='viridis', ax=ax)
        ax.set_title(f'Jumlah Data per Kategori {column}', fontsize=14)
        st.pyplot(fig)

# Preprocess data and implement Naive Bayes
st.header('⚙️ Klasifikasi dengan Naive Bayes')

# Map categorical variables to numeric
data['Sex'] = data['Sex'].map({'F': 0, 'M': 1})
data['BP'] = data['BP'].map({'LOW': 0, 'NORMAL': 1, 'HIGH': 2})
data['Cholesterol'] = data['Cholesterol'].map({'NORMAL': 0, 'HIGH': 1})

st.write("Data Setelah Encoding:")
st.dataframe(data.head())

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
st.subheader('🔍 Hasil Evaluasi Model')
accuracy = accuracy_score(y_test, y_pred)
st.metric(label="Accuracy", value=f"{accuracy * 100:.2f}%")

st.write("**Classification Report:**")
st.text(classification_report(y_test, y_pred))

# Display confusion matrix
st.markdown("**Confusion Matrix:**")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns