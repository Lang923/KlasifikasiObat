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
    page_title="Klasifikasi Obat dengan Naive Bayes",
    page_icon="💊",
    layout="wide"
)

# Title and description
st.title('💊 Klasifikasi Rekomendasi Obat dengan Algoritma Naive Bayes')
st.markdown("""
    **Aplikasi ini menggunakan algoritma Naive Bayes** untuk Mengklasifikasikan jenis obat yang direkomendasikan
    berdasarkan profil pasien seperti usia, jenis kelamin, tekanan darah, kolesterol, dan rasio natrium ke kalium.
""")

# Load dataset
with st.sidebar.expander('📄 Dataset'):
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
        ax.set_title(f'Distribusi {column}')
        st.pyplot(fig)

# Plot countplot for categorical columns
with col2:
    for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
        fig, ax = plt.subplots()
        sns.countplot(x=data[column], palette='viridis', ax=ax)
        ax.set_title(f'Jumlah Data per Kategori {column}')
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
st.pyplot(fig)

# Allow user to input new data for prediction
st.header('🧪  Klasifikasi Obat untuk Data Baru')

with st.form(key='prediction_form'):
    age = st.number_input('Usia', min_value=0, max_value=120, value=30)
    sex = st.selectbox('Jenis Kelamin', options=['F', 'M'])
    bp = st.selectbox('Tekanan Darah', options=['LOW', 'NORMAL', 'HIGH'])
    cholesterol = st.selectbox('Kolesterol', options=['NORMAL', 'HIGH'])
    na_to_k = st.number_input('Rasio Na ke K', min_value=0.0, max_value=50.0, value=15.0)

    submit_button = st.form_submit_button(label='Klasifikasi')

if submit_button:
    # Map inputs to numerical values
    sex = 0 if sex == 'F' else 1
    bp = {'LOW': 0, 'NORMAL': 1, 'HIGH': 2}[bp]
    cholesterol = 0 if cholesterol == 'NORMAL' else 1

    # Predict using the trained model
    new_data = [[age, sex, bp, cholesterol, na_to_k]]
    prediction = model.predict(new_data)
    st.success(f'✅ Rekomendasi Obat: {prediction[0]}')
