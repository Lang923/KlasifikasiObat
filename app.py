import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import io
from sklearn.preprocessing import LabelBinarizer

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
with st.expander('📄 Dataset'):
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
        st.markdown(f"**Penjelasan: Distribusi {column}**\n")
        st.markdown(f"Grafik ini menunjukkan distribusi nilai untuk fitur `{column}`. Pada plot ini, kita dapat melihat sebaran data untuk fitur usia (`Age`) dan rasio Na/K (`Na_to_K`). Plot ini juga mencakup estimasi distribusi kepadatan (kde) yang membantu kita memahami pola data.")

# Plot countplot for categorical columns
with col2:
    for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
        fig, ax = plt.subplots()
        sns.countplot(x=data[column], palette='viridis', ax=ax)
        ax.set_title(f'Jumlah Data per Kategori {column}')
        st.pyplot(fig)
        st.markdown(f"**Penjelasan: Jumlah Data per Kategori {column}**\n")
        st.markdown(f"Grafik ini menunjukkan distribusi frekuensi dari kategori dalam kolom `{column}`. Countplot ini memberikan gambaran tentang seberapa banyak masing-masing kategori muncul dalam dataset, seperti jenis kelamin, tekanan darah, kadar kolesterol, dan jenis obat.")

# Scatter plot for numerical features
st.header('📍 Scatter Plot antara Usia dan Rasio Na/K')
fig, ax = plt.subplots()
sns.scatterplot(x=data['Age'], y=data['Na_to_K'], hue=data['Drug'], palette='viridis', ax=ax)
ax.set_title('Usia vs Rasio Na/K')
st.pyplot(fig)
st.markdown("""
    **Penjelasan: Scatter Plot Usia vs Rasio Na/K**
    Scatter plot ini menggambarkan hubungan antara usia (`Age`) dan rasio natrium terhadap kalium (`Na_to_K`), dengan warna yang mewakili kategori obat yang direkomendasikan (`Drug`).
    Plot ini memberikan gambaran apakah ada pola tertentu antara dua fitur numerik ini dan bagaimana distribusinya berdasarkan kategori obat yang berbeda.
""")

# Boxplot for numerical features to show outliers
st.header('📍 Boxplot untuk Fitur Numerik')
fig, ax = plt.subplots()
sns.boxplot(data=data[['Age', 'Na_to_K']], ax=ax)
ax.set_title('Boxplot Distribusi Fitur Numerik')
st.pyplot(fig)
st.markdown("""
    **Penjelasan: Boxplot untuk Fitur Numerik**
    Boxplot ini digunakan untuk menggambarkan distribusi data dari fitur numerik seperti `Age` dan `Na_to_K`. 
    Ini juga memperlihatkan adanya outlier, yaitu data yang terletak di luar batas kotak. Data ini bisa menunjukkan nilai yang tidak biasa dalam dataset yang perlu diperhatikan lebih lanjut.
""")

# Handling outliers: Remove outliers based on the IQR method
st.header('📍 Menangani Outliers')

# Calculate IQR for Age and Na_to_K
Q1 = data[['Age', 'Na_to_K']].quantile(0.25)
Q3 = data[['Age', 'Na_to_K']].quantile(0.75)
IQR = Q3 - Q1

# Define the limits for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
data_cleaned = data[((data[['Age', 'Na_to_K']] >= lower_bound) & (data[['Age', 'Na_to_K']] <= upper_bound)).all(axis=1)]

# Calculate how many outliers were removed
outliers_removed = len(data) - len(data_cleaned)
st.write(f"Jumlah data yang dibuang karena outliers: {outliers_removed} data.")
st.write(f"Jumlah data yang tersisa setelah penghapusan outliers: {len(data_cleaned)} data.")

# Visualize cleaned data
st.markdown("**Visualisasi Data Setelah Penghapusan Outliers:**")
col1, col2 = st.columns(2)

with col1:
    for column in ['Age', 'Na_to_K']:
        fig, ax = plt.subplots()
        sns.histplot(data_cleaned[column], kde=True, color='green', ax=ax)
        ax.set_title(f'Distribusi {column} setelah Penghapusan Outliers')
        st.pyplot(fig)

# Preprocess data and implement Naive Bayes
st.header('⚙️ Klasifikasi dengan Naive Bayes')

# Map categorical variables to numeric
data_cleaned['Sex'] = data_cleaned['Sex'].map({'F': 0, 'M': 1})
data_cleaned['BP'] = data_cleaned['BP'].map({'LOW': 0, 'NORMAL': 1, 'HIGH': 2})
data_cleaned['Cholesterol'] = data_cleaned['Cholesterol'].map({'NORMAL': 0, 'HIGH': 1})

st.write("Data Setelah Encoding: ")
st.dataframe(data_cleaned.head())

# Define features and target
X = data_cleaned[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data_cleaned['Drug']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
st.markdown(""" 
    **Akurasi** mengukur seberapa banyak prediksi yang benar dibandingkan dengan total data yang diuji. 
    Nilai akurasi sebesar 91,38% menunjukkan bahwa model ini benar dalam 91,38% dari total data uji yang ada.
""")

# Display detailed Classification Report
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

# ROC Curve for Multi-Class classification
st.subheader('📈 ROC Curve')

# Binarize the output labels for multi-class
lb = LabelBinarizer()
y_bin = lb.fit_transform(y_test)  # Menggunakan y_test yang benar

# Store the ROC AUC for each class
fig, ax = plt.subplots()
for i in range(len(lb.classes_)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], model.predict_proba(X_test)[:, i])
    roc_auc = roc_auc_score(y_bin[:, i], model.predict_proba(X_test)[:, i])
    ax.plot(fpr, tpr, label=f'Class {lb.classes_[i]} (AUC = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_title('ROC Curve for Multi-Class')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
st.pyplot(fig)

# Display AUC score for multi-class
roc_auc_multi_class = roc_auc_score(y_bin, model.predict_proba(X_test), average="macro", multi_class="ovr")
st.write(f"**AUC Score (Multi-Class):** {roc_auc_multi_class:.2f}")
st.markdown("""
    **AUC Score (Area Under the Curve)** adalah ukuran yang mengukur kualitas pemisahan model antara kelas-kelas. 
    Nilai AUC sebesar 0.98 menunjukkan bahwa model ini memiliki kemampuan yang sangat baik dalam membedakan antara kelas yang berbeda.
    AUC score mendekati 1.0 menunjukkan bahwa model mampu membedakan setiap kelas dengan sangat baik, meskipun ada beberapa kesalahan prediksi.
""")

# Allow user to input new data for prediction in the sidebar
st.sidebar.header('🧪  Klasifikasi Obat untuk Data Baru')

# Create form in the sidebar
with st.sidebar.form(key='prediction_form'):
    age = st.sidebar.number_input('Usia', min_value=0, max_value=120, value=30)
    sex = st.sidebar.selectbox('Jenis Kelamin', options=['Perempuan', 'Laki-laki'])
    bp = st.sidebar.selectbox('Tekanan Darah', options=['LOW', 'NORMAL', 'HIGH'])
    cholesterol = st.sidebar.selectbox('Kolesterol', options=['NORMAL', 'HIGH'])
    na_to_k = st.sidebar.number_input('Rasio Na ke K', min_value=0.0, max_value=50.0, value=15.0)

    submit_button = st.form_submit_button(label='Klasifikasi')

# Handle form submission
if submit_button:
    # Map inputs to numerical values
    sex = 0 if sex == 'Perempuan' else 1
    bp = {'LOW': 0, 'NORMAL': 1, 'HIGH': 2}[bp]
    cholesterol = 0 if cholesterol == 'NORMAL' else 1

    # Predict using the trained model
    new_data = [[age, sex, bp, cholesterol, na_to_k]]
    prediction = model.predict(new_data)
    st.sidebar.success(f'✅ Rekomendasi Obat: {prediction[0]}')
