import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime

# Set style for seaborn
sns.set_style("whitegrid")

# Load data
#@st.cache
def load_data():
    df_day = pd.read_csv('https://raw.githubusercontent.com/Roysintax/proyek-analisis-data---bangkit/main/Dataset/day.csv')
    df_hour = pd.read_csv('https://raw.githubusercontent.com/Roysintax/proyek-analisis-data---bangkit/main/Dataset/hour.csv')
    # Fill missing values
    df_hour.fillna(method='ffill', inplace=True)
    df_day.fillna(method='ffill', inplace=True)
    # Convert 'dteday' to datetime format to ensure compatibility with date input
    df_day['dteday'] = pd.to_datetime(df_day['dteday'])
    return df_day, df_hour

df_day, df_hour = load_data()

# Sidebar for dataset selection and date range input
st.sidebar.title("Visualisasi Data")
analysis_type = st.sidebar.selectbox("Pilih Analisis:", ['Visualisasi', 'Clustering'])

if analysis_type == 'Clustering':
    # Only show clustering options if 'Clustering' is selected
    num_clusters = st.sidebar.slider('Pilih Jumlah Kluster:', min_value=2, max_value=10, value=4)
    # Ensure date inputs are within the range of available data
    start_date = st.sidebar.date_input('Tanggal Mulai', value=df_day['dteday'].min())
    end_date = st.sidebar.date_input('Tanggal Akhir', value=df_day['dteday'].max())

    # Convert user input dates to datetime format
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if st.sidebar.button('Lakukan Clustering'):
        # Filter data based on selected date range
        df_filtered = df_day[(df_day['dteday'] >= start_date) & (df_day['dteday'] <= end_date)].copy()
        
        # Select features for clustering
        features = df_filtered[['temp', 'atemp', 'hum', 'windspeed', 'cnt']]
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        df_filtered['cluster'] = kmeans.fit_predict(features)
        
        # Display results
        st.write("Hasil Clustering dari", start_date, "hingga", end_date)
        st.write(df_filtered[['dteday', 'cluster']])
        
        # Plot the clusters
        fig, ax = plt.subplots()
        sns.scatterplot(x='temp', y='cnt', data=df_filtered, hue='cluster', palette='viridis', ax=ax)
        plt.title('Clustering Hasil Peminjaman Sepeda Berdasarkan Suhu dan Jumlah Peminjaman')
        plt.xlabel('Suhu')
        plt.ylabel('Jumlah Peminjaman')
        st.pyplot(fig)
else:
    # Show visualization options if 'Visualisasi' is selected
    option = st.sidebar.selectbox("Pilih Dataset yang ingin dianalisis:", ['Jumlah Peminjaman per Jam', 'Jumlah Peminjaman per Hari'])
    
    # Visualization for Hourly Rentals
    if option == 'Jumlah Peminjaman per Jam':
        st.write("Distribusi Peminjaman Sepeda per Jam: Hari Kerja vs Akhir Pekan")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='hr', y='cnt', hue='workingday', data=df_hour, ax=ax)
        plt.title('Distribusi Peminjaman Sepeda per Jam: Hari Kerja vs Akhir Pekan')
        plt.xlabel('Jam')
        plt.ylabel('Jumlah Peminjaman')
        st.pyplot(fig)
    
    # Visualization for Daily Rentals
    elif option == 'Jumlah Peminjaman per Hari':
        st.write("Perbandingan Jumlah Peminjaman Sepeda: Hari Kerja vs Akhir Pekan")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(x='weekday', y='cnt', data=df_day, ax=ax2)
        plt.title('Perbandingan Jumlah Peminjaman Sepeda: Hari Kerja vs Akhir Pekan')
        plt.xlabel('Hari dalam Seminggu (0=Sunday, 6=Saturday)')
        plt.ylabel('Rata-rata Jumlah Peminjaman')
        st.pyplot(fig2)
