import csv
import random

# Buat list untuk menyimpan data
data = []

# Ubah Map label cuaca di CSV ke bahasa Indonesia
label_map = {
    'Sunny': 'Cerah',
    'Rainy': 'Hujan',
    'Cloudy': 'Berawan',
}

# Baca data dari file CSV
with open('weather_classification_data.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        try:
            suhu = float(row['Temperature'])
            kelembapan = float(row['Humidity'])
            label = label_map.get(row['Weather Type'], row['Weather Type'])
            data.append((suhu, kelembapan, label))
        except:
            continue # Buat skip baris yang error

# Fungsi hitung jarak Euclidean (Untuk Menghitung jarak 2Dimensi antara data bara ke dataset)
def euclidean_distance(x1, x2):
    return ((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)**0.5

# Fungsi klasifikasi KNN (Mengurutkan data dari jarak yang terdekat ke terjauh, kemudian mengambil K data terdekat untuk voting))
def knn_predict(suhu, kelembapan, k=3):
    distances = []
    for row in data:
        dist = euclidean_distance((suhu, kelembapan), (row[0], row[1]))
        distances.append((dist, row[2]))
    
    # Urutkan berdasarkan jarak
    distances.sort(key=lambda x: x[0])
    
    # Ambil K label terdekat
    k_labels = [label for _, label in distances[:k]]
    
    # Voting mayoritas (Mengambil mana label yang paling sering muncul nah itu yang akan menjadi hasil prediksi)
    prediction = max(set(k_labels), key=k_labels.count)
    return prediction

# Contoh prediksi
#Data Baru
suhu_input = 30
kelembapan_input = 80

# Jumlah data di dataset
print(f"Jumlah data dalam dataset: {len(data)}")

# Tampilkan data yang akan diprediksi
print(f"Data yang akan diprediksi: Suhu = {suhu_input}°C, Kelembapan = {kelembapan_input}%")

# Panggil fungsi prediksi
hasil = knn_predict(suhu_input, kelembapan_input)

# Tampilkan hasil prediksi
print(f"Prediksi cuaca untuk suhu {suhu_input}°C dan kelembapan {kelembapan_input}%: {hasil}")
