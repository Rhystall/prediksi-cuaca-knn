# Data training Cuaca sebagai Contoh
data = [
    (30, 70, "Cerah"),
    (28, 90, "Hujan"),
    (27, 85, "Hujan"),
    (33, 65, "Cerah"),
    (26, 80, "Hujan"),
    (31, 75, "Berawan"),
    (29, 70, "Cerah"),
    (25, 90, "Hujan"),
    (32, 68, "Cerah"),
    (30, 85, "Berawan")
]

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

# Panggil fungsi prediksi
hasil = knn_predict(suhu_input, kelembapan_input)

# Tampilkan hasil prediksi
print(f"Prediksi cuaca untuk suhu {suhu_input}°C dan kelembapan {kelembapan_input}%: {hasil}")
