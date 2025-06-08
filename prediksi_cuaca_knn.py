# Buat list untuk menyimpan data
data = []

# Ubah Map label cuaca ke Bahasa Indonesia
label_map = {
    'Sunny': 'Cerah',
    'Rainy': 'Hujan',
    'Cloudy': 'Berawan',
}

# Baca data dari file CSV secara manual tanpa import csv
with open('weather_classification_data.csv', 'r') as f:
    lines = f.readlines()

# Ambil header (baris pertama)
header = lines[0].strip().split(',')

# Pastikan header berisi kolom yang sesuai
idx_temp = header.index("Temperature")
idx_hum = header.index("Humidity")
idx_label = header.index("Weather Type")

# Baca setiap baris selanjutnya
for line in lines[1:]:
    try:
        row = line.strip().split(',')
        suhu = float(row[idx_temp])
        kelembapan = float(row[idx_hum])
        label = label_map.get(row[idx_label], row[idx_label])
        data.append((suhu, kelembapan, label))
    except:
        continue  # Skip baris error


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


#Evaluasi Model
#Split data menjadi data latih dan data uji manual
def split_data(dataset, test_ratio=0.2):
    total = len(dataset)
    test_size = int(total * test_ratio)
    
    # Shuffle data manual
    shuffled = dataset.copy()
    for i in range(len(shuffled)):
        j = (i * 17 + 13) % total
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

    return shuffled[:-test_size], shuffled[-test_size:]

#Fungsi Evaluasi Model
def evaluate_model(test_data, k=3):
    actual = []
    predicted = []

    for suhu, kelembapan, label in test_data:
        pred = knn_predict(suhu, kelembapan, k)
        actual.append(label)
        predicted.append(pred)

    unique_labels = list(set(actual))
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    accuracy = correct / len(actual)

    # Buat confusion matrix per label (sederhana)
    from collections import Counter
    label_set = set(actual + predicted)
    cm = {label: {"TP":0, "FP":0, "FN":0} for label in label_set}

    for a, p in zip(actual, predicted):
        for label in label_set:
            if a == label and p == label:
                cm[label]["TP"] += 1
            elif a == label and p != label:
                cm[label]["FN"] += 1
            elif a != label and p == label:
                cm[label]["FP"] += 1

    print(f"\nEvaluasi Model (k = {k})")
    print(f"Akurasi: {accuracy:.2f}")
    for label in cm:
        tp = cm[label]["TP"]
        fp = cm[label]["FP"]
        fn = cm[label]["FN"]
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        print(f"\nLabel: {label}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall   : {recall:.2f}")
        print(f"  F1-Score : {f1:.2f}")


# Contoh prediksi
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

# Evaluasi model
# Split dataset jadi data latih dan uji
train_data, test_data = split_data(data, test_ratio=0.3)

# Replace global data dengan data latih (agar knn_predict pakai data latih)
data = train_data

# Evaluasi model
evaluate_model(test_data, k=3)
