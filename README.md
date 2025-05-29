# Laporan Proyek Machine Learning - Zita Natalia Arisenda

## Domain Proyek

Perusahaan sering kali mengalami kesulitan dalam menilai kinerja karyawan secara objektif karena penilaian manual rawan bias dan kurang konsisten. Hal ini dapat berdampak pada keputusan penting seperti promosi, pelatihan, hingga retensi karyawan. Dengan memanfaatkan data historis dan pendekatan prediktif, perusahaan dapat mengidentifikasi potensi karyawan sejak dini serta mengambil langkah strategis untuk meningkatkan produktivitas. Menurut Hasan et al. (2024), pendekatan terintegrasi yang menggabungkan analitik bisnis dan machine learning dapat membantu manajemen dalam membuat keputusan yang lebih tepat dan berbasis data terkait kinerja karyawan.

Referensi:
Hasan, M. R., Ray, R. K., & Chowdhury, F. R. (2024). Employee Performance Prediction: An Integrated Approach of Business Analytics and Machine Learning. Journal of Business Management Studies, 6(1), 215–223.

## Business Understanding

### Problem Statements
Menjelaskan pernyataan masalah latar belakang:
- Bagaimana cara mengidentifikasi faktor-faktor utama yang memengaruhi kinerja karyawan?
Perusahaan perlu memahami variabel-variabel penting yang berkontribusi terhadap performa kerja karyawan agar dapat meningkatkan produktivitas dan efisiensi organisasi.
- Bagaimana memprediksi skor kinerja karyawan berdasarkan atribut-atribut yang tersedia?
Penilaian kinerja secara manual membutuhkan waktu dan rentan subjektivitas. Diperlukan sistem berbasis data yang mampu memberikan prediksi kinerja karyawan secara otomatis dan objektif.

### Goals
- Menemukan fitur-fitur penting yang memengaruhi skor kinerja karyawan.
Dengan menggunakan teknik feature importance dari model berbasis pohon keputusan, mengidentifikasi variabel yang paling relevan dalam menentukan performa.
- Membangun model klasifikasi yang mampu memprediksi skor kinerja karyawan secara akurat.
Model ini diharapkan membantu manajemen dalam melakukan evaluasi dan pengambilan keputusan berbasis data.

### Solution statements
- Menerapkan dua algoritma klasifikasi: Decision Tree dan Random Forest.
Dua model pohon digunakan untuk membandingkan performa.
- Melakukan evaluasi menggunakan metrik klasifikasi.
Untuk mengukur performa model, digunakan metrik seperti accuracy, F1-score, confusion matrix, dan classification report agar solusi yang dibangun dapat dievaluasi secara kuantitatif.
- Melakukan pembagian data latih dan data uji serta validasi silang (cross-validation).
Hal ini dilakukan untuk menghindari bias dan memastikan model memiliki generalisasi yang baik terhadap data baru.

## Data Understanding

Dataset yang digunakan dalam proyek ini berjudul Extended Employee Performance and Productivity Data, dengan 100000 baris dan 20 kolom, yang dapat diunduh melalui tautan berikut: https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data/data
Dataset ini berisi informasi terkait karakteristik dan produktivitas karyawan, serta skor kinerja mereka. Tujuan dari penggunaan dataset ini adalah untuk memahami pola-pola yang berkontribusi terhadap performa kerja dan selanjutnya membangun model prediktif untuk mengklasifikasikan skor kinerja karyawan.

### Variabel pada Dataset
Berikut adalah daftar fitur (variabel) yang tersedia dalam dataset:
- Employee_ID: ID unik dari setiap karyawan.
- Department: Departemen tempat karyawan bekerja.
- Gender: Jenis kelamin karyawan.
- Age: Usia karyawan.
- Job_Title: Jabatan atau posisi pekerjaan karyawan.
- Hire_Date: Tanggal mulai bekerja di perusahaan.
- Years_At_Company: Lama bekerja di perusahaan dalam tahun.
- Education_Level: Tingkat pendidikan terakhir yang diselesaikan.
- Performance_Score: Skor performa kerja karyawan.
- Monthly_Salary: Gaji bulanan karyawan.
- Work_Hours_Per_Week: Rata-rata jam kerja per minggu.
- Projects_Handled: Jumlah proyek yang telah ditangani oleh karyawan.
- Overtime_Hours: Rata-rata jam lembur.
- Sick_Days: Jumlah hari sakit yang diambil oleh karyawan.
- Remote_Work_Frequency: Frekuensi kerja jarak jauh.
- Team_Size: Ukuran tim tempat karyawan bekerja.
- Training_Hours: Jumlah jam pelatihan yang diikuti.
- Promotions: Jumlah promosi yang telah diterima karyawan.
- Employee_Satisfaction_Score: Skor kepuasan karyawan terhadap pekerjaannya.
- Resigned: Status apakah karyawan telah mengundurkan diri.

### Eksplorasi Data dan Visualisasi
Beberapa langkah eksplorasi data dilakukan untuk memahami karakteristik data:
- Informasi Dataset dan Statistik Deskriptif.
Dataset terdiri dari kolom numerik dan kategorikal.
Tidak ditemukan nilai kosong (missing values), data duplikat, maupun outlier yang signifikan.
- Distribusi Variabel Numerik.
Dilakukan visualisasi histogram untuk variabel numerik seperti:
Monthly_Salary dan Work_Hours_Per_Week menunjukkan distribusi yang tidak merata dengan konsentrasi pada nilai tertentu.
Age, Annual_Training_Hours, dan Employee_Satisfaction_Score memiliki distribusi yang lebih seimbang.
- Distribusi Variabel Kategorikal.
Visualisasi menggunakan barplot terhadap variabel:
Department, Job_Title, dan Gender menunjukkan distribusi yang relatif merata.
Education_Level didominasi oleh tingkat Bachelor.
Resigned lebih banyak bernilai False (artinya lebih banyak karyawan yang masih aktif).

## Data Preparation

Pada tahap ini, dilakukan beberapa langkah data preparation yang penting untuk memastikan data dalam kondisi siap digunakan untuk proses pemodelan.
1. Pemeriksaan Nilai Kosong (Missing Values).
- Langkah: Dilakukan pengecekan terhadap seluruh kolom untuk mengetahui apakah terdapat nilai kosong.
- Hasil: Tidak ditemukan nilai kosong di dataset.
- Alasan: Data yang memiliki nilai kosong dapat menyebabkan error atau bias dalam proses pelatihan model. Oleh karena itu, pemeriksaan ini penting untuk menjamin integritas data.
2. Pemeriksaan Duplikasi Data.
- Langkah: Menggunakan fungsi duplicated() untuk mengecek apakah ada baris data yang terduplikasi.
- Hasil: Tidak ditemukan baris duplikat dalam dataset.
- Alasan: Data duplikat dapat memperkuat bobot data tertentu secara tidak proporsional dan menurunkan kualitas prediksi model.
3. Pemeriksaan dan Deteksi Outlier.
- Langkah: Menggunakan metode Interquartile Range (IQR) untuk mendeteksi outlier pada variabel numerik seperti usia, gaji, jam kerja, dsb.
- Hasil: Tidak ditemukan outlier.
- Alasan: Deteksi outlier penting untuk mengetahui apakah ada nilai ekstrem yang dapat mengganggu pembelajaran model.
4. Penghapusan Kolom yang Tidak Relevan.
- Langkah: Kolom Employee_ID dan Hire_Date dihapus dari dataset.
- Alasan: Employee_ID bersifat unik dan tidak mengandung informasi yang berguna untuk prediksi. Hire_Date juga bersifat individual dan belum diolah menjadi fitur yang relevan seperti masa kerja.
5. Pemisahan Fitur dan Target.
- Langkah: Kolom Performance_Score dipisahkan sebagai target (y), sedangkan kolom lainnya menjadi fitur (X).
- Alasan: Agar model dapat mempelajari hubungan antara atribut karyawan (fitur) dan skor kinerja (target) dalam proses pelatihan.
6. Encoding Variabel Kategorikal.
- Langkah: Menggunakan LabelEncoder untuk mengubah nilai kategorikal (seperti Department, Gender, Job_Title, dll) menjadi bentuk numerik.
- Alasan: Algoritma pembelajaran mesin seperti Decision Tree dan Random Forest hanya dapat menerima input numerik. Encoding diperlukan agar model dapat memproses variabel kategorikal.
7. Split Data: Train-Test Split.
- Langkah: Data dibagi menjadi data latih dan data uji dengan proporsi 80:20 menggunakan train_test_split.
- Alasan: Pemisahan ini bertujuan untuk menguji generalisasi model terhadap data yang belum pernah dilihat sebelumnya.

## Modeling

Pada tahap ini, dilakukan pembangunan model machine learning untuk memprediksi skor kinerja karyawan berdasarkan data historis dan atribut-atribut. Dua algoritma digunakan dan dibandingkan performanya: Decision Tree Classifier dan Random Forest Classifier.

### Decision Tree Classifier
Decision Tree bekerja dengan membagi data secara rekursif berdasarkan fitur yang paling memisahkan target. Pembagian ini dilakukan menggunakan Gini impurity, yang merupakan metode default untuk mengukur kualitas pemisahan dalam algoritma Decision Tree di scikit-learn. Pada setiap simpul, algoritma memilih fitur terbaik dan titik potong terbaik untuk memisahkan data. Proses ini berlangsung hingga tidak ada lagi informasi yang bisa dipisahkan.

- Tahapan:
Model Decision Tree dibangun menggunakan pustaka sklearn.tree.DecisionTreeClassifier.
Model dilatih menggunakan data hasil split (fitur dan target).
Evaluasi dilakukan menggunakan metrik accuracy dan classification report.

- Parameter Default:
criterion='gini',
max_depth=None,
random_state=42.

- Kelebihan:
Mudah dipahami dan divisualisasikan.
Dapat menangani data numerik dan kategorikal.
Tidak memerlukan normalisasi fitur.

- Kekurangan:
Cenderung overfitting pada data latih.
Sensitif terhadap perubahan kecil pada data.

### Random Forest Classifier
Random Forest adalah ensemble learning yang membangun banyak pohon keputusan (dalam proyek ini, 100 pohon) menggunakan data acak. Setiap pohon dilatih dengan subset acak dari fitur dan data, lalu hasil prediksi tiap pohon dikombinasikan. Metode ini efektif untuk menghindari overfitting karena variasi antar pohon dan stabil dalam performa.

- Tahapan:
Model Random Forest dibangun menggunakan RandomForestClassifier dari sklearn.ensemble.
Model ini merupakan ensemble dari beberapa Decision Tree.
Sama seperti sebelumnya, evaluasi dilakukan menggunakan akurasi dan metrik klasifikasi.

- Parameter Default:
n_estimators=100,
criterion='gini',
random_state=42.

- Kelebihan:
Lebih stabil dan akurat dibandingkan Decision Tree tunggal.
Mampu mengatasi overfitting karena menggunakan banyak pohon.
Memberikan skor pentingnya fitur (feature importance).

- Kekurangan:
Lebih kompleks dan membutuhkan waktu pelatihan lebih lama.
Interpretasi model menjadi lebih sulit dibanding satu pohon.

### Pemilihan Model Terbaik
Model Random Forest Classifier dipilih sebagai model terbaik karena:
Memberikan hasil prediksi yang lebih akurat dan konsisten, memiliki kemampuan generalisasi yang lebih baik, dan mengurangi risiko overfitting dengan metode ensemble learning.

## Evaluation

Tahap evaluasi ini bertujuan menilai seberapa baik model yang dibangun mampu memprediksi performa karyawan berdasarkan fitur-fitur. Evaluasi dilakukan dengan beberapa metrik klasifikasi, validasi silang (cross-validation), dan analisis feature importance.

### Metrik Evaluasi yang Digunakan
Karena target prediksi berupa klasifikasi multikelas terhadap skor performa karyawan (performance_score), digunakan metrik evaluasi berikut:

1. Accuracy: Persentase jumlah prediksi yang benar terhadap total seluruh prediksi.
   Formula:
![](https://raw.githubusercontent.com/zitaarisenda/Prediktif/main/Screenshot%202025-05-26%20211702.png)

2. Precision: Seberapa banyak prediksi positif yang benar dari total prediksi positif yang dibuat oleh model.
   Formula:
​![](https://raw.githubusercontent.com/zitaarisenda/Prediktif/main/Screenshot%202025-05-26%20211712.png)

3. Recall: Seberapa banyak kasus positif yang benar-benar dapat dideteksi oleh model.
   Formula:
![](https://raw.githubusercontent.com/zitaarisenda/Prediktif/main/Screenshot%202025-05-26%20211722.png)

4. F1-Score: Harmonic mean dari precision dan recall.
   Formula:
![](https://raw.githubusercontent.com/zitaarisenda/Prediktif/main/Screenshot%202025-05-26%20211732.png)

5. Confusion Matrix: Matriks yang menunjukkan perbandingan antara prediksi model dengan label sebenarnya.

### Hasil Evaluasi Model
1. Decision Tree Classifier
- Akurasi Training: 1.0000
- Akurasi Testing: 1.0000
- Classification Report:
- Semua metrik (precision, recall, F1-score) menunjukkan nilai 1.00 pada seluruh kelas.
- Confusion Matrix: Tidak ada kesalahan prediksi.

2. Random Forest Classifier
- Akurasi Training: 1.0000
- Akurasi Testing: 1.0000
- Classification Report:
- Semua metrik (precision, recall, F1-score) mencapai 1.00 pada seluruh kelas.
- Confusion Matrix: Semua prediksi akurat.

### Cross Validation
Metode:
Dilakukan Stratified K-Fold Cross Validation sebanyak 5 fold.
Digunakan untuk memastikan bahwa model tidak overfitting dan performanya stabil di berbagai subset data.

Hasil:
1. Decision Tree
- Training Score: ~1.0 flat di semua ukuran data.
- Validation Score: Juga ~1.0 flat di semua ukuran data.
- Mean CV Score: 1.0, Std = 0.0. (indikasi kuat overfitting)

Decision Tree sangat mudah overfit, terutama jika tidak menggunakan pruning atau pembatasan kedalaman pohon. Kemungkinan besar model menghafal data, bukan belajar pola. Model mudah overfit karena hanya satu pohon yang menghafal data, sangat baik di data training tapi gagal generalisasi ke data baru.

2. Random Forest
- Training Score: ~1.0
- Validation Score: ~0.999
- Mean CV Score: 0.9999, Std = 0.0001

Learning curve menunjukkan kurva training dan validation hampir menyatu, bahkan di data kecil. Model sangat baik dan stabil. Model dapat menghindari overfit dengan menggabungkan banyak pohon. Akurasi tinggi wajar karena data bersih, fitur sangat informatif, dan tidak ada masalah imbalance.

### Feature Importance
Dari hasil pelatihan model Random Forest, dilakukan analisis terhadap fitur yang paling berpengaruh terhadap hasil prediksi.

Top 5 Fitur Paling Berpengaruh:
- Monthly_Salary
- Job_Title
- Employee_Satisfaction_Score
- Training_Hours
- Projects_Handled

Interpretasi:
Fitur Monthly_Salary menjadi yang paling dominan, menunjukkan bahwa kompensasi karyawan berkaitan erat dengan skor performa mereka. Ini bisa mencerminkan bahwa gaji selaras dengan kontribusi dan tanggung jawab kerja. Diikuti oleh jabatan yang juga berpengaruh terhadap produktivitas dan kinerja.

