# Penjelasan Proyek: "Recommendation Is a Dish Better Served Warm"

Dokumen ini supaya seluruh anggota kelompok paham proyek dari awal sampai akhir: isi paper, alasan pilihan kita, cara pakai dataset, cara membaca data, dan hubungannya dengan `notebooks/01_toy_example.ipynb`.

---

## 1. Paper ini membahas apa? (versi sederhana)

### Analogi
Bayangkan kamu pelayan restoran yang harus merekomendasikan menu.
- **Pelanggan baru** yang baru pesan 1 kali: kamu belum tahu seleranya. Ini **cold**.
- **Pelanggan tetap** yang sudah pesan 20 kali: kamu sudah hafal seleranya. Ini **warm**.

Pertanyaannya: **mulai pesanan ke berapa pelanggan bisa dianggap "sudah dikenal"?** Judul paper bermain kata dengan ini: rekomendasi lebih enak disajikan untuk yang "hangat" (warm).

Hal yang sama berlaku untuk **item**. Film yang baru ditonton 1 orang belum "dikenal" sistem.

### Masalah yang diangkat paper
Peneliti sistem rekomendasi biasanya **membuang** user/item yang interaksinya sedikit sebelum melatih model. Contohnya "buang user dengan kurang dari 5 interaksi". Batas ini disebut **threshold** (ambang).

Masalahnya:
- Penulis memeriksa paper di konferensi RecSys 2022–2024. Dari 65 paper yang melakukan filtering, **tidak ada satu pun** yang menjelaskan kenapa memilih angka threshold-nya. Angkanya ditebak.
- Threshold **terlalu tinggi** → data berguna ikut terbuang.
- Threshold **terlalu rendah** → user/item yang sebenarnya belum cukup dikenal ikut dipakai dan menjadi noise.
- Akibatnya, hasil evaluasi antar-paper jadi tidak bisa dibandingkan dengan adil.

### Solusi paper (proposed method)
Paper **tidak** membuat model rekomendasi baru. Paper membuat **cara mengukur** threshold yang tepat berdasarkan data (data-driven). Ada dua eksperimen:

| | **Item-Based** | **User-Based** |
|---|---|---|
| Pertanyaan | Item butuh berapa interaksi di data latih supaya mulai sering direkomendasikan? | User butuh riwayat sepanjang apa supaya rekomendasinya mulai akurat? |
| Yang diubah | Jumlah interaksi satu item di data latih (N = 1, 2, 3, …) | Panjang riwayat user saat minta rekomendasi (N = 1, 2, 3, …) |
| Model dilatih ulang? | **Ya**, untuk setiap N | **Tidak**, model tetap |
| Metrik | HR\*@K, NDCG\*@K (versi modifikasi) | HR@K, NDCG@K (biasa) |

Setelah itu dibuat grafik **metrik vs N**. Titik ketika grafik **naik paling tajam** adalah threshold-nya. Titik itu dicari secara otomatis dengan **sliding window regresi linear**.

### Hasil paper (Table 2)
- Threshold item sekitar **6–15 interaksi**, threshold user sekitar **4–12 interaksi**.
- Nilainya **berbeda-beda** tergantung dataset dan model. Kesimpulannya: **jangan pakai angka tebakan, ukur dulu untuk setiap dataset & model.**
- Model sekuensial (SASRec) butuh riwayat user lebih panjang daripada model klasik.

---

## 2. Kenapa kita memilih ini? (dan kenapa bukan yang lain)

Setiap pilihan di bawah ditulis dalam tiga bagian: **alasan memilih**, **kelemahan alternatifnya**, dan **kelemahan pilihan kita sendiri**. Bagian terakhir penting supaya kita siap kalau ditanya dosen.

### 2.1 Metode: framework cold-to-warm dari paper
**Alasan:**
- Perintah tugas menyarankan memakai **proposed method** paper. Proposed method paper ini adalah framework User-Based & Item-Based.
- Framework ini tidak menghasilkan rekomendasi sendiri, padahal tugas meminta **output berupa rekomendasi**. Karena itu kita pakai **model dasar** (ItemKNN) untuk menghasilkan rekomendasi, lalu framework paper dijalankan di atasnya.

**Kelemahan cara lain untuk menentukan threshold:**

| Cara lain | Kelemahan |
|---|---|
| **Angka tebakan / p-core filtering** (mis. "buang user < 5 interaksi"). Ini yang dipakai mayoritas paper. | Tidak ada dasar. Angkanya bisa terlalu tinggi (data berguna terbuang) atau terlalu rendah (noise masuk). Hasil antar-paper jadi tidak bisa dibandingkan. Inilah yang dikritik paper. |
| **Clustering** (Visnovsky dkk., 2021: K-means + Davies-Bouldin index) | Hanya diuji di 2 dataset TV. Hanya untuk user, tidak untuk item. Hasilnya bergantung pada pilihan jumlah cluster. Tidak langsung mengukur kualitas rekomendasi. |
| **Melihat grafik secara manual** ("kira-kira naiknya di sini") | Subjektif. Dua orang bisa menunjuk titik berbeda. |

**Kelemahan metode paper (pilihan kita):**
- Setup Item-Based **memotong** interaksi item, sehingga urutan konsumsi alami user ikut rusak (diakui sendiri oleh paper, Sec 4.2).
- Sliding window **sensitif terhadap noise**. Di N besar, jumlah user/item yang tersisa sedikit sehingga grafiknya bergoyang, dan lonjakan palsu bisa terpilih sebagai threshold.
- Paper tidak menyebut lebar window maupun titik mana di window yang dijadikan threshold. Kita harus berasumsi.
- Item-Based **mahal**: harus melatih ulang model sebanyak (jumlah N × jumlah item × jumlah model) kali.

### 2.2 Model: ItemKNN (+ EASE^R sebagai pembanding)
**Alasan memilih ItemKNN:**
- **Dipakai di paper**, jadi hasilnya bisa dibandingkan dengan Table 2.
- **Paling mudah dihitung tangan**: cukup hitung kemiripan antar item lalu dijumlahkan. Cocok untuk contoh perhitungan di slide.
- Tidak butuh GPU, dan hasilnya selalu sama setiap dijalankan (deterministik).
- Melatih ulang cepat (sekitar 1 detik di ML-1M), jadi Item-Based yang butuh ratusan kali latih ulang masih mungkin dijalankan.

**Kelemahan model lain di paper (untuk kebutuhan kita):**

| Model | Cara kerja singkat | Kelemahan untuk proyek ini |
|---|---|---|
| **EASE^R** | regresi linear item-ke-item dengan solusi rumus tertutup | Butuh **invers matriks** item × item, yang sulit ditunjukkan dengan hitung tangan. Komputasinya O(n³) terhadap jumlah item, jadi berat untuk dataset dengan banyak item. Tetap kita pakai sebagai pembanding di ML-1M karena masih cepat di 3.706 item. |
| **PureSVD** | faktorisasi matriks dengan SVD menjadi faktor laten | Faktor laten **sulit diinterpretasi** ("dimensi ke-3 artinya apa?"). Perhitungan SVD tidak praktis ditulis tangan di slide. Perlu memilih rank (jumlah faktor). |
| **SASRec-CE** | transformer (self-attention) yang membaca urutan riwayat | Butuh **GPU** dan waktu training lama. Hasil training **acak** (beda tiap run). Banyak hyperparameter. Melatih ulang ratusan kali untuk Item-Based tidak realistis. Perhitungan attention tidak mungkin dijelaskan langkah demi langkah dalam 20–50 menit. |

**Kelemahan ItemKNN (pilihan kita):**
- **Mengabaikan urutan** riwayat: [A, B, C] dan [C, B, A] menghasilkan rekomendasi yang sama. Paper menunjukkan model sekuensial (SASRec) berperilaku berbeda, karena butuh riwayat lebih panjang. Temuan ini tidak akan terlihat di eksperimen kita.
- Bukan model terbaik (bukan state-of-the-art).
- Cenderung bias ke item populer, karena item populer punya banyak co-occurrence.
- Matriks similarity berukuran item × item, jadi boros memori untuk katalog besar.

### 2.3 Dataset: MovieLens-1M
**Alasan:**
- **Dipakai di paper**, jadi ada angka acuan (ItemKNN: threshold user ≈ 4, item ≈ 9).
- Dataset **paling padat** di paper (density 4.47%). Threshold-nya **paling stabil antar model** (item 9–10, user 4–6), jadi paling mudah direplikasi dan dicek.
- Jumlah item kecil (3.706), sehingga matriks item × item hanya ±55 MB dan muat di laptop.
- Formatnya sederhana dan sangat populer, mudah dijelaskan.

**Kelemahan dataset lain di paper:**

| Dataset | Statistik (Table 1 paper) | Kelemahan untuk proyek ini |
|---|---|---|
| **BeerAdvocate** (ulasan bir) | 7.606 user, **22.307 item**, density 0.83% | Matriks item × item = 22.307² ≈ 500 juta angka (±2 GB), berat untuk laptop, dan invers untuk EASE sangat lambat. Domain bir kurang familiar untuk contoh presentasi. |
| **Behance** (platform karya desain) | 8.097 user, **32.434 item**, density 0.21%, rata-rata **16.8 interaksi/item** | Matriks item × item ±4 GB, paling berat. Banyak item interaksinya sedikit, sehingga hanya sedikit item tersisa untuk N besar dan grafik jadi noisy. Threshold sangat bervariasi antar model (item 6–15). |
| **Amazon Beauty** (produk kecantikan) | 26.116 user, 17.037 item, density **0.05%**, rata-rata **8.8 interaksi/user** | Sangat jarang (sparse). Kebanyakan user hanya punya beberapa interaksi, sehingga untuk N ≥ 10 jumlah user yang bisa dievaluasi tinggal sedikit dan confidence interval sangat lebar. Threshold user bervariasi 4–12 antar model. |

**Kelemahan ML-1M (pilihan kita):**
- **Sudah difilter** oleh pembuatnya: setiap user minimal 20 rating. Jadi tidak ada user yang benar-benar cold di data mentah. Di User-Based ini tidak masalah, karena riwayat dipotong sendiri menjadi N = 1, 2, …
- **Terlalu padat** dibanding data dunia nyata (e-commerce biasanya < 0.1%). Threshold dari ML-1M belum tentu berlaku di dataset lain, dan ini justru pesan paper.
- **Data lama** (2000–2003).
- **Timestamp adalah waktu memberi rating, bukan waktu menonton.** Banyak user merating puluhan film dalam satu sesi, sehingga urutan "riwayat" tidak selalu mencerminkan urutan menonton yang sebenarnya.
- Ada data rating 1–5, tapi kita (dan paper) mengabaikannya. Film yang diberi bintang 1 tetap dianggap "disukai".

### 2.4 Cara membagi data: global timepoint
**Alasan:** meniru dunia nyata, karena model hanya belajar dari masa lalu. Tidak ada kebocoran data. Ini juga cara yang dipakai paper.

**Kelemahan cara lain:**

| Cara lain | Kelemahan |
|---|---|
| **Leave-one-out** (interaksi terakhir tiap user jadi test) | Paling umum dipakai, tapi **bocor**. "Interaksi terakhir" user A bisa terjadi tahun 2000, sementara model dilatih dengan interaksi user B tahun 2002. Model jadi "melihat masa depan". |
| **Split acak** (80% train / 20% test diacak) | Lebih bocor lagi, karena urutan waktu diabaikan sama sekali. Tidak realistis. |

**Kelemahan global timepoint (pilihan kita):**
- Hanya user yang aktif setelah GT yang bisa jadi user test. Di ML-1M banyak user hanya aktif sebentar, sehingga jumlah user test lebih sedikit.
- Distribusi user test bisa berbeda dengan user train (misalnya lebih banyak user "setia" yang aktif lama).

### 2.5 Metrik: HR@K & NDCG@K tanpa sampling
**Alasan:** standar di riset rekomendasi top-K. NDCG memperhitungkan posisi rekomendasi. Dihitung terhadap **seluruh katalog** (unsampled), mengikuti paper.

**Kelemahan alternatif:**

| Alternatif | Kelemahan |
|---|---|
| **Sampled metrics** (ground truth hanya diadu dengan 100 item acak) | Lebih cepat, tapi terbukti **bias** dan bisa membalik urutan model mana yang lebih baik (Krichene & Rendle, 2020). |
| **Precision@K** | Dengan hanya 1 ground truth per user, nilai maksimalnya 1/K. Sulit dibaca dan tidak menambah informasi dibanding HR. |
| **RMSE / MAE** (error prediksi rating) | Mengukur tebakan **bintang**, bukan kualitas **daftar rekomendasi**. Tidak cocok untuk tugas next-item prediction. |

**Kelemahan pilihan kita:** HR dan NDCG hanya mengukur akurasi. Keragaman (diversity), kebaruan (novelty), dan apakah item ekor ikut direkomendasikan tidak terukur.

### 2.6 User-Based dan Item-Based
**User-Based** cukup melatih model **sekali**, jadi murah. Dijalankan untuk ItemKNN dan EASE^R.

**Kelemahan Item-Based:** harus melatih ulang model untuk setiap item × setiap N. Dengan 1000 item × 16 nilai N seperti paper, itu 16.000 kali latih ulang per model. Latih ulang biasa butuh berjam-jam.

**Solusi kita:** untuk ItemKNN tanpa pruning, "latih ulang" dihitung eksak dengan hanya menghitung ulang satu kolom similarity (lihat §7). Dengan begitu Item-Based dijalankan pada **skala paper**: 1000 item, semua user test, semua langkah successive evaluation.

**Kelemahan solusi ini:** trik tersebut hanya berlaku untuk ItemKNN. Item-Based untuk EASE^R tidak kita jalankan.

> Catatan: versi awal memakai 50 item dan 1.500 titik evaluasi. Hasilnya 0 semua, karena nilai HR\* di paper sangat kecil (±10⁻⁴–10⁻⁶) dan tidak terdeteksi oleh evaluasi sekecil itu. Karena itu kita pindah ke skala penuh.

---

## 3. Cara pakai dataset

### Mendapatkan data
1. Unduh `ml-1m.zip` dari https://grouplens.org/datasets/movielens/1m/
2. Ekstrak ke folder proyek sehingga strukturnya menjadi:
   ```
   data/raw/ml-1m/
   ├── ratings.dat   ← yang utama kita pakai
   ├── movies.dat    ← untuk menampilkan judul film
   ├── users.dat     ← tidak dipakai
   └── README
   ```
3. Folder `data/` tidak ikut di-commit ke git (ada di `.gitignore`). Jadi setiap anggota perlu mengunduh sendiri.

### Isi file
Semua file memakai pemisah `::` (dua titik dua).

**`ratings.dat`**: format `UserID::MovieID::Rating::Timestamp`
```
1::1193::5::978300760
1::661::3::978302109
```

**`movies.dat`**: format `MovieID::Title::Genres`
```
1193::One Flew Over the Cuckoo's Nest (1975)::Drama
661::James and the Giant Peach (1996)::Animation|Children's|Musical
```

**`users.dat`**: format `UserID::Gender::Age::Occupation::Zip`. File ini tidak dipakai karena metode paper hanya memakai interaksi.

### Statistik (sama dengan Table 1 paper)
| | Nilai |
|---|---|
| User | 6.040 |
| Film | 3.706 |
| Interaksi | 1.000.209 |
| Density | 4.47% (dari semua pasangan user–film, hanya 4.47% yang pernah dirating) |
| Rata-rata interaksi per user | 166 |
| Rata-rata interaksi per film | 270 |
| Rentang waktu | April 2000 – Februari 2003 |

> **Fakta menarik yang relevan dengan paper:** setiap user di ML-1M punya **minimal 20 rating**. Artinya dataset ini sudah difilter oleh pembuatnya dengan threshold 20, angka yang juga tidak dijelaskan alasannya. Ini contoh nyata masalah yang diangkat paper.

---

## 4. Cara membaca data (detail)

### Membaca satu baris
```
1::1193::5::978300760
```
| Bagian | Nilai | Arti |
|---|---|---|
| UserID | `1` | user nomor 1 |
| MovieID | `1193` | film *One Flew Over the Cuckoo's Nest (1975)*, dilihat di `movies.dat` |
| Rating | `5` | memberi bintang 5 (skala 1–5) |
| Timestamp | `978300760` | waktu dalam **Unix time** (detik sejak 1 Januari 1970) = **31 Des 2000, 22:12:40 UTC** |

Kalimat lengkapnya: *"User 1 menonton dan memberi rating film One Flew Over the Cuckoo's Nest pada malam tahun baru 2000."*

### Rating kita abaikan (implicit feedback)
Mengikuti paper, **semua rating dianggap "pernah berinteraksi"**, baik bintang 1 maupun 5. Alasannya, tugas yang diteliti paper adalah *next-item prediction*: menebak film apa yang **akan ditonton** berikutnya, bukan menebak bintangnya. Jadi data kita menjadi tabel 0/1:
- 1 = user pernah menonton film itu
- 0 = belum pernah

### Timestamp sangat penting
Timestamp dipakai untuk:
1. **Mengurutkan riwayat** setiap user (film mana yang ditonton duluan).
2. **Membagi data train/test berdasarkan waktu** (lihat di bawah).

### Kode membaca data (`src/coldwarm.py`)
```python
df = pd.read_csv("ratings.dat", sep="::", engine="python",
                 names=["user_id", "item_id", "rating", "timestamp"])
df = df[["user_id", "item_id", "timestamp"]]      # rating dibuang
```
- `sep="::"` karena pemisahnya dua karakter. Untuk itu `engine="python"` wajib.
- `movies.dat` dibaca dengan `encoding="latin-1"` karena ada judul berkarakter non-ASCII.

### Encoding ID
MovieID di file tidak berurutan (ada 1, 2, …, 3952, tapi hanya 3.706 yang terpakai). Fungsi `cw.encode()` mengubahnya menjadi indeks rapat 0, 1, 2, … supaya bisa dipakai sebagai nomor baris/kolom matriks. Hal yang sama dilakukan untuk UserID.

### Matriks interaksi X
Dari tabel tadi dibuat matriks **X** berukuran (jumlah user × jumlah film):
```
         film0  film1  film2 ...
user0      1      0      1
user1      0      1      1
...
```
Karena 95.5% isinya 0, matriks disimpan dalam format **sparse** (hanya angka 1 yang disimpan) supaya hemat memori.

### Split global timepoint (cara membagi train/test)
Data **tidak** dibagi acak, tapi berdasarkan **waktu**. Tujuannya meniru dunia nyata: model hanya boleh belajar dari masa lalu untuk menebak masa depan.

```
     masa lalu                         GT                masa depan
|──────────────────────────────────────|─────────────────────|
  90% interaksi → train & validasi       10% interaksi → test
```
1. **GT (global timepoint)** = titik waktu ketika 90% interaksi sudah terjadi. Di ML-1M, GT ≈ **29 Desember 2000**.
2. User yang punya interaksi **setelah GT** menjadi **user test**:
   - interaksinya sebelum GT = **riwayat** (input ke model)
   - interaksi **pertama** setelah GT = **ground truth** (jawaban yang harus ditebak)
3. User lain dibagi acak: 90% untuk **train**, 10% untuk **validasi** (memilih hyperparameter).

Kenapa tidak memakai "ambil interaksi terakhir tiap user" (leave-one-out) yang umum dipakai? Karena cara itu bisa **membocorkan masa depan** (data leakage). Model bisa belajar dari interaksi user lain yang terjadi **setelah** waktu yang sedang ditebak.

---

## 5. Hubungan dengan `notebooks/01_toy_example.ipynb`

Notebook 01 adalah **versi mini** dari semua langkah di atas: 6 user × 6 film. Dengan data sekecil ini, setiap angka bisa dicek dengan kalkulator. Setiap bagian notebook berpasangan dengan konsep di dokumen ini dan dengan paper:

| Bagian notebook | Apa yang terjadi | Konsep di dokumen ini | Paper |
|---|---|---|---|
| **1. Input** | 22 interaksi (user, film A–F, waktu) | §4 cara membaca data, matriks X | Sec 4.1.1 |
| **2. Preprocessing** | GT = 28.9 → U5, U6 jadi user test; ground truth U5 = C, U6 = D | §4 split global timepoint | Sec 4.1.2 |
| **3. ItemKNN** | co-occurrence G = XᵀX → cosine similarity, mis. sim(A,B) = 4/(√4·√5) = 0.894 | §2 kenapa ItemKNN | Sec 4.1.4 |
| **4. Output rekomendasi** | U5 [A,B,D] → skor C = 1.685, F = 1.078, E = 0.724 → rekomendasi **[C, F]** ✅ | output yang diminta tugas | – |
| **5. User-Based** | riwayat dipotong N = 1, 2, 3; film di riwayat **lengkap** dibuang dari kandidat → HR@2 = 0.5 → 1 → 1, NDCG@2 = 0.315 → 0.815 → 1 | §1 eksperimen User-Based | Sec 3.2 |
| **6. Deteksi threshold** | slope window [1,2] = 0.5 (tercuram), window [2,3] = 0.185 → **threshold = 2** | §1 sliding window | Sec 4.2 |
| **7. Item-Based** | interaksi item B dibatasi N = 1, 2, 3 + latih ulang → HR\*@2 = 0 → 0.333 → 1, NDCG\*@2 = 0 → 0.333 → 0.877 | §1 eksperimen Item-Based | Sec 3.1, Eq. 1–2 |

### Penjelasan singkat tiap langkah notebook

**Langkah 3: ItemKNN.** Idenya: *"film yang sering ditonton orang yang sama itu mirip"*.
- `G[A][B] = 4` artinya ada 4 user yang menonton A **dan** B (U1, U2, U5, U6).
- Angka itu dibagi dengan popularitas masing-masing film (cosine), supaya film yang sangat populer tidak otomatis "mirip dengan semua film".

**Langkah 4: Rekomendasi.** Skor setiap film = jumlah kemiripannya dengan semua film di riwayat user. Film yang sudah ditonton dibuang, lalu ambil K film dengan skor tertinggi.

Di langkah 5 (User-Based), yang dibuang adalah film di riwayat **lengkap** user, bukan hanya N film yang diberikan ke model. Alasannya: user memang sudah menonton film itu, jadi menebaknya tidak berguna. Ini sama dengan notebook 02 dan merupakan **asumsi** kita (lihat §7).

**Langkah 5: Metrik.** Ada dua metrik:
- **HR@K (Hit Rate):** apakah jawaban benar ada di top-K? (1 = ya, 0 = tidak)
- **NDCG@K:** sama seperti HR, tapi memberi nilai lebih tinggi kalau jawabannya ada di urutan atas. Rumusnya 1 / log₂(peringkat + 1):
  - peringkat 1 → 1.0
  - peringkat 2 → 0.631
  - peringkat 3 → 0.5
  - tidak masuk top-K → 0

Di notebook (seed 0):
- U5 dengan 1 film saja ([D]) mendapat [F, E] → **gagal**. Dengan 2 film ([A, B]), film C langsung muncul di peringkat 1.
- U6 dengan 1 film ([F]) mendapat [E, D] → D di peringkat 2 (NDCG 0.631). Dengan 3 film ([A, F, B]), D naik ke peringkat 1.

Rata-ratanya NDCG@2 = 0.315 → 0.815 → 1.0. Lonjakan terbesar terjadi dari N = 1 ke N = 2: itulah **transisi cold → warm**.

**Langkah 6: Sliding window.**
1. Ambil beberapa titik berurutan di grafik (window).
2. Tarik garis lurus terbaik di window itu (regresi linear).
3. Lihat kemiringannya (slope).
4. Window dengan slope **paling curam** menandai lonjakan performa, dan **N terakhir di window itu** adalah threshold.

**Langkah 7: Item-Based.** Fokusnya pada satu film (B). Film B "disembunyikan" sebagian dari data latih: hanya 1, lalu 2, lalu 3 penonton yang tersisa. Setiap kali, model dilatih ulang. Lalu dicek seberapa sering B muncul di rekomendasi user test. Karena itu metriknya diberi tanda \*: yang diukur adalah "apakah **item B** muncul", bukan "apakah jawaban benar muncul".

Satu istilah lagi, **successive evaluation**: holdout user ditambahkan ke riwayat satu per satu. Satu user bisa memberi beberapa titik evaluasi. Contohnya U5 dievaluasi saat riwayatnya [A, D] **dan** [A, D, C].

### Dari notebook 01 ke notebook 02
Notebook 02 menjalankan **langkah yang persis sama** dengan fungsi yang sama dari `src/coldwarm.py`, tapi:
- datanya ML-1M asli (1 juta interaksi), bukan 22 interaksi
- K = 10, bukan 2
- N = 1, 2, …, 64, bukan 1–3
- ada tuning hyperparameter di set validasi
- hasil threshold dibandingkan dengan Table 2 paper

---

## 6. Kamus istilah

| Istilah | Arti sederhana |
|---|---|
| Cold / warm | belum / sudah cukup dikenal sistem |
| Threshold | batas jumlah interaksi antara cold dan warm |
| Interaksi | user pernah menonton/merating sebuah film |
| Implicit feedback | yang dilihat hanya "pernah berinteraksi atau tidak", rating diabaikan |
| Top-K | K rekomendasi teratas |
| Ground truth | jawaban sebenarnya (film yang benar-benar ditonton berikutnya) |
| GT (global timepoint) | batas waktu pemisah train dan test |
| Holdout | interaksi user test setelah GT (disembunyikan dari model) |
| HR@K | apakah jawaban benar ada di top-K |
| NDCG@K | seperti HR, tapi memperhitungkan urutan |
| HR\*@K, NDCG\*@K | versi Item-Based: apakah **item yang diteliti** ada di top-K |
| Successive evaluation | holdout dimasukkan satu per satu, dievaluasi setiap langkah |
| Sliding window regression | mencari bagian grafik yang naik paling curam |
| Data leakage | model tanpa sengaja melihat data "masa depan" |
| Sparse matrix | matriks yang kebanyakan isinya 0, hanya nilai non-0 yang disimpan |

---

## 7. Apa yang diketahui dan tidak diketahui dari paper

### Metodenya diketahui, yang hilang sebagian detail implementasi
Paper menjelaskan **kerangka metodenya** dengan jelas, jadi metode bisa kita jalankan dengan setia. Yang tidak ada adalah sebagian **detail implementasi**. Paper menulis bahwa kodenya ada di `github.com/dalibra/recommendation-dish-better-served-warm`, tapi saat dicek (November 2026) link itu **404 (tidak ditemukan)** dan pencarian di GitHub juga tidak menemukannya. Kemungkinan repo-nya privat, dihapus, atau berganti nama. Akibatnya detail yang tidak tertulis di paper tidak bisa kita cek.

### Yang dijelaskan jelas di paper (dan kita ikuti)
- Split global timepoint di kuantil 0.9. Train dan validasi dibagi acak per user.
- **User-Based:** riwayat di-sample N secara uniform, diurutkan waktu, model tidak dilatih ulang, ground truth = interaksi pertama setelah GT.
- **Item-Based:** sisakan N interaksi acak, latih ulang, user test dikeluarkan dari train, item yang diteliti dibuang dari test & validasi, lalu successive evaluation. S = 1000 item.
- Rumus metrik HR, NDCG, HR\*, NDCG\* (Eq. 1–4), K = 1, 5, 10, 50, 100.
- Deteksi threshold dengan sliding window regresi linear, memilih slope positif tercuram.
- Dataset (ML-1M, dll.) dan model (EASE^R, PureSVD, ItemKNN, SASRec-CE).

### Yang TIDAK disebutkan di paper (dan cara kita mengisinya)
| Detail | Pengaruhnya | Pilihan kita |
|---|---|---|
| **Varian ItemKNN**: rumus similarity, jumlah tetangga, pembobotan | **Paling besar.** ItemKNN kita (cosine, skor = jumlah similarity) sangat bias ke film populer, sehingga film dengan sedikit interaksi hampir tidak pernah direkomendasikan. ItemKNN di paper jelas berperilaku lain. | Cosine + shrinkage, jumlah tetangga dipilih lewat validasi. Untuk Item-Based dipakai tanpa pruning. |
| **Filter seen**: film yang sudah ditonton dibuang berdasarkan riwayat yang **dipotong** atau riwayat **lengkap**? | Besar. Di ML-1M, HR@10 ItemKNN kita ±0.025 kalau hanya membuang riwayat terpotong, dan ±0.055 kalau membuang riwayat lengkap. Nilai paper ±0.05–0.06. | Riwayat **lengkap**, karena hasilnya mendekati paper dan sistem nyata memang tahu apa yang sudah ditonton user. |
| **Lebar sliding window** dan titik mana di window yang dijadikan threshold | Threshold bisa bergeser beberapa angka. | 3 titik (2 di notebook 01), threshold = N terakhir di window. |
| Grid hyperparameter, nilai N persisnya, cara menangani skor seri | Kecil. | Grid kecil penuh. N = 1, 2, …, 8, 10, 12, 16, 20, 24, 32, 48, 64 (mengikuti sumbu grafik paper). Skor seri dihitung pesimis. |

### Artinya untuk proyek kita
- **Kerangka metode dijalankan setia**, jadi tugas "implementasikan proposed method" tetap terpenuhi.
- **Angka di Table 2 belum tentu bisa didapat persis**, karena sebagian hasil bergantung pada detail yang tidak dipublikasikan. Ini wajar di riset. Malah selaras dengan pesan paper: threshold bergantung pada dataset **dan model**. Di sini terlihat bahwa threshold bahkan bergantung pada **implementasi** model yang sama-sama bernama "ItemKNN".
- Saat presentasi, sampaikan dengan jujur. Contoh kalimat:
  > *"Kerangka metode kami mengikuti paper. Beberapa detail implementasi tidak tersedia karena repo kode penulis tidak bisa diakses, jadi kami memakai asumsi X, Y, Z. Hasil User-Based mendekati paper, sedangkan Item-Based berbeda karena …"*

  Penjelasan seperti ini justru menunjukkan kita memahami metodenya, bukan sekadar menyalin angka.

### Penyederhanaan lain
- **Notebook 01** hanya ilustrasi perhitungan dan tidak memakai set validasi. Hasil yang dibandingkan dengan paper ada di notebook 02.
- **Notebook 02** hanya memakai ML-1M dan dua model (ItemKNN, EASE^R), bukan 4 dataset × 4 model.
- **Tuning** memakai grid kecil penuh, bukan 20 titik acak seperti paper.
- **"Latih ulang" di Item-Based** dihitung dengan cara cepat: untuk ItemKNN tanpa pruning, cukup kolom similarity item yang diteliti yang dihitung ulang. Kesetaraannya dengan latih ulang penuh sudah diuji (HR\* identik, selisih NDCG\* ≤ 0.0005). Dengan cara ini setup Item-Based bisa memakai skala paper: 1000 item, semua user test, semua langkah.

---

## 8. Hasil eksperimen ML-1M (notebook 02)

### Setup yang dijalankan
- Data: ML-1M lengkap. Statistiknya sama persis dengan Table 1 paper: 6.040 user, 3.706 film, 1.000.209 interaksi.
- Split: GT = **29 Desember 2000**. Hasilnya 836.101 interaksi train, 483 user validasi, **1.180 user test**.
- Hyperparameter hasil tuning di set validasi: ItemKNN `k = 400, shrink = 50`, EASE^R `λ = 200`.
- Contoh output: seorang user test yang terakhir menonton *Wayne's World 2*, *You've Got Mail*, *Austin Powers*, dll. mendapat rekomendasi *Groundhog Day*, *Forrest Gump*, *Star Wars: Episode V*, … Ground truth-nya *The Englishman Who Went Up a Hill…*, yang tidak masuk top-10. Rekomendasinya didominasi film populer, sebuah tanda awal bias popularitas.

### 8.1 User-Based: cocok dengan paper ✅
| Model | Threshold kita (HR / NDCG) | Paper (Table 2) |
|---|---|---|
| ItemKNN | **4 / 4** | 4 ± 1.4 |
| EASE^R | **4 / 5** | 5 ± 1.1 |

Nilai HR@10 ItemKNN untuk beberapa N (rata-rata 5× sampling, ± = 95% CI):

| N | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| HR@10 | 0.032 ± .005 | 0.040 ± .006 | **0.049** ± .008 | 0.050 ± .009 | 0.059 ± .011 | 0.057 ± .012 | 0.056 ± .013 |
| NDCG@10 | 0.017 | 0.020 | **0.024** | 0.025 | 0.028 | 0.029 | 0.029 |

**Cara membaca:**
- Dari N = 1 ke N = 4, HR naik ±50% (0.032 → 0.049). Ini fase **cold**: setiap interaksi tambahan sangat membantu.
- Setelah N ≈ 4, kurva hampir **datar** (0.05–0.06). User sudah **warm**, dan interaksi tambahan hanya sedikit menambah akurasi.
- Bentuk ini sama dengan Figure 4 paper (ItemKNN ML-1M: ±0.049 di N = 1, datar di ±0.06).
- Grafik: `results/user_based_ml1m.png`.

**Catatan jujur:**
- Confidence interval antar N saling tumpang tindih, karena user test hanya 1.180. Jadi angka "4" sebaiknya dibaca sebagai "sekitar 4", sama seperti paper yang menulis ± 1.4.
- Hasil ini bergantung pada pilihan filter seen dengan riwayat **lengkap** (§7). Tanpa pilihan itu, kurvanya datar di ±0.025 dan threshold-nya tidak bermakna.

### 8.2 Item-Based: threshold tidak ditemukan ❌
- Skalanya sama dengan paper: **1000 item**, **95.812 titik evaluasi** (successive), selesai dalam 25 menit.
- Hasilnya, HR\*@K dan NDCG\*@K = **0 untuk semua N ≤ 64 dan semua K (1–100)**. Di N = 64, **0 dari 509 item** pernah masuk top-100 satu kali pun.
- Akibatnya kurvanya datar di 0 dan sliding window tidak menemukan threshold. Paper mendapat **9 ± 1.8**.

**Ini bukan bug:**
- Perhitungan cepatnya sudah diuji sama dengan latih ulang sungguhan.
- Item yang memakai seluruh interaksinya memang direkomendasikan. Contoh: film terpopuler (2.243 interaksi) masuk top-100 di 99.5% titik evaluasi, dan film ke-300 terpopuler (475 interaksi) di 25%.

**Penyebabnya: ItemKNN kita sangat bias ke item populer.**
- Skor sebuah film = **jumlah** similarity-nya dengan ±150–200 film di riwayat user.
- Film dengan sedikit penonton hanya mirip dengan sedikit film, jadi jumlah skornya selalu kalah dari film populer yang mirip dengan hampir semua film.
- Uji tambahan: film dengan 177 interaksi penuh pun tidak pernah masuk top-100. Film ke-300 terpopuler baru mulai muncul (HR\*@100 = 0.001) setelah punya **256** interaksi.
- Artinya, bagi ItemKNN kita, transisi cold → warm untuk item berada di **ratusan** interaksi, bukan sekitar 9.
- EASE^R yang diuji sekilas berperilaku sama: 0 untuk N ≤ 32.

**Lanjutan: grid N diperluas sampai 512.** Untuk melihat di mana item mulai warm, ditambahkan N = 96, 128, 192, 256, 384, 512 (run yang sama, 1000 item, 24.6 menit):

| N | Item yang dirata-rata | HR\*@100 | NDCG\*@100 | HR\*@10 | Item yang pernah masuk top-100 |
|---|---|---|---|---|---|
| ≤ 64 | 1000 → 509 | 0 | 0 | 0 | 0 |
| 96 | 408 | 0.00000005 | 0.00000001 | 0 | 2 |
| 128 | 337 | 0.00002 | 0.000003 | 0 | 12 |
| 192 | 232 | 0.0016 | 0.0003 | 0 | 54 |
| 256 | 179 | 0.0077 | 0.0013 | 0.00002 | 93 |
| 384 | 111 | 0.046 | 0.0079 | 0.0002 | 93 |
| 512 | 64 | 0.170 | 0.031 | 0.0052 | 61 dari 64 |

Cara membaca:
- Item baru **mulai** muncul di rekomendasi sekitar N ≈ 100–200 interaksi.
- Sliding window memilih **N = 512** (untuk NDCG\*@10, @50, @100), karena slope tercuram ada di window terakhir.
- Tetapi 512 adalah **ujung grid** dan kurvanya **belum datar**, jadi angka 512 bukan threshold yang sah. Yang bisa disimpulkan hanya: threshold item untuk ItemKNN kita **di atas ±400 interaksi**, jauh dari 9 ± 1.8 di paper.
- Kalau grid dibatasi seperti paper (N ≤ 64), threshold tetap **tidak ditemukan**.
- Hati-hati membaca N besar: Eq. 3–4 hanya merata-rata item yang punya ≥ N interaksi. Di N = 512 tinggal 64 item, semuanya film populer. Sebagian kenaikan kurva berasal dari perubahan kelompok item ini, bukan hanya dari bertambahnya N.
- Grafik: `results/item_based_ml1m.png` (sumbu N skala log2, garis abu-abu = batas grid paper di N = 64).

**Kenapa berbeda dengan paper?** Kemungkinan besar varian ItemKNN di paper berbeda, misalnya ada normalisasi atau pembobotan yang mengurangi dominasi item populer. Detail ini tidak tertulis dan kodenya tidak bisa diakses (§7).

**Pelajaran untuk presentasi:** hasil ini justru **mendukung** pesan utama paper, yaitu bahwa threshold tidak bisa diambil dari angka umum dan harus diukur ulang untuk setiap dataset **dan model**. Model yang sama-sama bernama "ItemKNN" bisa punya threshold item yang sangat berbeda (±9 vs ratusan).

### 8.3 Status & rencana lanjutan
| Hal | Status |
|---|---|
| User-Based ItemKNN & EASE^R | ✅ selesai, cocok dengan paper |
| Item-Based ItemKNN, N ≤ 64 | ✅ selesai, threshold tidak ditemukan (penjelasan di atas) |
| Item-Based dengan N diperluas (96–512) untuk mencari di mana ItemKNN kita mulai warm | ✅ selesai. Item mulai muncul di N ≈ 100–200; kurva belum datar di N = 512, jadi threshold > ±400 (tabel di atas) |
| Menyamakan filter seen di notebook 01 dengan notebook 02 (riwayat lengkap) | ✅ selesai. Riwayat U6 diganti jadi [A, F, B] → ground truth D supaya kurva toy tetap naik (NDCG@2 0.315 → 0.815 → 1). Angka di §5 sudah diperbarui; **slide contoh hitung perlu disesuaikan** |
