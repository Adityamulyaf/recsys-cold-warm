# Script Presentasi: Cold-to-Warm Threshold

Script ini sama dengan catatan pembicara (speaker notes) di slide. Perkiraan durasi bicara sekitar 20 sampai 25 menit (±2.150 kata), ditambah sesi tanya jawab. Slide 20 juga berisi persiapan jawaban untuk pertanyaan yang mungkin muncul.

## Slide 1: Kapan user dan item dianggap “warm”?

Selamat pagi atau siang semuanya. Kami dari kelompok [nama kelompok].

Hari ini kami akan membahas paper berjudul “Recommendation Is a Dish Better Served Warm”, dari RecSys 2025.

Pertanyaan utama paper ini sederhana: berapa interaksi yang dibutuhkan sebuah user atau sebuah item, supaya sistem rekomendasi sudah “kenal” dengan mereka?

Presentasi kami ada tiga bagian. Pertama, isi paper. Kedua, contoh hitung manual dengan data kecil. Ketiga, hasil eksperimen kami di dataset MovieLens-1M, lalu kami bandingkan dengan paper.

## Slide 2: Banyak paper membuang user dan item yang interaksinya sedikit. Angka batasnya hanya ditebak.

Sebelum melatih model rekomendasi, peneliti biasanya membuang user atau item yang interaksinya sedikit. Misalnya: buang user yang punya kurang dari 5 interaksi. Angka 5 ini disebut threshold.

Penulis paper memeriksa paper di konferensi RecSys dari 2022 sampai 2024. Ada 65 paper yang melakukan filtering seperti ini. Dan tidak ada satu pun yang menjelaskan kenapa memilih angka itu.

Kenapa ini masalah? Kalau angkanya terlalu tinggi, data yang berguna ikut terbuang. Kalau terlalu rendah, user atau item yang belum cukup dikenal ikut dipakai dan menjadi noise. Akibatnya, hasil antar paper juga sulit dibandingkan dengan adil.

## Slide 3: Berapa interaksi yang dibutuhkan supaya sistem “kenal”?

Bayangkan pelayan restoran. Pelanggan yang baru pesan sekali, kita belum tahu seleranya. Itu cold. Pelanggan yang sudah pesan dua puluh kali, kita sudah hafal. Itu warm. Pertanyaannya: mulai pesanan ke berapa pelanggan bisa dianggap sudah dikenal?

Paper ini menjawabnya dengan dua eksperimen.

User-Based: kita ubah panjang riwayat user, dari 1 interaksi, 2, 3, dan seterusnya. Modelnya tetap, tidak dilatih ulang. Lalu kita lihat kapan rekomendasinya mulai akurat.

Item-Based: kita ubah jumlah interaksi satu item di data latih. Karena data latihnya berubah, model harus dilatih ulang untuk setiap N. Lalu kita lihat kapan item itu mulai sering muncul di rekomendasi.

Penting: paper ini tidak membuat model rekomendasi baru. Yang dibuat adalah cara mengukur threshold.

## Slide 4: Lima langkah dari data sampai threshold

Ini alur metode paper dalam lima langkah.

Satu, data dibagi berdasarkan waktu. Kita cari titik waktu GT, yaitu kuantil 0.9 dari semua timestamp. User yang punya interaksi setelah GT menjadi user test. Cara ini mencegah model melihat masa depan.

Dua, latih model rekomendasi biasa dari data sebelum GT.

Tiga, potong datanya menjadi N interaksi. Untuk User-Based yang dipotong riwayat user. Untuk Item-Based yang dipotong interaksi item di data latih.

Empat, ukur kualitas rekomendasi untuk setiap N. Hasilnya sebuah grafik: sumbu X adalah N, sumbu Y adalah metrik.

Lima, cari titik di grafik yang naik paling tajam. Caranya dengan sliding window: ambil beberapa titik berurutan, tarik garis lurus, lalu lihat kemiringannya. Window yang paling curam menandai perpindahan dari cold ke warm.

## Slide 5: Threshold berbeda untuk setiap dataset dan model

Hasil paper. Di empat dataset dan empat model, threshold item berada di sekitar 6 sampai 15 interaksi. Threshold user di sekitar 4 sampai 12 interaksi.

Angkanya berbeda-beda tergantung dataset dan model. Paper juga menemukan bahwa model sekuensial seperti SASRec butuh riwayat user yang lebih panjang.

Tabel di kanan adalah angka MovieLens-1M yang nanti kami bandingkan dengan hasil kami. ItemKNN: threshold user 4, threshold item 9. EASE: threshold user 5.

Jadi pesan utamanya: jangan pakai angka tebakan seperti 5 atau 20. Ukur dulu untuk setiap dataset dan model.

## Slide 6: Yang kami pakai dan alasannya

Sekarang bagian implementasi kami.

Metode yang kami ambil adalah framework dari paper, karena tugas meminta proposed method. Tapi framework ini tidak menghasilkan rekomendasi sendiri. Jadi kami butuh model dasar. Kami pakai ItemKNN, dan EASE sebagai pembanding. Keduanya dipakai di paper, dan ItemKNN bisa dihitung dengan tangan.

Datasetnya MovieLens-1M. Ini dataset paling padat di paper, dan ada angka pembanding di Table 2.

Split dan metrik kami samakan dengan paper.

Supaya setiap langkah bisa dicek dengan kalkulator, kami mulai dengan data mainan: 6 user dan 6 film. Semua angka di slide berikutnya berasal dari notebook 01.

## Slide 7: Data mainan: 22 interaksi, film A sampai F

Ini input kami. Ada 6 user, U1 sampai U6, dan 6 film, A sampai F. Setiap pasangan berarti user menonton film itu pada waktu tertentu. Contohnya U1 menonton A di waktu 1, B di waktu 2, dan seterusnya.

Kami tidak memakai rating. Yang dipakai hanya pernah menonton atau tidak. Ini disebut implicit feedback, sama seperti di paper.

Dataset MovieLens-1M bentuknya sama. Satu baris berisi user, film, rating, dan timestamp. Contoh di kanan: user 1 menonton film 1193 pada timestamp itu. Ratingnya kami abaikan.

Perhatikan U5 dan U6. Mereka punya interaksi di waktu 30-an. Itu nanti menjadi data test.

## Slide 8: Split global timepoint: GT = 28.9

Langkah preprocessing. Kami ambil kuantil 0.9 dari semua timestamp. Hasilnya GT sama dengan 28.9.

User yang punya interaksi setelah GT menjadi user test. Di sini ada dua: U5 dan U6.

Riwayat U5 adalah A, B, D. Interaksi pertamanya setelah GT adalah C. Jadi C adalah jawaban yang harus ditebak model.

Riwayat U6 adalah A, F, B. Jawabannya D.

Semua interaksi sebelum GT, termasuk riwayat U5 dan U6, dipakai untuk melatih model. Interaksi setelah GT tidak boleh dilihat model.

## Slide 9: Film yang ditonton orang yang sama dianggap mirip

Model dasar kami adalah ItemKNN. Idenya sederhana: film yang sering ditonton oleh orang yang sama dianggap mirip.

Langkah pertama, kami hitung matriks G. Isinya jumlah user yang menonton dua film sekaligus. Contohnya G baris A kolom B sama dengan 4, artinya ada 4 user yang menonton A dan B: U1, U2, U5, dan U6. Angka di diagonal adalah popularitas film. A ditonton 4 user, B ditonton 5 user.

Langkah kedua, kami ubah menjadi cosine similarity. Angka G dibagi akar popularitas kedua film. Tujuannya supaya film yang sangat populer tidak otomatis dianggap mirip dengan semua film.

Contoh: similarity A dan B sama dengan 4 dibagi akar 4 kali akar 5. Hasilnya 0.894.

## Slide 10: Skor film = jumlah kemiripan dengan riwayat

Ini output yang diminta tugas, yaitu rekomendasi.

Skor sebuah film adalah jumlah similarity film itu dengan setiap film di riwayat user. Film yang sudah pernah ditonton dibuang dari kandidat.

Untuk U5, riwayatnya A, B, D. Skor C adalah similarity C dengan A, ditambah dengan B, ditambah dengan D. Hasilnya 1.685, paling tinggi. Rekomendasi top-2 untuk U5 adalah C dan F. Jawaban yang benar adalah C, dan C ada di peringkat 1.

Untuk U6, riwayatnya A, F, B. Skor tertinggi adalah D. Rekomendasinya D dan C. Jawabannya D, juga di peringkat 1.

Ini terjadi kalau model melihat seluruh riwayat. Pertanyaannya: bagaimana kalau riwayatnya lebih pendek?

## Slide 11: Riwayat dipotong menjadi N film, model tetap

Ini eksperimen User-Based. Modelnya tetap, tidak dilatih ulang. Yang kami ubah adalah panjang riwayat yang diberikan ke model.

Untuk setiap N, kami ambil N film secara acak dari riwayat user, lalu diurutkan sesuai waktu. Film yang ada di riwayat lengkap user tetap dibuang dari kandidat, karena user memang sudah menontonnya.

Metriknya NDCG. Kalau jawaban di peringkat 1, nilainya 1. Kalau di peringkat 2, nilainya 1 dibagi log basis 2 dari 3, yaitu 0.631. Kalau tidak masuk top-K, nilainya 0.

N sama dengan 1: U5 hanya diberi film D. Rekomendasinya F dan E. C tidak muncul, jadi nilainya 0. U6 diberi F, D muncul di peringkat 2. Rata-ratanya 0.315.

N sama dengan 2: U5 diberi A dan B, C langsung di peringkat 1. Rata-ratanya naik ke 0.815.

N sama dengan 3: keduanya tepat di peringkat 1. Rata-ratanya 1.

Jadi semakin panjang riwayat, rekomendasinya semakin akurat. Lonjakan terbesar terjadi dari N 1 ke N 2.

## Slide 12: Sliding window: cari kemiringan paling curam

Setelah punya grafik, kita cari threshold-nya.

Caranya sliding window. Ambil beberapa titik berurutan, di sini dua titik. Tarik garis lurus dengan least squares, lalu hitung kemiringannya.

Dari N 1 ke N 2, nilainya naik dari 0.315 ke 0.815. Kemiringannya 0.5. Dari N 2 ke N 3, naiknya hanya 0.185.

Window yang paling curam adalah N 1 sampai 2. Kami ambil N terakhir di window itu, jadi threshold-nya 2. Artinya, di data mainan ini user sudah warm setelah punya 2 interaksi.

Satu catatan jujur: paper tidak menyebut berapa lebar window, dan titik mana yang dijadikan threshold. Jadi ini asumsi kami. Di data mainan kami pakai lebar 2 karena hanya ada 3 titik. Di ML-1M kami pakai lebar 3.

## Slide 13: Interaksi film B dibatasi menjadi N, lalu model dilatih ulang

Eksperimen kedua adalah Item-Based. Sekarang yang diteliti adalah satu film, yaitu B.

Di data latih, B ditonton oleh U1, U2, dan U3. Interaksi user test tidak dipakai untuk melatih.

Untuk setiap N, kami sisakan N penonton B secara acak, buang sisanya, lalu latih ulang model.

Evaluasinya memakai successive evaluation. Data holdout user test ditambahkan satu per satu, jadi satu user bisa memberi beberapa titik evaluasi. U5 dievaluasi saat riwayatnya A, D, dan saat riwayatnya A, D, C. U6 dievaluasi saat riwayatnya A, F. Totalnya tiga titik.

Metriknya diberi tanda bintang karena yang diukur berbeda. Bukan apakah jawaban benar muncul, tapi apakah film B muncul.

Hasilnya: dengan 1 penonton, B tidak pernah muncul. Dengan 2 penonton, B muncul di 1 dari 3 titik. Dengan 3 penonton, B muncul di semua titik. Jadi film juga butuh cukup interaksi supaya mulai direkomendasikan.

## Slide 14: Eksperimen MovieLens-1M

Itu tadi contoh kecilnya. Sekarang kami jalankan langkah yang persis sama, dengan fungsi yang sama dari kode kami, tapi di data asli MovieLens-1M yang berisi satu juta interaksi.

Bedanya: K sekarang 10, N dari 1 sampai 64, dan ada tuning hyperparameter di data validasi.

## Slide 15: ML-1M setelah split, dan contoh rekomendasinya

Statistik datanya sama persis dengan Table 1 di paper: 6,040 user, 3,706 film, dan sekitar satu juta interaksi.

Titik GT jatuh di 29 Desember 2000. Ada 836 ribu interaksi untuk training, dan 1,180 user test.

Hyperparameter ItemKNN kami pilih dengan tuning di data validasi. Hasil terbaiknya k sama dengan 400 tetangga dan shrink 50. Untuk EASE, lambda sama dengan 200.

Di kanan adalah contoh output rekomendasi untuk satu user test. Riwayat user ini banyak film komedi seperti Wayne's World 2 dan Austin Powers. Model merekomendasikan film seperti Groundhog Day dan Forrest Gump.

Perhatikan bahwa yang muncul kebanyakan film yang sangat populer. Ini akan penting nanti.

## Slide 16: Naik sampai N = 4, lalu datar. Sama dengan paper.

Ini hasil User-Based di ML-1M. Sumbu X adalah panjang riwayat N, dari 1 sampai 64. Sumbu Y adalah Hit Rate di top-10.

Dari N 1 ke N 4, Hit Rate naik sekitar 50 persen, dari 0.032 ke 0.049. Ini fase cold: setiap interaksi tambahan sangat membantu. Setelah N 4, grafiknya hampir datar di sekitar 0.05 sampai 0.06. User sudah warm.

Sliding window menemukan threshold di N sama dengan 4 untuk ItemKNN. Paper mendapat 4 plus minus 1.4. Untuk EASE kami dapat 4 dari HR dan 5 dari NDCG, paper mendapat 5. Jadi hasil User-Based kami cocok dengan paper.

Dua catatan jujur. Pertama, rentang kepercayaan antar N saling tumpang tindih karena user test hanya 1,180. Jadi angka 4 sebaiknya dibaca sebagai sekitar 4. Kedua, hasil ini memakai asumsi bahwa film di riwayat lengkap user dibuang dari kandidat. Paper tidak menjelaskan bagian ini.

## Slide 17: Sampai N = 64, tidak ada film yang direkomendasikan

Sekarang hasil Item-Based. Di sini hasil kami berbeda dengan paper.

Kami memakai 1000 film dan hampir 96 ribu titik evaluasi, sama dengan skala paper. Hasilnya: sampai N sama dengan 64, tidak ada satu film pun yang masuk top-100, di titik evaluasi mana pun. Semua metrik bernilai nol. Jadi threshold tidak ditemukan. Paper mendapat 9.

Karena itu kami perluas grid N sampai 512. Film baru mulai muncul di sekitar 100 sampai 200 interaksi. Di N 512, 61 dari 64 film sudah pernah masuk top-100. Tapi grafiknya masih naik tajam, belum datar. Jadi kami tidak bisa menyebut angka threshold yang pasti. Yang bisa kami katakan: untuk ItemKNN kami, threshold item berada di atas sekitar 400 interaksi.

Catatan: di N besar, yang dirata-rata hanya film yang punya minimal N interaksi. Di N 512 tinggal 64 film, semuanya film populer.

## Slide 18: Kenapa hasil Item-Based berbeda?

Kenapa hasil Item-Based kami berbeda jauh dari paper?

Pertama, kami pastikan ini bukan bug. Cara cepat kami menghitung latih ulang sudah kami cek, hasilnya sama dengan latih ulang sungguhan. Lalu film yang punya banyak data memang direkomendasikan. Film terpopuler, dengan 2,243 interaksi, masuk top-100 di 99.5 persen titik evaluasi.

Penyebabnya adalah bias popularitas di ItemKNN kami. Skor sebuah film adalah jumlah similarity dengan semua film di riwayat user. Rata-rata user di ML-1M punya ratusan film di riwayatnya. Film populer mirip dengan hampir semua film, jadi jumlah skornya besar. Film yang jarang ditonton hanya mirip dengan sedikit film, jadi selalu kalah.

Kemungkinan besar ItemKNN di paper memakai normalisasi atau pembobotan yang berbeda. Sayangnya detail ini tidak tertulis di paper, dan link repositori kodenya tidak bisa diakses.

## Slide 19: Yang kami pelajari

Kesimpulan kami ada tiga.

Satu, hasil User-Based kami cocok dengan paper. User di ML-1M sudah warm setelah sekitar 4 sampai 5 interaksi.

Dua, hasil Item-Based kami berbeda. ItemKNN kami butuh ratusan interaksi per film sebelum film itu mulai direkomendasikan, bukan sekitar 9 seperti di paper.

Tiga, justru perbedaan ini mendukung pesan utama paper. Dua model yang sama-sama bernama ItemKNN bisa punya threshold item yang sangat berbeda. Artinya threshold tidak bisa diambil dari angka umum seperti 5 atau 20. Threshold harus diukur untuk setiap dataset dan setiap model.

## Slide 20: Pertanyaan?

Itu presentasi dari kami. Kode, notebook, dan penjelasan lengkapnya ada di repositori GitHub kami. Terima kasih, kami persilakan kalau ada pertanyaan.

Persiapan untuk pertanyaan yang mungkin muncul:

Kenapa tidak pakai leave-one-out? Karena cara itu bisa membocorkan masa depan. Model bisa belajar dari interaksi user lain yang terjadi setelah waktu yang sedang ditebak.

Kenapa hanya ML-1M? Karena dataset lain di paper jauh lebih besar dan lebih jarang. ML-1M juga punya angka pembanding di Table 2.

Apa saja asumsi kalian? Lebar sliding window 3 titik, threshold diambil dari N terakhir di window, dan film di riwayat lengkap user dibuang dari kandidat. Paper tidak menjelaskan ketiganya.

Kenapa setiap user di ML-1M punya minimal 20 rating? Karena dataset ini sudah difilter oleh pembuatnya. Ini contoh nyata threshold yang tidak dijelaskan alasannya, persis masalah yang diangkat paper.
