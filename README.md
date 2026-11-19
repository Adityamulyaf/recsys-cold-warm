# Cold-to-Warm Threshold: Implementasi Proyek Kapita Selekta

Implementasi skala kecil dari proposed method di paper
**"Recommendation Is a Dish Better Served Warm"** (Gusak, Sukhorukov, Frolov, RecSys 2025, [arXiv:2508.07856](https://arxiv.org/abs/2508.07856)).

Metode paper adalah **framework evaluasi** untuk mencari berapa interaksi yang dibutuhkan user atau item agar dianggap *warm*. Supaya framework ini menghasilkan rekomendasi, dipakai model dasar **ItemKNN** (dan **EASE^R**). Keduanya termasuk model yang dipakai di paper.

## Struktur
| File | Isi |
|---|---|
| `src/coldwarm.py` | split global timepoint, ItemKNN, EASE^R, HR/NDCG & HR\*/NDCG\*, analisis User-Based & Item-Based, deteksi threshold dengan sliding window |
| `notebooks/01_toy_example.ipynb` | contoh hitung manual (6 user × 6 item) dari input sampai threshold, untuk slide |
| `notebooks/02_ml1m_experiment.ipynb` | eksperimen MovieLens-1M; hasilnya dibandingkan dengan Table 2 paper |
| `results/` | grafik & CSV hasil notebook 02 |
| `docs/` | paper, slide, dan deskripsi tugas |

## Menjalankan
```bash
pip install -r requirements.txt
# unduh https://files.grouplens.org/datasets/movielens/ml-1m.zip, ekstrak ke data/raw/ml-1m/
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_toy_example.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_ml1m_experiment.ipynb
```
Bagian Item-Based di notebook 02 butuh ±35 menit. Set `RUN_ITEM_BASED = False` kalau hanya butuh User-Based.

## Kesesuaian dengan paper
| Komponen | Paper | Di sini |
|---|---|---|
| Split | Global timepoint kuantil 0.9; train/valid dibagi acak per user | sama |
| Tuning | 20 titik acak dari grid, di set validasi | grid kecil penuh (NDCG@10 validasi) |
| User-Based | model tetap; riwayat di-sample N secara uniform lalu diurutkan waktu; GT = interaksi pertama setelah GT; HR@10/NDCG@10; rata-rata atas user dengan ≥ N interaksi | sama; sampling diulang 5×; film di riwayat **lengkap** dibuang dari kandidat (**asumsi**, paper tidak merinci) |
| Item-Based | N interaksi acak item *i* + latih ulang; user test dikeluarkan dari train; item *i* dibuang dari test/valid; successive evaluation; HR\*/NDCG\* (Eq. 1–4); S = 1000 item | sama (S = 1000), ItemKNN tanpa pruning; latih ulang dihitung eksak lewat pembaruan satu kolom similarity (`ItemBasedItemKNN`) |
| Threshold | sliding window least squares, slope positif tercuram | sama; **asumsi**: lebar window 3 titik, threshold = N terakhir di window itu (paper tidak menyebutkan keduanya) |
| Model | EASE^R, PureSVD, ItemKNN, SASRec-CE | ItemKNN, EASE^R |

Contoh di notebook 01 hanya ilustrasi perhitungan, bukan replikasi.
