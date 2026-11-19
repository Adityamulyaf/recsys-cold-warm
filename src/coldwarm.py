"""Implementasi sederhana dari framework cold-to-warm threshold.

Referensi: Gusak, Sukhorukov, Frolov. "Recommendation Is a Dish Better Served Warm".
RecSys 2025. Nomor bagian di komentar mengacu ke paper tersebut.

Isi modul:
- split data dengan global timepoint (Sec 4.1.2)
- model ItemKNN dan EASE^R (Sec 4.1.4)
- metrik HR@K/NDCG@K standar dan HR*@K/NDCG*@K modifikasi (Sec 4.1.3, Eq. 1-4)
- analisis User-Based (Sec 3.2) dan Item-Based (Sec 3.1)
- deteksi threshold dengan sliding window least squares (Sec 4.2)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Grid N kira-kira mengikuti sumbu-x Figure 3 & 4 di paper.
DEFAULT_NS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 48, 64]

# ASUMSI: paper tidak menyebut lebar window, dan repo resmi tidak bisa diakses.
DEFAULT_WINDOW = 3


# --------------------------------------------------------------------------- data
def load_ml1m(path: str) -> pd.DataFrame:
    """Baca ratings.dat ML-1M; semua rating dianggap interaksi implisit."""
    df = pd.read_csv(path, sep="::", engine="python",
                     names=["user_id", "item_id", "rating", "timestamp"])
    return df[["user_id", "item_id", "timestamp"]]


def encode(df: pd.DataFrame):
    """Ubah id user/item asli menjadi indeks 0..n-1."""
    df = df.copy()
    df["u"], users = pd.factorize(df["user_id"])
    df["i"], items = pd.factorize(df["item_id"])
    return df, len(users), len(items)


@dataclass
class Split:
    train: pd.DataFrame          # interaksi untuk melatih model
    val_hist: dict               # user valid -> riwayat (list item, urut waktu)
    val_gt: dict                 # user valid -> item terakhir (ground truth validasi)
    test_hist: dict              # user test -> riwayat sebelum GT (urut waktu)
    test_holdout: dict           # user test -> semua item setelah GT (urut waktu)
    gt_time: float


def global_timepoint_split(df: pd.DataFrame, q: float = 0.9, val_frac: float = 0.1,
                           include_test_history_in_train: bool = True,
                           seed: int = 0) -> Split:
    """Split global timepoint (Sec 4.1.2, Fig 2b-c).

    - GT = kuantil-q dari timestamp semua interaksi.
    - User yang punya interaksi setelah GT -> user test; interaksi sebelum GT = riwayat input.
    - User lain dibagi acak per user menjadi train/valid. Untuk user valid,
      interaksi terakhir = ground truth dan sisanya = input.
    - include_test_history_in_train: True untuk User-Based. Untuk Item-Based
      paper mengeluarkan interaksi user test dari train, jadi pakai False.
    """
    df = df.sort_values(["timestamp", "u", "i"], kind="mergesort")
    gt = df["timestamp"].quantile(q)
    before, after = df[df["timestamp"] <= gt], df[df["timestamp"] > gt]

    test_users = set(after["u"].unique()) & set(before["u"].unique())
    test_hist = before[before["u"].isin(test_users)].groupby("u")["i"].apply(list).to_dict()
    test_holdout = after[after["u"].isin(test_users)].groupby("u")["i"].apply(list).to_dict()

    rng = np.random.default_rng(seed)
    other = np.array(sorted(set(before["u"].unique()) - test_users))
    val_users = set(rng.choice(other, size=int(len(other) * val_frac), replace=False))

    val_df = before[before["u"].isin(val_users)]
    val_seq = val_df.groupby("u")["i"].apply(list)
    val_seq = val_seq[val_seq.map(len) >= 2]
    val_hist = {u: s[:-1] for u, s in val_seq.items()}
    val_gt = {u: s[-1] for u, s in val_seq.items()}

    train_mask = ~before["u"].isin(val_users)
    if not include_test_history_in_train:
        train_mask &= ~before["u"].isin(test_users)
    return Split(before[train_mask], val_hist, val_gt, test_hist, test_holdout, gt)


def to_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    X = sp.csr_matrix((np.ones(len(df), dtype=np.float32), (df["u"].values, df["i"].values)),
                      shape=(n_users, n_items))
    X.data[:] = 1.0  # duplikat -> biner
    return X


# --------------------------------------------------------------------------- models
class ItemKNN:
    """ItemKNN (Sarwar dkk., 2001) dengan cosine similarity dan pruning top-k tetangga.

    sim(a, b) = |U_a ∩ U_b| / (sqrt(|U_a|) * sqrt(|U_b|) + shrink)
    skor(u, j) = Σ_{h ∈ riwayat(u)} sim(h, j)
    """

    def __init__(self, k: int = 100, shrink: float = 0.0):
        self.k, self.shrink = k, shrink

    def fit(self, X: sp.csr_matrix):
        G = (X.T @ X).toarray().astype(np.float32)          # co-occurrence item-item
        norms = np.sqrt(np.diag(G)).astype(np.float32)
        S = G / (np.outer(norms, norms) + self.shrink + 1e-12)
        np.fill_diagonal(S, 0.0)
        if self.k and self.k < S.shape[0]:                  # simpan k tetangga terdekat per item
            drop = np.argpartition(-S, self.k, axis=0)[self.k:, :]
            np.put_along_axis(S, drop, 0.0, axis=0)
        self.S = S
        return self

    def score(self, H: sp.csr_matrix) -> np.ndarray:
        return np.asarray(H @ self.S)


class EASE:
    """EASE^R (Steck, 2019): B = I - P·diag(1/diag(P)),  P = (XᵀX + λI)^-1."""

    def __init__(self, lam: float = 500.0):
        self.lam = lam

    def fit(self, X: sp.csr_matrix):
        G = (X.T @ X).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.lam
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        np.fill_diagonal(B, 0.0)
        self.S = B.astype(np.float32)
        return self

    def score(self, H: sp.csr_matrix) -> np.ndarray:
        return np.asarray(H @ self.S)


def history_matrix(histories: list[list[int]], n_items: int) -> sp.csr_matrix:
    rows = np.repeat(np.arange(len(histories)), [len(h) for h in histories])
    cols = np.concatenate([np.asarray(h, dtype=np.int64) for h in histories]) if histories else []
    H = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(len(histories), n_items))
    H.data[:] = 1.0
    return H


def recommend(model, histories: list[list[int]], n_items: int, K: int = 10,
              filter_seen: bool = True, exclude: list[list[int]] | None = None,
              batch: int = 2048) -> np.ndarray:
    """Top-K rekomendasi (indeks item, urut skor menurun) untuk tiap riwayat.

    `exclude`: item yang dibuang dari kandidat per user. Defaultnya = riwayat itu
    sendiri. Di User-Based kita isi dengan riwayat LENGKAP user (lihat user_based_curve).
    """
    out = np.empty((len(histories), K), dtype=np.int64)
    for s in range(0, len(histories), batch):
        hb = histories[s:s + batch]
        scores = model.score(history_matrix(hb, n_items))
        if filter_seen:
            eb = hb if exclude is None else exclude[s:s + batch]
            for r, h in enumerate(eb):
                scores[r, h] = -np.inf
        if K >= n_items:
            top = np.tile(np.arange(n_items), (len(hb), 1))[:, :K]
        else:
            top = np.argpartition(-scores, K, axis=1)[:, :K]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
        out[s:s + batch] = np.take_along_axis(top, order, axis=1)
    return out


# --------------------------------------------------------------------------- metrics
def hit_ndcg(recs: np.ndarray, targets) -> tuple[np.ndarray, np.ndarray]:
    """HR@K dan NDCG@K standar, satu target per user.

    HR = 1[target ∈ top-K];  NDCG = 1 / log2(rank + 1), rank dimulai dari 1
    (IDCG = 1 karena hanya ada satu item relevan).
    """
    targets = np.asarray(targets)[:, None]
    match = recs == targets
    hit = match.any(axis=1).astype(float)
    rank = match.argmax(axis=1) + 1
    ndcg = np.where(hit > 0, 1.0 / np.log2(rank + 1), 0.0)
    return hit, ndcg


def modified_hit_ndcg(recs: np.ndarray, item: int) -> tuple[float, float]:
    """HR*@K(i) dan NDCG*@K(i), Eq. (1)-(2): seberapa sering & di posisi mana item i muncul."""
    match = recs == item
    hit = match.any(axis=1)
    rank = match.argmax(axis=1) + 1
    return float(hit.mean()), float(np.where(hit, 1.0 / np.log2(rank + 1), 0.0).mean())


def ci95(x) -> float:
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


# --------------------------------------------------------------------------- tuning
def tune(model_cls, grid: list[dict], X, split: Split, n_items: int, K: int = 10):
    """Pilih hyperparameter berdasarkan NDCG@K di set validasi (Sec 4.1.2)."""
    users = list(split.val_hist)
    hists = [split.val_hist[u] for u in users]
    gts = [split.val_gt[u] for u in users]
    results = []
    for params in grid:
        m = model_cls(**params).fit(X)
        _, ndcg = hit_ndcg(recommend(m, hists, n_items, K), gts)
        results.append((params, float(ndcg.mean())))
    best = max(results, key=lambda r: r[1])
    return best[0], results


# --------------------------------------------------------------------------- user-based
def user_based_curve(model, split: Split, n_items: int, Ns=DEFAULT_NS, K: int = 10,
                     seed: int = 0, n_repeats: int = 1,
                     filter_full_history: bool = False) -> pd.DataFrame:
    """Analisis User-Based (Sec 3.2).

    Model sudah dilatih dan tidak diubah. Untuk tiap N dan tiap user test dengan
    |L(u)| >= N: ambil N interaksi secara uniform dari riwayatnya, urutkan sesuai
    waktu, beri rekomendasi, lalu bandingkan dengan interaksi PERTAMA setelah GT.

    n_repeats: sampling diulang beberapa kali per user, lalu metrik per user
        dirata-rata. Ini mengurangi noise akibat satu kali sampling acak.
    filter_full_history: True = item di riwayat LENGKAP user dibuang dari kandidat
        (bukan hanya N item sampel). Paper tidak merinci ini (Sec 4.2 hanya bilang
        hasil konsisten dengan/tanpa "filter seen"). Tanpa ini, slot top-K terisi
        film yang sudah pernah ditonton dan HR turun jauh di bawah nilai paper.
    """
    rng = np.random.default_rng(seed)
    users = list(split.test_hist)
    rows = []
    for N in Ns:
        sel = [u for u in users if len(split.test_hist[u]) >= N]
        if len(sel) < 2:
            break
        gts = [split.test_holdout[u][0] for u in sel]
        exclude = [split.test_hist[u] for u in sel] if filter_full_history else None
        hit, ndcg = np.zeros(len(sel)), np.zeros(len(sel))
        for _ in range(n_repeats):
            hists = []
            for u in sel:
                L = split.test_hist[u]
                idx = np.sort(rng.choice(len(L), size=N, replace=False))  # urutan waktu tetap terjaga
                hists.append([L[j] for j in idx])
            h, n = hit_ndcg(recommend(model, hists, n_items, K, exclude=exclude), gts)
            hit += h / n_repeats
            ndcg += n / n_repeats
        rows.append(dict(N=N, n_users=len(sel), HR=hit.mean(), HR_ci=ci95(hit),
                         NDCG=ndcg.mean(), NDCG_ci=ci95(ndcg)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- item-based
def item_based_curve(model_factory, split: Split, n_users: int, n_items: int,
                     Ns=DEFAULT_NS, Ks=(1, 5, 10, 50, 100), n_items_sample: int = 50,
                     max_steps: int = 3, max_test_users: int | None = 500,
                     min_item_interactions: int | None = None, seed: int = 0,
                     verbose: bool = True) -> pd.DataFrame:
    """Analisis Item-Based (Sec 3.1) dengan successive evaluation (Fig 2a).

    `split` harus dibuat dengan include_test_history_in_train=False.
    Untuk tiap item i (sampel S item) dan tiap N <= |M(i)|:
      1. ambil N interaksi acak item i di train, buang sisanya -> τ_N
      2. latih ulang model dengan τ_N
      3. item i dibuang dari riwayat & holdout user test (dan valid)
      4. successive evaluation: riwayat + holdout[:t] -> top-K, t = 0..max_steps-1
      5. HR*@K(i), NDCG*@K(i) = rata-rata atas semua titik evaluasi
    Penyederhanaan (komputasi): S, max_steps, dan max_test_users lebih kecil dari paper.
    """
    rng = np.random.default_rng(seed)
    train = split.train
    counts = train["i"].value_counts()
    pool = counts.index.values if min_item_interactions is None else \
        counts[counts >= min_item_interactions].index.values
    items = rng.choice(pool, size=min(n_items_sample, len(pool)), replace=False)

    test_users = list(split.test_hist)
    if max_test_users and len(test_users) > max_test_users:
        test_users = list(rng.choice(test_users, size=max_test_users, replace=False))

    Kmax = max(Ks)
    rows = []
    for c, item in enumerate(items, 1):
        M = np.flatnonzero(train["i"].values == item)
        # Riwayat untuk successive evaluation, tanpa item i
        eval_hists = []
        for u in test_users:
            base = [x for x in split.test_hist[u] if x != item]
            hold = [x for x in split.test_holdout[u] if x != item]
            for t in range(min(max_steps, len(hold))):
                eval_hists.append(base + hold[:t])
        for N in Ns:
            if N > len(M):
                break
            keep = rng.choice(M, size=N, replace=False)
            drop = np.setdiff1d(M, keep)
            tau = train.drop(train.index[drop])
            model = model_factory().fit(to_matrix(tau, n_users, n_items))
            recs = recommend(model, eval_hists, n_items, Kmax)
            row = dict(item=int(item), N=N, n_train=len(M))
            for K in Ks:
                h, n = modified_hit_ndcg(recs[:, :K], item)
                row[f"HR*@{K}"], row[f"NDCG*@{K}"] = h, n
            rows.append(row)
        if verbose:
            print(f"[{c}/{len(items)}] item {item} (|M|={len(M)}) selesai")
    return pd.DataFrame(rows)


class ItemBasedItemKNN:
    """Item-Based skala penuh untuk ItemKNN TANPA pruning (k=None).

    Kenapa bisa cepat tapi tetap eksak (setara dengan melatih ulang):
    - Membuang interaksi item i hanya mengubah baris/kolom i dari G = XᵀX, dan
      norma item lain tidak berubah. Jadi sim(h, j) untuk h, j ≠ i tetap sama.
    - Item i dibuang dari semua riwayat test, sehingga baris i tidak pernah dipakai.
      Skor item lain (j ≠ i) TIDAK bergantung pada N.
    - Yang berubah terhadap N hanya kolom i: sim(h, i) = G'[h,i] / (‖h‖·√N + shrink),
      dengan G'[h,i] = jumlah dari N user tersisa yang juga berinteraksi dengan h.
    Jadi untuk tiap (i, N) cukup menghitung skor item i, lalu membandingkannya
    dengan daftar skor tertinggi item lain yang sudah dihitung sekali.
    Dengan cara ini setup paper bisa dijalankan penuh: S = 1000 item, semua user test,
    semua langkah successive evaluation.
    `split` harus dibuat dengan include_test_history_in_train=False.
    """

    def __init__(self, split: Split, n_users: int, n_items: int, shrink: float = 0.0,
                 Ks=(1, 5, 10, 50, 100)):
        self.split, self.n_items, self.shrink, self.Ks = split, n_items, shrink, tuple(Ks)
        self.Kp1 = max(Ks) + 1
        self.X = to_matrix(split.train, n_users, n_items).tocsc()
        self.S = ItemKNN(k=None, shrink=shrink).fit(self.X.tocsr()).S
        self.norms = np.sqrt(np.asarray(self.X.sum(axis=0)).ravel())
        self.users = list(split.test_hist)
        self.base = [split.test_hist[u] for u in self.users]
        self.hold = [split.test_holdout[u] for u in self.users]
        # Satu baris evaluasi per (user, t): riwayat = base + hold[:t]
        lens = np.array([len(h) for h in self.hold])
        self.row_user = np.repeat(np.arange(len(self.users)), lens)
        self.start = np.concatenate([[0], np.cumsum(lens)[:-1]])
        self.hold_flat = np.concatenate([np.asarray(h, dtype=np.int64) for h in self.hold])
        self.B = history_matrix(self.base, n_items)
        tops = [self._user_rows(b, h) for b, h in zip(self.base, self.hold)]
        self.topv = np.vstack([t[0] for t in tops])
        self.topi = np.vstack([t[1] for t in tops])
        self.contains = {}
        for k, (b, h) in enumerate(zip(self.base, self.hold)):
            for it in set(b) | set(h):
                self.contains.setdefault(it, []).append(k)

    def _user_rows(self, base, hold, drop=None):
        """Top-(Kmax+1) skor item lain per langkah successive evaluation satu user."""
        if drop is not None:
            base = [x for x in base if x != drop]
            hold = [x for x in hold if x != drop]
        T = len(hold)
        v = self.S[base].sum(axis=0) if base else np.zeros(self.n_items, np.float32)
        steps = np.vstack([np.zeros((1, self.n_items), np.float32),
                           np.cumsum(self.S[hold[:-1]], axis=0)]) if T > 1 else np.zeros((T, self.n_items), np.float32)
        M = v[None, :] + steps[:T]
        if base:
            M[:, base] = -np.inf
        for j, it in enumerate(hold[:-1]):        # item holdout ke-j sudah "dilihat" di langkah > j
            M[j + 1:, it] = -np.inf
        if drop is not None:
            M[:, drop] = -np.inf
        top = np.argpartition(-M, self.Kp1, axis=1)[:, :self.Kp1]
        return np.take_along_axis(M, top, axis=1), top, base, hold

    def column(self, item: int, keep_users) -> np.ndarray:
        """Kolom similarity sim(·, i) setelah item i hanya punya interaksi dari keep_users."""
        N = len(keep_users)
        co = np.asarray(self.X[keep_users].sum(axis=0)).ravel()   # G'[h, i]
        col = co / (self.norms * np.sqrt(N) + self.shrink + 1e-12)
        col[item] = 0.0
        return col

    def prepare(self, item: int):
        """Bagian yang tidak bergantung pada N: daftar skor item lain tanpa item i."""
        aff = set(self.contains.get(item, []))
        mask = ~np.isin(self.row_user, list(aff))
        O = np.where(self.topi[mask] == item, -np.inf, self.topv[mask])
        aff_rows = [self._user_rows(self.base[k], self.hold[k], drop=item) for k in sorted(aff)]
        if aff_rows:
            O = np.vstack([O] + [r[0] for r in aff_rows])
        return dict(mask=mask, O=O, aff=[(r[2], r[3]) for r in aff_rows])

    def ranks(self, prep, col: np.ndarray) -> np.ndarray:
        """Peringkat item i di setiap baris evaluasi (inf kalau di luar top-Kmax)."""
        ce = np.concatenate([[0.0], np.cumsum(col[self.hold_flat])])
        r = np.arange(len(self.row_user))
        score = (self.B @ col)[self.row_user] + ce[r] - ce[self.start[self.row_user]]
        parts = [score[prep["mask"]]]
        for base, hold in prep["aff"]:
            b = col[base].sum() if base else 0.0
            parts.append(b + np.concatenate([[0.0], np.cumsum(col[hold])])[:len(hold)])
        score = np.concatenate(parts)
        # Skor seri dihitung pesimis: item i berada di bawah item lain yang skornya sama.
        rank = 1 + (prep["O"] >= score[:, None]).sum(axis=1)
        return np.where(rank <= max(self.Ks), rank, np.inf)

    def metrics(self, rank: np.ndarray) -> dict:
        out = {}
        for K in self.Ks:
            hit = rank <= K
            out[f"HR*@{K}"] = float(hit.mean())
            out[f"NDCG*@{K}"] = float(np.where(hit, 1.0 / np.log2(np.where(hit, rank, 1) + 1), 0.0).mean())
        return out

    def curve(self, Ns=DEFAULT_NS, n_items_sample: int = 1000, seed: int = 0,
              verbose: bool = True) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        train = self.split.train
        pool = np.unique(train["i"].values)
        items = rng.choice(pool, size=min(n_items_sample, len(pool)), replace=False)
        rows = []
        for c, item in enumerate(items, 1):
            M_users = train.loc[train["i"].values == item, "u"].values      # M(i)
            prep = self.prepare(item)
            for N in Ns:
                if N > len(M_users):
                    break
                keep = rng.choice(M_users, size=N, replace=False)
                rows.append(dict(item=int(item), N=N, n_train=len(M_users),
                                 **self.metrics(self.ranks(prep, self.column(item, keep)))))
            if verbose and c % 100 == 0:
                print(f"[{c}/{len(items)}] item selesai")
        return pd.DataFrame(rows)


def aggregate_item_based(df: pd.DataFrame) -> pd.DataFrame:
    """Eq. (3)-(4): rata-rata atas item dengan >= N interaksi, ditambah 95% CI antar item."""
    metrics = [c for c in df.columns if c.startswith(("HR*", "NDCG*"))]
    g = df.groupby("N")
    out = g[metrics].mean()
    for m in metrics:
        out[m + "_ci"] = g[m].apply(ci95)
    out["n_items"] = g.size()
    return out.reset_index()


# --------------------------------------------------------------------------- threshold
def detect_threshold(Ns, values, window: int = DEFAULT_WINDOW):
    """Sliding window least squares (Sec 4.2).

    Untuk tiap window `window` titik berturut-turut di sepanjang N, hitung slope
    garis terbaik y = a·N + b (least squares). Window dengan slope positif
    tercuram dipilih. Threshold = N terakhir di window itu, yaitu titik saat
    kenaikan tajam selesai (ASUMSI, karena paper tidak menyebut titik mana).
    Mengembalikan (threshold, tabel slope).
    """
    Ns, values = np.asarray(Ns, float), np.asarray(values, float)
    rows = []
    for s in range(len(Ns) - window + 1):
        x, y = Ns[s:s + window], values[s:s + window]
        a, _ = np.polyfit(x, y, 1)
        rows.append(dict(start=Ns[s], end=Ns[s + window - 1], slope=a))
    tab = pd.DataFrame(rows)
    pos = tab[tab["slope"] > 0]
    if pos.empty:
        return None, tab
    best = pos.loc[pos["slope"].idxmax()]
    return int(best["end"]), tab
