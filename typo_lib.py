# -*- coding: utf-8 -*-
"""
typo_lib.py
===========
Pustaka inti (shared library) untuk sistem DETEKSI & KOREKSI TYPO Bahasa Indonesia
berbasis *Traditional Machine Learning* (tanpa deep learning sama sekali).

Berisi:
  1. Fungsi util: tokenizer, edit distance (Damerau-Levenshtein), pola keyboard.
  2. Kelas `TypoCorrector` yang membungkus:
        - model deteksi level-KATA   (Pipeline TF-IDF char n-gram + classifier)
        - model deteksi level-KALIMAT (Pipeline TF-IDF word+char + classifier)
        - kamus koreksi (vocabulary + frekuensi kata benar)
        - indeks kemiripan (TF-IDF char n-gram untuk mencari kandidat koreksi)
     Beserta fungsi:  detect_typo(), suggest_correction(), correct_sentence(),
     serta save()/load() memakai joblib.

Catatan desain:
  - File ini dipakai BERSAMA oleh notebook (untuk evaluasi & demo) dan oleh
    `typo_app.py` (menu terminal), agar logika inferensi tidak terduplikasi.
  - SEMUA komentar & pesan dalam Bahasa Indonesia (sesuai permintaan).
"""

import os
import re
import math
from collections import Counter

import numpy as np
import joblib


# ===========================================================================
# 1. UTILITAS DASAR
# ===========================================================================

# Regex tokenizer: mengambil "kata" termasuk kata berhubung (mis. "siap-siap").
# \w mencakup huruf & angka; (?:-\w+)* mempertahankan tanda hubung di tengah kata.
_TOKEN_RE = re.compile(r"\w+(?:-\w+)*", flags=re.UNICODE)

# Regex untuk memecah teks menjadi token + non-token (spasi/tanda baca) SEKALIGUS,
# sehingga saat merangkai kembali kalimat, spasi & tanda baca asli tetap terjaga.
_SPLIT_RE = re.compile(r"(\w+(?:-\w+)*)", flags=re.UNICODE)

# Huruf vokal Bahasa Indonesia (dipakai untuk fitur rasio vokal, dsb.)
_VOWELS = set("aiueo")

# Peta tetangga papan ketik QWERTY (untuk pola typo "salah pencet tombol sebelah").
# Dipakai memberi BONUS pada kandidat koreksi yang berbeda 1 huruf tetangga keyboard.
_KEYBOARD_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg",
    "y": "tugh", "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc", "g": "ftyhbv",
    "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}


def tokenize(text):
    """Pecah teks menjadi daftar token kata (huruf dikecilkan/lowercase)."""
    return _TOKEN_RE.findall(str(text).lower())


def split_keep_tokens(text):
    """
    Pecah teks menjadi potongan-potongan bergantian: [non-kata, kata, non-kata, ...].
    Berguna untuk mengganti hanya token kata yang salah lalu merangkai ulang
    kalimat dengan spasi & tanda baca aslinya tetap utuh.
    """
    return _SPLIT_RE.split(str(text))


def damerau_levenshtein(a, b):
    """
    Hitung jarak edit Damerau-Levenshtein antara dua string.
    Operasi yang dihitung: sisip (insertion), hapus (deletion), substitusi,
    dan TRANSPOSISI dua huruf bersebelahan -> cocok untuk typo bahasa.
    """
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # Matriks DP berukuran (la+1) x (lb+1).
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # hapus
                d[i][j - 1] + 1,        # sisip
                d[i - 1][j - 1] + cost  # substitusi / sama
            )
            # Transposisi dua huruf bersebelahan.
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def _is_keyboard_typo(a, b):
    """
    True bila `a` dan `b` hanya berbeda 1 substitusi huruf yang BERSEBELAHAN
    di papan ketik QWERTY (mis. 'bunyi' vs 'bunyu' -> i/u bertetangga).
    """
    if len(a) != len(b):
        return False
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    if len(diff) != 1:
        return False
    x, y = diff[0]
    return y in _KEYBOARD_NEIGHBORS.get(x, "")


def word_shape_features(word):
    """
    Fitur numerik sederhana sebuah kata (dipakai untuk eksplorasi/visualisasi
    fitur tradisional: panjang kata, rasio vokal, deret konsonan terpanjang, dll).
    Mengembalikan dict agar mudah ditampilkan di notebook.
    """
    w = str(word).lower()
    n = len(w)
    n_vowel = sum(ch in _VOWELS for ch in w)
    # Deret konsonan terpanjang (typo sering memunculkan tumpukan konsonan).
    max_cons = cur = 0
    for ch in w:
        if ch.isalpha() and ch not in _VOWELS:
            cur += 1
            max_cons = max(max_cons, cur)
        else:
            cur = 0
    # Karakter berulang berturut-turut (mis. "bukuu").
    has_repeat = any(w[i] == w[i + 1] for i in range(n - 1))
    return {
        "panjang": n,
        "rasio_vokal": (n_vowel / n) if n else 0.0,
        "deret_konsonan_max": max_cons,
        "ada_angka": int(any(ch.isdigit() for ch in w)),
        "ada_huruf_berulang": int(has_repeat),
    }


# ===========================================================================
# 2. KELAS UTAMA: TypoCorrector
# ===========================================================================

class TypoCorrector:
    """
    Pembungkus seluruh komponen sistem deteksi & koreksi typo.

    Parameter
    ---------
    word_model : sklearn Pipeline
        Model deteksi level-kata (TF-IDF char n-gram -> classifier).
    sentence_model : sklearn Pipeline
        Model deteksi level-kalimat (TF-IDF word+char -> classifier).
    correction_freq : dict {kata_benar: frekuensi}
        Kamus kata benar beserta frekuensi kemunculan (dari kolom kalimat_awal).
    index_vectorizer : fitted TfidfVectorizer
        Vectorizer char n-gram untuk mencari kandidat koreksi via kemiripan cosine.
    index_matrix : sparse matrix (n_vocab, n_fitur)
        Matriks TF-IDF dari seluruh kosakata (urut sesuai `vocab`).
    vocab : list[str]
        Daftar kata benar (urut sama dengan baris `index_matrix`).
    config : dict
        Ambang batas & bobot skor (lihat default_config()).
    """

    def __init__(self, word_model, sentence_model, correction_freq,
                 index_vectorizer, index_matrix, vocab, config=None):
        self.word_model = word_model
        self.sentence_model = sentence_model
        self.correction_freq = dict(correction_freq)
        self.index_vectorizer = index_vectorizer
        self.index_matrix = index_matrix
        self.vocab = list(vocab)
        self.vocab_set = set(self.vocab)
        self.config = config or self.default_config()
        # Frekuensi maksimum (untuk normalisasi skor frekuensi 0..1).
        self._max_log_freq = math.log1p(max(self.correction_freq.values())) if self.correction_freq else 1.0

    # ---------------------------------------------------------------- config
    @staticmethod
    def default_config():
        """Ambang & bobot default untuk deteksi dan pemeringkatan koreksi."""
        return {
            "word_threshold": 0.5,       # ambang probabilitas typo level-kata
            "sentence_threshold": 0.5,   # ambang probabilitas typo level-kalimat
            "top_n_candidates": 50,      # kandidat awal dari kemiripan cosine
            "top_k": 3,                  # jumlah saran koreksi yang dikembalikan
            # Bobot skor pemeringkatan kandidat koreksi (dijumlahkan).
            # Nilai ini dipilih lewat grid-search kecil pada pasangan held-out
            # (memaksimalkan akurasi koreksi Top-1/Top-3 untuk typo di luar kamus).
            "w_cosine": 0.30,            # kemiripan n-gram karakter (TF-IDF)
            "w_editdist": 0.40,          # 1 - jarak edit ternormalisasi
            "w_freq": 0.25,              # frekuensi kata (kata umum diprioritaskan)
            "w_keyboard": 0.05,          # bonus pola typo tetangga keyboard
            "correction_threshold": 0.30,  # ambang lebih longgar untuk menargetkan koreksi
        }

    # -------------------------------------------------------------- deteksi
    def _word_typo_proba(self, word):
        """Probabilitas (0..1) bahwa sebuah KATA adalah typo, dari model ML."""
        return float(self.word_model.predict_proba([word.lower()])[0][1])

    def _sentence_typo_proba(self, text):
        """Probabilitas (0..1) bahwa sebuah KALIMAT mengandung typo, dari model ML."""
        return float(self.sentence_model.predict_proba([text])[0][1])

    def detect_typo(self, text):
        """
        Deteksi apakah `text` (kata atau kalimat) mengandung typo.

        Mengembalikan dict:
          - input        : teks asli
          - level        : 'kata' atau 'kalimat'
          - is_typo      : bool (kesimpulan akhir)
          - confidence   : probabilitas pendukung kesimpulan (0..1)
          - word_proba   : prob typo dari model kata (untuk input 1 kata)
          - sent_proba   : prob typo dari model kalimat (untuk input kalimat)
          - flagged      : daftar kata yang ditandai sebagai typo
          - word_probs   : dict {kata: prob_typo} untuk tiap token (level kalimat)
        """
        text = str(text).strip()
        tokens = tokenize(text)
        wt = self.config["word_threshold"]
        st = self.config["sentence_threshold"]

        # ---- Kasus 1: input hanya SATU kata -> pakai model level-kata.
        if len(tokens) <= 1:
            word = tokens[0] if tokens else ""
            if not word:
                return {"input": text, "level": "kata", "is_typo": False,
                        "confidence": 0.0, "word_proba": 0.0, "sent_proba": None,
                        "flagged": [], "word_probs": {}}
            # Lookup kamus: bila kata persis ada di kosakata benar -> anggap benar.
            in_vocab = word in self.vocab_set
            wp = self._word_typo_proba(word)
            is_typo = (not in_vocab) and (wp >= wt)
            return {
                "input": text, "level": "kata", "is_typo": is_typo,
                "confidence": wp if is_typo else (1.0 - wp),
                "word_proba": wp, "sent_proba": None,
                "flagged": [word] if is_typo else [],
                "word_probs": {word: wp},
            }

        # ---- Kasus 2: input KALIMAT -> gabungkan model kalimat + model kata.
        sp = self._sentence_typo_proba(text)
        flagged = []
        word_probs = {}
        for tok in tokens:
            if tok in self.vocab_set:
                word_probs[tok] = 0.0  # ada di kamus -> dianggap benar (lookup)
                continue
            p = self._word_typo_proba(tok)
            word_probs[tok] = p
            if p >= wt:
                flagged.append(tok)
        # Kesimpulan: typo bila model kalimat menandai ATAU ada kata yang ditandai.
        is_typo = (sp >= st) or (len(flagged) > 0)
        return {
            "input": text, "level": "kalimat", "is_typo": is_typo,
            "confidence": sp if is_typo else (1.0 - sp),
            "word_proba": None, "sent_proba": sp,
            "flagged": flagged, "word_probs": word_probs,
        }

    # -------------------------------------------------------------- koreksi
    def candidates(self, word, k=None):
        """
        Hasilkan daftar kandidat koreksi untuk sebuah `word`, terurut dari
        skor tertinggi: [(kata_kandidat, skor), ...].

        Metode (semua NON-deep-learning):
          1. Kemiripan cosine TF-IDF n-gram karakter -> ambil top-N kandidat.
          2. Pemeringkatan ulang dengan kombinasi:
               cosine + (1 - jarak edit ternormalisasi) + frekuensi + bonus keyboard.
        """
        k = k or self.config["top_k"]
        w = str(word).lower()
        if not w:
            return []

        cfg = self.config
        # (1) Kemiripan cosine: matrix (n_vocab x f) dot query (f x 1) = skor cosine
        #     (TF-IDF sudah ter-normalisasi L2, jadi dot product = cosine similarity).
        qv = self.index_vectorizer.transform([w])
        sims = np.asarray(self.index_matrix.dot(qv.T).todense()).ravel()
        n = min(cfg["top_n_candidates"], len(self.vocab))
        # argpartition: ambil index N skor terbesar (tanpa mengurutkan semuanya).
        top_idx = np.argpartition(-sims, n - 1)[:n]

        # (2) Pemeringkatan ulang dengan skor gabungan.
        scored = []
        for idx in top_idx:
            cand = self.vocab[idx]
            cos = float(sims[idx])
            dl = damerau_levenshtein(w, cand)
            norm_dist = dl / max(len(w), len(cand), 1)         # 0..1
            freq = self.correction_freq.get(cand, 1)
            norm_freq = math.log1p(freq) / self._max_log_freq  # 0..1
            kb = 1.0 if _is_keyboard_typo(w, cand) else 0.0
            score = (cfg["w_cosine"] * cos
                     + cfg["w_editdist"] * (1.0 - norm_dist)
                     + cfg["w_freq"] * norm_freq
                     + cfg["w_keyboard"] * kb)
            scored.append((cand, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def suggest_correction(self, text):
        """
        Sarankan koreksi untuk `text`.

        Mengembalikan dict:
          - input        : teks asli
          - is_typo      : bool
          - corrected    : teks setelah dikoreksi (kalimat utuh / kata)
          - suggestions  : {kata_salah: [(kandidat, skor), ...]}
          - message      : pesan ramah pengguna (Bahasa Indonesia)
        """
        det = self.detect_typo(text)
        if not det["is_typo"]:
            return {"input": text, "is_typo": False, "corrected": text,
                    "suggestions": {}, "message": "Tidak ada typo terdeteksi. "
                                                   "Tidak ada koreksi yang dibutuhkan."}

        # Tentukan kata yang akan dikoreksi.
        words_to_fix = list(det["flagged"])
        if not words_to_fix:
            # Fallback: kalimat terdeteksi typo (oleh model kalimat) tetapi tidak
            # ada kata yang melewati ambang deteksi kata. Targetkan kata di luar
            # kamus (OOV) yang paling mencurigakan agar koreksi tetap diberikan.
            ct = self.config.get("correction_threshold", 0.30)
            oov = [(w, p) for w, p in det.get("word_probs", {}).items()
                   if w not in self.vocab_set]
            oov.sort(key=lambda x: x[1], reverse=True)
            words_to_fix = [w for w, p in oov if p >= ct]
            if not words_to_fix and oov:
                words_to_fix = [oov[0][0]]   # best-effort: 1 kata OOV paling mencurigakan

        # Cari kandidat untuk tiap kata; simpan hanya bila ada saran yang berbeda.
        suggestions = {}
        for w in words_to_fix:
            cands = self.candidates(w, k=self.config["top_k"])
            if cands and cands[0][0] != w:
                suggestions[w] = cands

        if not suggestions:
            return {"input": text, "is_typo": True, "corrected": text,
                    "suggestions": {},
                    "message": "Terindikasi typo, namun tidak ada saran koreksi yang ditemukan."}

        # Rangkai ulang teks: ganti hanya token yang dikoreksi (top-1),
        # pertahankan kapitalisasi & tanda baca asli.
        corrected = self._rebuild(text, list(suggestions.keys()), suggestions)
        return {"input": text, "is_typo": True, "corrected": corrected,
                "suggestions": suggestions,
                "message": "Typo terdeteksi. Berikut saran koreksinya."}

    def correct_sentence(self, text):
        """Shortcut: kembalikan hanya string kalimat yang sudah dikoreksi."""
        return self.suggest_correction(text)["corrected"]

    def _rebuild(self, text, flagged, suggestions):
        """Ganti token typo dengan kandidat top-1 sambil menjaga bentuk asli teks."""
        flagged_set = set(flagged)
        parts = split_keep_tokens(text)
        for i, part in enumerate(parts):
            low = part.lower()
            if low in flagged_set and suggestions.get(low):
                best = suggestions[low][0][0]            # kandidat skor tertinggi
                parts[i] = _match_case(part, best)       # samakan pola kapitalisasi
        return "".join(parts)

    # --------------------------------------------------------- simpan/muat
    def save(self, models_dir="models"):
        """Simpan semua artefak (model, vectorizer, kamus) memakai joblib."""
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.word_model, os.path.join(models_dir, "word_model.joblib"))
        joblib.dump(self.sentence_model, os.path.join(models_dir, "sentence_model.joblib"))
        joblib.dump({"freq": self.correction_freq, "vocab": self.vocab},
                    os.path.join(models_dir, "correction_dict.joblib"))
        joblib.dump({"vectorizer": self.index_vectorizer, "matrix": self.index_matrix,
                     "vocab": self.vocab},
                    os.path.join(models_dir, "correction_index.joblib"))
        joblib.dump(self.config, os.path.join(models_dir, "metadata.joblib"))

    @classmethod
    def load(cls, models_dir="models"):
        """Muat seluruh artefak dari folder `models_dir` dan bangun TypoCorrector."""
        word_model = joblib.load(os.path.join(models_dir, "word_model.joblib"))
        sentence_model = joblib.load(os.path.join(models_dir, "sentence_model.joblib"))
        cdict = joblib.load(os.path.join(models_dir, "correction_dict.joblib"))
        cindex = joblib.load(os.path.join(models_dir, "correction_index.joblib"))
        config = joblib.load(os.path.join(models_dir, "metadata.joblib"))
        return cls(
            word_model=word_model,
            sentence_model=sentence_model,
            correction_freq=cdict["freq"],
            index_vectorizer=cindex["vectorizer"],
            index_matrix=cindex["matrix"],
            vocab=cindex["vocab"],
            config=config,
        )


def _match_case(original, replacement):
    """Samakan pola kapitalisasi `replacement` dengan token `original`."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


# ===========================================================================
# 3. MENU INTERAKTIF (dipakai notebook & typo_app.py)
# ===========================================================================

def run_menu(corrector=None, models_dir="models"):
    """
    Jalankan menu terminal interaktif (Bahasa Indonesia).

    - Jika `corrector` tidak diberikan, artefak dimuat dari `models_dir`.
    - Aman dijalankan di lingkungan non-interaktif (mis. eksekusi otomatis
      notebook): bila stdin tidak tersedia, menu akan berhenti dengan rapi.
    """
    if corrector is None:
        corrector = TypoCorrector.load(models_dir)

    print("=" * 60)
    print("  SISTEM DETEKSI & KOREKSI TYPO (Traditional ML)")
    print("=" * 60)

    while True:
        print("\nMENU:")
        print("  1. Periksa kata / kalimat (deteksi + koreksi)")
        print("  2. Keluar")
        try:
            pilihan = input("Pilih menu (1/2): ").strip()
        except (EOFError, Exception):
            # Lingkungan tanpa input interaktif -> keluar dengan rapi.
            print("\n[Info] Input interaktif tidak tersedia. Menu dihentikan.")
            return

        if pilihan == "2":
            print("Terima kasih! Program selesai.")
            return
        if pilihan != "1":
            print("Pilihan tidak valid. Silakan masukkan 1 atau 2.")
            continue

        try:
            teks = input("Masukkan kata/kalimat: ").strip()
        except (EOFError, Exception):
            print("\n[Info] Input interaktif tidak tersedia. Menu dihentikan.")
            return

        # Validasi: tolak input kosong.
        if not teks:
            print(">> Input kosong tidak diperbolehkan. Silakan coba lagi.")
            continue

        # Deteksi + koreksi.
        hasil = corrector.suggest_correction(teks)
        det = corrector.detect_typo(teks)

        print("-" * 60)
        if det["is_typo"]:
            print(f"Status      : ADA TYPO terdeteksi")
        else:
            print(f"Status      : TIDAK ADA typo")
        # Tampilkan confidence/probabilitas model.
        if det["level"] == "kata":
            print(f"Level       : kata")
            print(f"Prob. typo  : {det['word_proba']:.2%} (model kata)")
        else:
            print(f"Level       : kalimat")
            print(f"Prob. typo  : {det['sent_proba']:.2%} (model kalimat)")
        print(f"Confidence  : {det['confidence']:.2%}")

        if hasil["is_typo"]:
            print(f"Kata salah  : {', '.join(hasil['suggestions'].keys())}")
            print(f"Koreksi     : {hasil['corrected']}")
            print("Saran kandidat (top-k):")
            for w, cands in hasil["suggestions"].items():
                opsi = ", ".join(f"{c} ({s:.2f})" for c, s in cands)
                print(f"   - {w} -> {opsi}")
        else:
            print(">> Tidak ada koreksi yang dibutuhkan.")
        print("-" * 60)


if __name__ == "__main__":
    # Jika dijalankan langsung: muat artefak dari folder models/ lalu buka menu.
    run_menu()
