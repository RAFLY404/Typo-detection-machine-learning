# -*- coding: utf-8 -*-
"""
typo_app.py
===========
Aplikasi MENU TERMINAL untuk menguji sistem Deteksi & Koreksi Typo.

Cara pakai:
    1. Pastikan notebook `typo_detection_correction.ipynb` sudah dijalankan
       sampai selesai sehingga folder `models/` berisi artefak terlatih.
    2. Jalankan:  python typo_app.py

Program akan memuat model & kamus dari folder `models/`, lalu menampilkan
menu interaktif (Bahasa Indonesia) untuk menguji banyak input tanpa restart.
"""

import os
import sys

from typo_lib import TypoCorrector, run_menu

MODELS_DIR = "models"


def main():
    # Pastikan artefak hasil pelatihan tersedia.
    if not os.path.isdir(MODELS_DIR) or not os.path.exists(
        os.path.join(MODELS_DIR, "word_model.joblib")
    ):
        print("[ERROR] Folder 'models/' atau artefak model tidak ditemukan.")
        print("        Jalankan dulu notebook 'typo_detection_correction.ipynb'")
        print("        sampai selesai untuk membuat file model.")
        sys.exit(1)

    print("Memuat model dan kamus koreksi ...")
    corrector = TypoCorrector.load(MODELS_DIR)
    print("Model berhasil dimuat.\n")

    # Buka menu interaktif (logika menu ada di typo_lib.run_menu).
    run_menu(corrector)


if __name__ == "__main__":
    main()
