# Program & Panduan Hands-On — Sesi Siang
## BNI Officer Development Program Batch 2 — Training MLOps & AI Engineering Fundamentals (Day 19)
### Modul 3 (Lab Sesi 1 — Praktik Ringan) & Modul 4 (Lab Sesi 2 — Hands-On Penuh)

---

## 0. Ringkasan Program Siang

| Modul | Nama | Sifat | Estimasi Waktu |
|---|---|---|---|
| Modul 3 — Lab Sesi 1 | Setup & Versioning (Praktik Ringan) | Fondasi sebelum hands-on penuh | ±70 menit |
| Modul 4 — Lab Sesi 2 | Training s.d. Monitoring (Hands-On Penuh) | Praktik utama sesi siang | ±170 menit |
| **Total** | | | **±240 menit (4 jam, termasuk jeda antar langkah)** |

> Catatan: durasi Langkah 1–4 Modul 4 (30'/25'/25'/40') diambil langsung dari slide asli. Durasi Modul 3 dan Langkah 5–6 Modul 4 adalah estimasi wajar berdasarkan kompleksitas materi — sesuaikan dengan kecepatan kelas dan sisipkan rehat 10–15 menit di antara Modul 3 dan Modul 4.

---

## 1. Struktur Folder yang Digunakan Peserta

Setelah mengekstrak paket ini, struktur project peserta sebaiknya:

```
mlops-training/
├── data/
│   ├── train.csv              (3.000 baris)
│   ├── train_v2.csv           (3.000 baris + 1 kolom tambahan)
│   └── production_simulasi.csv (1.200 baris, distribusi income bergeser)
├── scripts/               (atau taruh di root, sesuaikan path CSV)
│   ├── eksperimen1.py
│   ├── train.py
│   ├── validate_data.py
│   ├── check_model_gate.py
│   ├── serve.py
│   └── check_drift.py
├── requirements.txt
├── requirements-api.txt
├── Dockerfile
└── .github/workflows/ml-ci-cd.yml
```

Semua script mengasumsikan dijalankan dari **root project** (`data/train.csv`, bukan `../data/train.csv`). Jika peserta menaruh script di dalam folder `scripts/`, jalankan dari root: `python scripts/train.py 100 5`, atau pindahkan isi `scripts/` ke root.

---

## MODUL 3 — LAB SESI 1: Setup & Versioning

### Prasyarat Teknis
- Python 3.10+ terpasang (`python --version`)
- Git terpasang (`git --version`)
- Code editor (VS Code / sejenis) & akses terminal
- Data lab sudah diunduh dari LMS (paket ini)
- Akun Git (GitHub/GitLab) aktif

### Definition of Done — Sesi 1
- [ ] Repo Git aktif dengan minimal 1 commit
- [ ] DVC terinisialisasi
- [ ] Bisa berpindah antar versi dataset dengan `dvc checkout`
- [ ] Minimal 1 run eksperimen tercatat lengkap di MLflow UI

### Langkah 1 — Setup Environment (15')

Buat virtual environment agar dependensi project terisolasi dari sistem.

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
# prompt harus menampilkan (venv)
```

**macOS / Linux (bash/zsh):**
```bash
python3 -m venv venv
source venv/bin/activate
# prompt harus menampilkan (venv)
```

**Install dependensi (semua OS, setelah venv aktif):**
```bash
pip install pandas scikit-learn mlflow dvc
```

**Poin penting**: aktifkan `(venv)` SEBELUM setiap `pip install`; kalau lupa, package akan terpasang di sistem global dan bisa bentrok dengan project lain.

✓ **Checkpoint**: tanda `(venv)` muncul di awal prompt terminal, dan `pip list` menampilkan `mlflow` serta `dvc`.

### Langkah 2 — Init Git & DVC (15')

DVC "menumpang" di atas Git — Git tetap menjadi sumber kebenaran riwayat proyek.

```bash
git init
# Harus muncul: Initialized empty Git repository in .../mlops-training/.git/

dvc init
# Harus muncul: pesan konfirmasi tanpa error; folder .dvc/ baru terbentuk

git add .dvc .gitignore
git commit -m "chore: initialize project with DVC"
# Harus muncul: ringkasan commit, mis. "2 files changed, ... insertions"
```

✓ **Checkpoint**: jalankan `git log --oneline` — minimal muncul 1 baris commit.

### Langkah 3 — Versioning Dataset & Kembali ke Versi Lama (20')

**A. Daftarkan & buat versi pertama:**
```bash
dvc add data/train.csv
git add data/train.csv.dvc data/.gitignore
git commit -m "chore: track training data v1"
```

**B. Simulasikan data baru (kolom tambahan) & buat versi kedua:**
```bash
cp data/train_v2.csv data/train.csv
dvc add data/train.csv
git add data/train.csv.dvc
git commit -m "chore: add feature column v2"
```

**C. Kembali ke versi lama, lalu balik lagi:**
```bash
git log --oneline               # catat hash commit v1
git checkout <hash_commit_v1> -- data/train.csv.dvc
dvc checkout
# data/train.csv sekarang kembali ke versi tanpa kolom tambahan

git checkout main -- data/train.csv.dvc   # atau nama branch Anda
dvc checkout
# data/train.csv kembali ke versi terbaru (dengan kolom tambahan)
```

**Mengapa ini penting**: kemampuan berpindah versi dataset kapan saja adalah fondasi *reproducibility* — siapa pun bisa mengulang eksperimen dengan data persis sama.

✓ **Checkpoint**: `data/train.csv` memiliki kolom tambahan (`jumlah_tanggungan`) setelah langkah terakhir, dan `git log` menampilkan minimal 2 commit data.

### Langkah 4 — Eksperimen Pertama dengan MLflow (20')

File: `scripts/eksperimen1.py` (tersedia di paket ini).

```bash
python scripts/eksperimen1.py
# → baris "Akurasi: 0.xxxx" muncul di terminal

mlflow ui
# → dashboard aktif di http://localhost:5000 (default)
```

**Port bentrok?** `mlflow ui --port <nomor_port>`

✓ **Checkpoint**: MLflow UI menampilkan minimal 1 run dengan parameter & metric tercatat.

### Troubleshooting Sesi 1

| Gejala | Penyebab Umum | Solusi |
|---|---|---|
| `(venv)` tidak muncul | Salah perintah aktivasi OS | Ulangi Langkah 1, pilih perintah sesuai OS Anda |
| `dvc: command not found` | venv tidak aktif saat install | Aktifkan venv, ulangi `pip install dvc` |
| `dvc add` error "not a git repository" | Lupa `git init` dulu | Jalankan `git init` sebelum `dvc init` |
| `KeyError: 'target'` | Nama kolom target berbeda di dataset | Cek `train.csv`, sesuaikan nama kolom di script |
| MLflow UI browser blank | Port 5000 dipakai proses lain | Gunakan `--port 5001`, sesuaikan URL akses |

### Self-Assessment Sesi 1
- [ ] Environment aktif & dependensi lengkap
- [ ] Git + DVC terinisialisasi dengan benar
- [ ] Minimal 2 versi dataset tercatat & bisa di-checkout
- [ ] Minimal 1 run eksperimen tercatat di MLflow

---

## MODUL 4 — LAB SESI 2: Training, Gate, Packaging, CI/CD & Monitoring

Melanjutkan repo dari Sesi 1.

### Langkah 1 — Multi-Eksperimen (30')

File: `scripts/train.py` (mengambil argumen `n_estimators` dan `max_depth`, mencatat AUC ke experiment `"training-siang"`).

```bash
python scripts/train.py 100 5
python scripts/train.py 200 8
python scripts/train.py 300 12
```

Lalu buka `mlflow ui`, urutkan kolom AUC dari terbesar.

| n_estimators | max_depth | AUC (contoh) |
|---|---|---|
| 100 | 5 | ~0.74–0.75 |
| 200 | 8 | ~0.74–0.75 |
| 300 | 12 | ~0.74–0.75 |

*Angka Anda boleh berbeda — catat run terbaik untuk dipakai di Langkah 2–4.*

✓ **Checkpoint**: ada 3 run di experiment `"training-siang"`, masing-masing dengan parameter dan AUC berbeda. Catat run mana yang terbaik dan berapa `n_estimators`/`max_depth`-nya.

### Langkah 2 — Data Quality Gate — `validate_data.py` (25')

File: `scripts/validate_data.py` (menggunakan Great Expectations).

```bash
python -c "import pandas as pd; print(pd.read_csv('data/train.csv').columns.tolist())"
python scripts/validate_data.py
```

- ✅ **LOLOS** → pipeline lanjut ke training
- ❌ **GAGAL** → pipeline dihentikan (`exit 1`)

Expectation yang dicek: `income` tidak null, `age` tidak null dan berada di rentang 18–100, `target` hanya berisi {0, 1}.

**Latihan di kelas (poin penting)**:
1. Ubah 1 nilai `age` di `data/train.csv` menjadi `200` → jalankan ulang → script berhenti dengan `exit code 1`
2. Perbaiki kembali → jalankan ulang → script lolos dan pipeline lanjut ke training

Validasi data mencegah data kotor bahkan sampai ke tahap training.

### Langkah 3 — Model Quality Gate — `check_model_gate.py` (25')

File: `scripts/check_model_gate.py` (`MINIMUM_AUC = 0.75`).

```bash
python scripts/check_model_gate.py 0.81
echo $?              # bash/zsh — harus 0
echo $LASTEXITCODE   # PowerShell — harus 0

python scripts/check_model_gate.py 0.60
echo $?              # harus 1 (GAGAL)
```

**Poin penting**:
- Model tidak lolos ke production tanpa memenuhi standar AUC minimum yang disepakati bersama tim bisnis
- Exit code inilah yang nanti dipakai CI/CD untuk otomatis menghentikan pipeline
- Angka threshold adalah kesepakatan bisnis, bukan angka sembarang dari data scientist

**Quiz**: Siapakah yang menentukan standar minimum AUC? → *Standar minimum AUC yang disepakati bersama tim bisnis.*

### Langkah 4 — Packaging: FastAPI & Docker (40')

1. Cari path model terbaik dari MLflow. Di `mlflow ui`, klik run dengan AUC tertinggi → tab **Artifacts** → salin path model.
2. File `scripts/serve.py` (FastAPI, endpoint `/health` dan `/predict`) sudah tersedia.
3. Uji dulu dengan `uvicorn` langsung (tanpa Docker) supaya bug mudah dilacak:

```bash
# salin folder model dari MLflow ke ./model terlebih dahulu, lalu:
uvicorn scripts.serve:app --host 0.0.0.0 --port 8080
```

Buka tab browser baru: `http://localhost:8080/health` → harus muncul `{"status":"ok"}`.

4. Buat `requirements-api.txt` (sudah tersedia di paket ini — isi: `pandas`, `scikit-learn`, `mlflow`, `fastapi`, `uvicorn`).
5. `Dockerfile` sudah tersedia di paket ini. Sesuaikan baris `COPY model ./model` dengan lokasi model Anda.

```bash
docker build -t mlops-training-model:v1 .
docker run -p 8080:8080 mlops-training-model:v1
```

Di terminal/shell lain:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "income": 8000000, "lama_bekerja_tahun": 5, "skor_kredit_internal": 700, "jumlah_pinjaman_aktif": 1}'
```

**Port bentrok?** Ganti mapping: `docker run -p 8081:8080 ...`

✓ **Checkpoint**: container berjalan dan endpoint `/predict` mengembalikan hasil prediksi valid (respons JSON berisi `{"prediction": ..., "probability": ...}`).

#### Troubleshooting Langkah 4

| Gejala | Penyebab Umum | Solusi |
|---|---|---|
| `roc_auc_score error: only one class` | Data terlalu kecil/tidak seimbang setelah split | Gunakan dataset dari fasilitator, bukan subset sendiri |
| `great_expectations` error saat inisialisasi context | Versi library tidak cocok | `pip install --upgrade great_expectations` |
| Docker build sangat lambat | Jaringan kantor/kampus lambat | Tunggu, atau minta image pre-built sebagai fallback |
| `docker run`: port already in use | Port 8080 dipakai proses lain | Ganti mapping `-p 8081:8080`, akses via `localhost:8081` |
| `curl: Connection refused` | Container belum start / crash | Cek `docker ps` lalu `docker logs <container_id>` |

### Langkah 5 — Pipeline CI/CD dengan GitHub Actions (30')

1. Pastikan repo lokal sudah terhubung ke GitHub/GitLab. Jika belum:
```bash
git remote add origin https://github.com/<username-anda>/mlops-training.git
```
2. Pastikan `requirements.txt` lengkap di root project.
3. File workflow `.github/workflows/ml-ci-cd.yml` sudah tersedia di paket ini:

```yaml
name: ml-ci-cd
on: [push]
jobs:
  test-and-train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: python scripts/validate_data.py
      - run: python scripts/train.py 200 8
      - name: Cek model gate
        run: python scripts/check_model_gate.py 0.70
```

4. Commit dan push:
```bash
mkdir -p .github/workflows
git add .
git commit -m "ci: add GitHub Actions pipeline for data/model validation"
git branch -M main          # jika ada error terkait nama branch (main vs master)
git push -u origin main
```
5. Buka repo di GitHub, klik tab **Actions**. Workflow run akan tampil: ikon kuning (berjalan) → hijau ✅ (sukses) atau merah ❌ (gagal).

**Urutan eksekusi CI/CD**: 1) Checkout kode → 2) Install dependencies → 3) Validasi data → 4) Training model → 5) Cek model gate. Setiap step harus lolos sebelum lanjut ke step berikutnya.

✓ **Checkpoint**: tab Actions di GitHub menampilkan run dengan status hijau (✅); semua step tercentang di log.

#### Skenario Kegagalan yang Disengaja (bagian dari latihan)

1. Ubah threshold di `check_model_gate.py` di dalam workflow menjadi angka mustahil (mis. `0.99`), untuk membuktikan CI benar-benar menjaga gerbang.
2. Commit dan push ke GitHub.
3. Buka tab Actions lagi — harus muncul run baru dengan status merah ❌. Klik untuk membuka detail, lihat step "Cek model gate" — pesan error harus terlihat jelas menyebutkan model tidak lolos.
4. Perbaiki kembali ke nilai realistis (mis. `0.70`), commit, push, pastikan kembali hijau.

#### Troubleshooting Langkah 5

| Gejala | Penyebab Umum | Solusi |
|---|---|---|
| `git push` minta username/password terus | Belum setup token/SSH key | Gunakan Personal Access Token GitHub sebagai password, atau setup SSH key |
| Tab Actions tidak menampilkan run apa pun | File workflow salah lokasi/nama folder | Pastikan path persis `.github/workflows/ml-ci-cd.yml` (bukan `.github/workflow/` tanpa "s") |
| CI gagal di step `pip install` | `requirements.txt` tidak lengkap/ada typo | Bandingkan dengan `requirements.txt` di paket ini |

### Langkah 6 — Monitoring Data Drift dengan Evidently AI (20')

Model sudah deploy bukan akhir cerita — dunia nyata terus berubah.

File: `scripts/check_drift.py`.

```bash
pip install evidently
python scripts/check_drift.py
```

Hasil tersimpan sebagai `drift_report.html` di folder project — buka file tersebut di browser.

Cari ringkasan drift per kolom — minimal 1 kolom (biasanya `income`) ditandai drift.

**Kenapa `income`?** `production_simulasi.csv` sengaja dibuat dengan distribusi `income` yang bergeser (lebih tinggi), agar drift benar-benar terdeteksi dan terlihat nyata di laporan.

#### Troubleshooting Langkah 6

| Gejala | Penyebab Umum | Solusi |
|---|---|---|
| `evidently` gagal import setelah install | venv tidak aktif saat install | Pastikan `(venv)` muncul sebelum `pip install evidently` |
| `drift_report.html` kosong/blank | Kolom training vs production tidak sama persis | Samakan kolom `production_simulasi.csv` dengan `train.csv` |

**Quiz**: Apa yang dimaksud dengan data drift, dan mengapa perlu dipantau terus-menerus meski model sudah deploy? → *Perubahan distribusi data dunia nyata dari waktu ke waktu; performa model bisa menurun walau kodenya tidak berubah.*

---

## Ringkasan (Checklist Akhir Sesi Siang)

- [ ] Repo Git + DVC berisi minimal 2 versi data yang tercatat
- [ ] Minimal 4 run tercatat di MLflow (1 dari Sesi 1 + 3 dari Sesi 2)
- [ ] `validate_data.py` pernah terbukti gagal dan berhasil
- [ ] `check_model_gate.py` pernah terbukti gagal dan berhasil
- [ ] Container Docker berjalan dan `/predict` mengembalikan hasil
- [ ] GitHub Actions pernah menampilkan status merah dan hijau
- [ ] `drift_report.html` berhasil dibuat dan minimal 1 kolom drift teridentifikasi

Jika semua tercentang — pipeline MLOps end-to-end hari ini berfungsi penuh: dari data mentah sampai model yang bisa diakses lewat API, teruji otomatis, dan termonitor.

---

## Catatan Kompatibilitas Versi Library (penting untuk fasilitator)

Semua script dalam paket ini sudah **diuji berjalan end-to-end** dengan versi library terbaru per September 2026. Beberapa API library di ekosistem MLOps berubah cukup cepat — jika peserta meng-install versi yang berbeda dari yang sudah diuji, siapkan catatan berikut:

- **MLflow**: beberapa versi terbaru menolak `mlflow.sklearn.log_model()` dengan format default (`skops`) karena alasan keamanan. `train.py` di paket ini sudah menggunakan `serialization_format="pickle"` untuk menghindari error `UntrustedTypesFoundException`. Jika peserta memakai versi MLflow lama, parameter ini tetap valid dan aman diabaikan.
- **Great Expectations**: `validate_data.py` ditulis untuk API Fluent versi ≥1.0 (`context.data_sources.add_pandas(...)`). Jika ter-install versi <1.0 (API lama `context.sources.pandas_default`), script perlu disesuaikan — cek dengan `python -c "import great_expectations as gx; print(gx.__version__)"`.
- **Evidently**: `check_drift.py` ditulis untuk API versi ≥0.7 (`from evidently import Report, Dataset` + `from evidently.presets import DataDriftPreset`). Versi <0.5 memakai API lama (`from evidently.report import Report`, `from evidently.metric_preset import DataDriftPreset`) — sudah tidak kompatibel dengan versi terbaru.
- **Path artifact model MLflow**: struktur folder `mlruns/` bisa berbeda antar versi MLflow (2.x vs 3.x). Instruksikan peserta untuk **selalu menyalin path model langsung dari MLflow UI** (tab Artifacts pada run terpilih) daripada menebak path secara manual.

Rekomendasi: sebelum hari-H, jalankan `pip install -r requirements.txt` di environment bersih dan jalankan seluruh urutan langkah sekali untuk memastikan versi yang ter-install kompatibel dengan environment training/lab komputer yang dipakai peserta.
