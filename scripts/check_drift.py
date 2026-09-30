"""
MODUL 4 - LAB SESI 2 - Langkah 6: Monitoring Data Drift dengan Evidently AI
Jalankan: python check_drift.py
Hasil: drift_report.html di root project -> buka di browser.

production_simulasi.csv sengaja dibuat dengan distribusi income yang bergeser,
sehingga kolom "income" biasanya akan terdeteksi drift di laporan.

Catatan versi: script ini ditulis untuk evidently >= 0.7 (API baru: Report/Dataset
di top-level "evidently"). Cek versi terpasang dengan:
python -c "import evidently; print(evidently.__version__)"
Untuk evidently versi lama (0.4.x), lihat catatan di panduan.
"""
import pandas as pd
from evidently import Report, Dataset
from evidently.presets import DataDriftPreset

reference = pd.read_csv("data/train.csv").drop(columns=["target"])
current = pd.read_csv("data/production_simulasi.csv").drop(columns=["target"])

reference_dataset = Dataset.from_pandas(reference)
current_dataset = Dataset.from_pandas(current)

report = Report([DataDriftPreset()])
run = report.run(current_dataset, reference_dataset)
run.save_html("drift_report.html")

print("Laporan drift tersimpan sebagai drift_report.html")
