"""
MODUL 4 - LAB SESI 2 - Langkah 3: Model Quality Gate
Jalankan dengan AUC terbaik dari Langkah 1, contoh:
    python check_model_gate.py 0.81
LOLOS -> exit code 0
GAGAL -> exit code 1  (cek dengan: echo $?  /  echo $LASTEXITCODE di PowerShell)

Latihan di kelas: jalankan juga dengan AUC rendah, contoh:
    python check_model_gate.py 0.60
"""
import sys

MINIMUM_AUC = 0.7

def check_gate(auc: float) -> bool:
    return auc >= MINIMUM_AUC

def main():
    if len(sys.argv) != 2:
        print("Cara pakai: python check_model_gate.py <auc>")
        sys.exit(1)

    auc = float(sys.argv[1])

    if check_gate(auc):
        print(f"LOLOS: Model lolos gate (AUC={auc} >= {MINIMUM_AUC})")
        sys.exit(0)
    else:
        print(f"GAGAL: Model tidak lolos gate (AUC={auc} < {MINIMUM_AUC})")
        sys.exit(1)

if __name__ == "__main__":
    main()
