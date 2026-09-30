"""
MODUL 4 - LAB SESI 2 - Langkah 2: Data Quality Gate
Jalankan: python validate_data.py
LOLOS -> exit code 0, pipeline lanjut ke training
GAGAL -> exit code 1, pipeline dihentikan

Latihan di kelas: ubah 1 nilai age di data/train.csv menjadi 200, jalankan ulang
(harus GAGAL), lalu perbaiki kembali dan jalankan ulang (harus LOLOS).

Catatan versi: script ini ditulis untuk great_expectations >= 1.0 (Fluent API).
Cek versi terpasang dengan: python -c "import great_expectations as gx; print(gx.__version__)"
"""
import sys
import pandas as pd
import great_expectations as gx
import great_expectations.expectations as gxe

DATA_PATH = "data/train.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    context = gx.get_context()
    data_source = context.data_sources.add_pandas("pandas_src")
    data_asset = data_source.add_dataframe_asset(name="train_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch_def")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name="train_suite")
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="income"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="age"))
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="age", min_value=18, max_value=100)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="target", value_set=[0, 1])
    )

    results = batch.validate(suite)

    if not results.success:
        print("GAGAL: data tidak valid — cek hasil expectation di atas.")
        sys.exit(1)

    print("LOLOS: data valid")

if __name__ == "__main__":
    main()
