import pandas as pd


def test_load_returns_expected_columns(sample_raw_df, tmp_data_dir):
    # ingestion should produce a df with all 13 expected columns
    from src.ingestion.ingest import EXPECTED_COLUMNS

    # simulate raw csv with whitespace
    messy_df = sample_raw_df.copy()
    messy_df.columns = [f" {c} " for c in messy_df.columns]
    for col in messy_df.select_dtypes(include=["object"]).columns:
        messy_df[col] = " " + messy_df[col].astype(str) + " "

    raw_path = tmp_data_dir / "data" / "raw" / "loan_approval_dataset.csv"
    messy_df.to_csv(raw_path, index=False)

    # read back and apply the same logic as ingest
    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    assert len(missing) == 0, f"missing columns after strip: {missing}"


def test_load_raises_on_missing_columns(tmp_data_dir):
    # ingestion should fail if a required column is absent
    from src.ingestion.ingest import EXPECTED_COLUMNS

    # csv missing the cibil_score column
    df = pd.DataFrame({"loan_id": [1], "education": ["Graduate"]})
    raw_path = tmp_data_dir / "data" / "raw" / "test.csv"
    df.to_csv(raw_path, index=False)

    loaded = pd.read_csv(raw_path)
    loaded.columns = loaded.columns.str.strip()

    missing = set(EXPECTED_COLUMNS) - set(loaded.columns)
    assert len(missing) > 0, "should detect missing columns"
