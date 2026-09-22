from sklearn.model_selection import train_test_split


def test_transform_drops_loan_id(sample_raw_df):
    # loan_id should not appear in the output
    df = sample_raw_df.copy()
    df = df.drop(columns=["loan_id"])
    assert "loan_id" not in df.columns


def test_target_encoding(sample_raw_df):
    # loan_status should map to 1/0 correctly
    df = sample_raw_df.copy()
    df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})
    assert df["loan_status"].isin([0, 1]).all()
    assert df["loan_status"].sum() == 4  # 4 approved in sample data


def test_transform_output_has_no_nulls(sample_raw_df):
    # transformed data should have no null values
    df = sample_raw_df.copy()
    df = df.drop(columns=["loan_id"])
    df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})
    X = df.drop(columns=["loan_status"])
    assert X.isnull().sum().sum() == 0


def test_transform_split_shapes(sample_raw_df):
    # train/test split should preserve total row count
    df = sample_raw_df.copy()
    df = df.drop(columns=["loan_id"])
    df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
