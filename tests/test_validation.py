from src.validation.validate import run_checks


def test_validation_passes_on_valid_data(sample_raw_df):
    # all checks should pass on a properly formed dataset
    results = run_checks(sample_raw_df)
    fails = [r for r in results if r["status"] == "fail"]
    assert len(fails) == 0, f"unexpected failures: {fails}"


def test_validation_fails_on_missing_column(sample_raw_df):
    # dropping a required column should trigger a failure
    df = sample_raw_df.drop(columns=["cibil_score"])
    results = run_checks(df)
    col_check = next(r for r in results if r["name"] == "columns_present")
    assert col_check["status"] == "fail"


def test_validation_fails_on_bad_cibil_score(sample_raw_df):
    # cibil score outside 300-900 should fail
    df = sample_raw_df.copy()
    df.loc[0, "cibil_score"] = 100
    results = run_checks(df)
    cibil_check = next(r for r in results if r["name"] == "cibil_range")
    assert cibil_check["status"] == "fail"


def test_validation_fails_on_invalid_education(sample_raw_df):
    # education with unexpected values should fail
    df = sample_raw_df.copy()
    df.loc[0, "education"] = "PhD"
    results = run_checks(df)
    edu_check = next(r for r in results if r["name"] == "education_values")
    assert edu_check["status"] == "fail"
