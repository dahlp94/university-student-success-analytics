import pandas as pd

from src.transform import transform


def test_transform_normalizes_column_names():
    frame = pd.DataFrame({"Unit ID": [1], "Institution Name": ["Example"]})
    result = transform(frame)
    assert list(result.columns) == ["unit_id", "institution_name"]
    assert result.loc[0, "institution_name"] == "Example"
