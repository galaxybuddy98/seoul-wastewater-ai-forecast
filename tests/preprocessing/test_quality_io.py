import pandas as pd

from src.preprocessing.quality import read_source_csv, write_quality_outputs


def test_cp949_source_reader_preserves_korean_columns(tmp_path):
    path = tmp_path / "수질.csv"
    pd.DataFrame({"측정일자": ["2024/01/01"], "물재생센터명칭": ["중랑물재생센터"]}).to_csv(
        path, index=False, encoding="cp949"
    )
    loaded = read_source_csv(path)
    assert loaded.columns.tolist() == ["측정일자", "물재생센터명칭"]
    assert loaded.loc[0, "물재생센터명칭"] == "중랑물재생센터"


def test_quality_outputs_write_csv_and_korean_markdown(tmp_path):
    summary = pd.DataFrame({"facility": ["중랑"], "variable": ["bod_mg_l"], "missing_count": [1]})
    candidates = pd.DataFrame({"variable": ["bod_mg_l"], "original_value": [1000.0]})
    paths = write_quality_outputs(summary, candidates, tmp_path, source_label="수질현황")
    assert paths["summary_csv"].exists()
    assert paths["outliers_csv"].exists()
    report = paths["report_md"].read_text(encoding="utf-8")
    assert "실제 원자료 기반" in report
    assert "수질현황" in report
