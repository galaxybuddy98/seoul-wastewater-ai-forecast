import pandas as pd
import pytest

from src.preprocessing.schema import (
    SchemaError,
    normalize_flow_columns,
    normalize_quality_columns,
)


def test_quality_schema_preserves_cod_toc_semantics_and_maps_tp():
    source = pd.DataFrame(
        {
            "측정일자": ["2021/01/01"],
            "물재생센터명칭": ["중랑물재생센터"],
            "처리장구분": ["제1처리장"],
            "계통구분": ["유입수"],
            "BOD": [100.0],
            "TOC(2021년 1월 이전자료는 COD)": [50.0],
            "SS": [80.0],
            "T-N": [30.0],
            "TP": [4.0],
            "PH": [7.0],
            "총대장군균": [10],
        }
    )

    normalized = normalize_quality_columns(source)

    assert normalized.loc[0, "tp_mg_l"] == 4.0
    assert normalized.loc[0, "toc_or_cod_value"] == 50.0
    assert normalized.loc[0, "toc_or_cod_label"] == "TOC"
    assert normalized.loc[0, "source_tp_column"] == "TP"


def test_quality_schema_rejects_missing_required_column():
    with pytest.raises(SchemaError, match="계통구분"):
        normalize_quality_columns(pd.DataFrame({"측정일자": ["2021/01/01"]}))


def test_flow_schema_keeps_primary_and_secondary_treatment_separate():
    source = pd.DataFrame(
        {
            "측정일자": ["2026/01/01"],
            "물재생센터명칭": ["서남물재생센터"],
            "처리장구분": ["시설현대화"],
            "1차하수처리량": [0.0],
            "2차하수처리량": [290053.0],
        }
    )

    normalized = normalize_flow_columns(source)

    assert normalized.loc[0, "primary_treatment_volume"] == 0.0
    assert normalized.loc[0, "secondary_treatment_volume"] == 290053.0
    assert "influent_flow" not in normalized.columns
