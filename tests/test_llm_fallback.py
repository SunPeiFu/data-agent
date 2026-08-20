from data_agent.intelligence.extractor import extract_entities
from data_agent.domain.models import DataLayer, DomainType


def test_extractor_falls_back_without_llm_model(monkeypatch) -> None:
    monkeypatch.delenv("DATA_AGENT_LLM_MODEL", raising=False)

    entities = extract_entities("安逸花业务线营销域下DWD层关于支付的表有哪些")

    assert entities.biz_line == "安逸花"
    assert entities.domain == DomainType.MARKETING
    assert entities.data_layer == DataLayer.DWD
    assert "支付" in entities.topic_keywords
