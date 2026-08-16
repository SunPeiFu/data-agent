from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field

# 意图识别如何做 第一步 定义意图分类
class IntentType(str, Enum):
    METADATA_SEARCH = "metadata_search"
    LINEAGE_SEARCH = "lineage_search"
    IMPACT_ANALYSIS = "impact_analysis"
    UNKNOWN = "unknown"

# 定义枚举 血缘方向
class LineageDirection(str, Enum):
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"
    BOTH = "both"

# 定义枚举 业务域
class DomainType(str, Enum):
    MARKETING = "营销域"
    RISK = "风控域"
    TRANSACTION = "交易域"
    USER = "用户域"
    FINANCE = "财务域"

# 定义枚举层级
class DataLayer(str, Enum):
    ODS = "ODS"
    DWD = "DWD"
    DWS = "DWS"
    ADS = "ADS"
    DIM = "DIM"

# 定义操作枚举
class OperationType(str, Enum):
    MODIFY_FIELD = "modify_field"
    ADD_FIELD = "add_field"
    DELETE_FIELD = "delete_field"
    RENAME_FIELD = "rename_field"
    UNKNOWN_CHANGE = "unknown_change"


class TableIdentifier(BaseModel):
    raw: str # raw含义 原始输入字符串 没有任何更新更改
    catalog: str | None = None
    schema_name: str | None = None
    table_name: str

    @computed_field
    @property
    def parts_count(self) -> int:
        return len([part for part in [self.catalog, self.schema_name, self.table_name] if part])

    @computed_field
    @property
    def is_fully_qualified(self) -> bool:
        return self.parts_count >= 2

    @classmethod
    def parse(cls, raw_table: str) -> "TableIdentifier":
        cleaned = raw_table.strip(" ，,。；;：:")
        parts = [part for part in cleaned.split(".") if part]
        if len(parts) >= 3:
            return cls(raw=cleaned, catalog=parts[-3], schema_name=parts[-2], table_name=parts[-1])
        if len(parts) == 2:
            return cls(raw=cleaned, schema_name=parts[0], table_name=parts[1])
        return cls(raw=cleaned, table_name=parts[0])

# 抽取意图识别的实体
class ExtractedEntities(BaseModel):
    biz_line: str | None = None
    domain: DomainType | None = None
    data_layer: DataLayer | None = None
    table: TableIdentifier | None = None
    field_name: str | None = None
    operation: OperationType | None = None
    topic_keywords: list[str] = Field(default_factory=list)
    lineage_direction: LineageDirection | None = None


class TaskStep(BaseModel):
    step_id: int
    tool_name: str
    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)
    parallel_group: str | None = None


class PlanningResult(BaseModel):
    question: str
    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    entities: ExtractedEntities
    task_steps: list[TaskStep] = Field(default_factory=list)
    need_clarification: bool = False
    clarification_question: str | None = None
    notes: list[str] = Field(default_factory=list)
