from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field

# 意图识别如何做 第一步 定义意图分类
class IntentType(str, Enum):
    METADATA_SEARCH = "metadata_search"
    LINEAGE_SEARCH = "lineage_search"
    IMPACT_ANALYSIS = "impact_analysis"
    UNKNOWN = "unknown"


class MetadataQueryMode(str, Enum):
    """Distinguish set-valued asset discovery from one-table metadata lookup."""

    DISCOVERY = "discovery"
    DETAIL = "detail"

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


class NormalizedTermType(str, Enum):
    BUSINESS_TERM = "business_term"
    METRIC = "metric"
    ENTITY = "entity"
    TABLE_TERM = "table_term"


class NormalizedTerm(BaseModel):
    text: str
    canonical: str
    term_type: NormalizedTermType
    source: str
    confidence: float = Field(ge=0.0, le=1.0)


class NormalizationTrace(BaseModel):
    field_name: str
    before: Any
    after: Any
    rule: str
    source: str


class SlotIssueType(str, Enum):
    MISSING = "missing"
    AMBIGUOUS = "ambiguous" # 模糊的
    INVALID = "invalid" # 无效
    CONFLICT = "conflict" # 冲突
    FORBIDDEN = "forbidden" # 拒绝
    LOW_CONFIDENCE = "low_confidence" #低置信度


class SlotValidationStage(str, Enum):
    PRE_METADATA = "pre_metadata"
    POST_METADATA = "post_metadata"


class MetadataCandidateSource(str, Enum):
    """Describe which retrieval path produced a table candidate."""

    MYSQL_IDENTIFIER = "mysql_identifier"
    MYSQL_TABLE_TERM = "mysql_table_term"
    MILVUS_MYSQL_VALIDATED = "milvus_mysql_validated"
    MOCK_FALLBACK = "mock_fallback"


class MetadataValidationStatus(str, Enum):
    """Separate authoritative candidates from degraded or unverified results."""

    VALIDATED = "validated"
    UNVERIFIED = "unverified"
    FALLBACK = "fallback"


class MetadataCandidateEvidence(BaseModel):
    """Auditable evidence used by post slot validation to gate tool execution."""

    full_table_name: str
    source: MetadataCandidateSource
    validation_status: MetadataValidationStatus
    score: float | None = None
    rank: int | None = None
    score_gap_to_next: float | None = None
    retrieval_mode: str | None = None


class AccessContext(BaseModel):
    """Authenticated subject context passed into the planning workflow."""

    user_id: str
    roles: list[str] = Field(default_factory=list)
    tenant_id: str | None = None
    allowed_domains: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)


class AuthorizationDecision(BaseModel):
    """Auditable authorization result for one action-resource pair."""

    allowed: bool
    action: str
    resource: str | None = None
    policy_id: str | None = None
    reason_code: str
    audit_id: str


class AgentRunStatus(str, Enum):
    """Lifecycle status for one initial, resume, retry, or replan execution."""

    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FORBIDDEN = "forbidden"
    HANDOFF = "handoff"
    FAILED = "failed"
    REPLAN_REQUIRED = "replan_required"
    VALIDATION_FAILED = "validation_failed"
    APPROVAL_REQUIRED = "approval_required"


class NodeRunStatus(str, Enum):
    """Execution status for one traced LangGraph business node."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TraceContext(BaseModel):
    """Identifiers shared by every span in one Agent execution segment."""

    trace_id: str
    run_id: str
    thread_id: str
    parent_run_id: str | None = None
    planner_version: str
    started_at: datetime


class NodeTrace(BaseModel):
    """Structured span for one LangGraph business node execution."""

    node_run_id: str
    trace_id: str
    run_id: str
    node_name: str
    status: NodeRunStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: float | None = None
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None


class TraceEvent(BaseModel):
    """Structured business decision event attached to a trace and node."""

    event_id: str
    trace_id: str
    run_id: str
    node_name: str
    event_type: str
    reason_code: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class SlotIssue(BaseModel):
    slot_name: str
    issue_type: SlotIssueType
    message: str
    blocking: bool = True


class ClarificationInputType(str, Enum):
    """Frontend control used to collect one clarification answer."""

    SINGLE_SELECT = "single_select"
    TEXT = "text"
    CONFIRM = "confirm"


class ClarificationOption(BaseModel):
    """One authorized option displayed in a clarification card."""

    option_id: str
    label: str
    value: str
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClarificationRequest(BaseModel):
    """Structured human-in-the-loop payload returned to a UI or API client."""

    clarification_id: str
    thread_id: str
    question: str
    slot_name: str
    issue_type: SlotIssueType
    input_type: ClarificationInputType
    options: list[ClarificationOption] = Field(default_factory=list)
    required: bool = True
    allow_custom_value: bool = False
    pending_issue_count: int = 0
    state_version: int = 1
    clarification_round: int = 1
    max_clarification_rounds: int = 3


class ClarificationResponse(BaseModel):
    """Versioned client response reserved for the future LangGraph resume endpoint."""

    clarification_id: str
    thread_id: str
    value: str = Field(min_length=1)
    option_id: str | None = None
    state_version: int = 1
    idempotency_key: str = Field(min_length=1)


class ClarificationAnswerRecord(BaseModel):
    """Auditable user-confirmed slot value retained across clarification rounds."""

    slot_name: str
    value: str
    source: str = "user_confirmed"
    confidence: float = 1.0
    clarification_round: int
    idempotency_key: str


class SlotValidationResult(BaseModel):
    stage: SlotValidationStage
    passed: bool = True
    issues: list[SlotIssue] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def blocking_issues(self) -> list[SlotIssue]:
        return [issue for issue in self.issues if issue.blocking]

    @property
    def needs_clarification(self) -> bool:
        return any(issue.issue_type != SlotIssueType.FORBIDDEN for issue in self.blocking_issues)

    @property
    def forbidden(self) -> bool:
        return any(issue.issue_type == SlotIssueType.FORBIDDEN for issue in self.blocking_issues)


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
    input_bindings: dict[str, int] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)
    parallel_group: str | None = None


class ToolCallStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    FORBIDDEN = "forbidden"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class TaskExecutionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class PlanValidationStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REPLAN_REQUIRED = "replan_required"
    CLARIFICATION_REQUIRED = "clarification_required"
    APPROVAL_REQUIRED = "approval_required"
    FORBIDDEN = "forbidden"


class ViolationSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PlanViolation(BaseModel):
    code: str
    severity: ViolationSeverity
    validator: str
    message: str
    suggested_status: PlanValidationStatus
    step_id: int | None = None
    field: str | None = None
    retryable: bool = False


class PlanValidationResult(BaseModel):
    status: PlanValidationStatus
    passed: bool
    violations: list[PlanViolation] = Field(default_factory=list)
    warnings: list[PlanViolation] = Field(default_factory=list)
    plan_hash: str
    registry_version: str
    policy_version: str


class ToolObservation(BaseModel):
    tool_call_id: str
    run_id: str | None = None
    step_id: int
    tool_name: str
    action: str
    status: ToolCallStatus
    attempts: int = 0
    duration_ms: float = 0.0
    output: Any = None
    error_code: str | None = None
    error_message: str | None = None


class TaskExecutionResult(BaseModel):
    status: TaskExecutionStatus
    observations: list[ToolObservation] = Field(default_factory=list)
    terminal_outputs: dict[int, Any] = Field(default_factory=dict)


class PlanningResult(BaseModel):
    question: str
    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    entities: ExtractedEntities
    metadata_query_mode: MetadataQueryMode | None = None
    task_steps: list[TaskStep] = Field(default_factory=list)
    need_clarification: bool = False
    clarification_question: str | None = None
    clarification_request: ClarificationRequest | None = None
    pending_clarification_issues: list[SlotIssue] = Field(default_factory=list)
    clarification_history: list[ClarificationAnswerRecord] = Field(default_factory=list)
    handoff_required: bool = False
    handoff_reason: str | None = None
    thread_id: str | None = None
    trace_id: str | None = None
    run_id: str | None = None
    parent_run_id: str | None = None
    notes: list[str] = Field(default_factory=list)
    normalized_terms: list[NormalizedTerm] = Field(default_factory=list)
    normalization_traces: list[NormalizationTrace] = Field(default_factory=list)
    execution_status: TaskExecutionStatus | None = None
    tool_observations: list[ToolObservation] = Field(default_factory=list)
    final_output: Any = None
    plan_validation: PlanValidationResult | None = None
    replan_required: bool = False
    approval_required: bool = False
