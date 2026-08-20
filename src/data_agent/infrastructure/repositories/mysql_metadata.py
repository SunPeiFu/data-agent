from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from data_agent.domain.models import ExtractedEntities, TableIdentifier


class MetadataRepositoryError(RuntimeError):
    """Raised when the metadata repository cannot be queried.

    原子职责：
    把底层 MySQL 连接失败、依赖缺失、SQL 执行失败统一包装成 repository 层异常，
    Planner 捕获后可以降级到 mock fallback，而不是让整个 Agent 流程崩掉。
    """


@dataclass(frozen=True)
class MetadataCandidate:
    """A resolved table candidate from metadata storage.

    原子职责：
    承载从 meta_table 查询出来的一张候选表。Planner 不直接使用 MySQL row，
    而是使用这个领域对象，方便后续替换 TiDB/DataHub/OpenMetadata 时保持上层稳定。
    """

    full_table_name: str
    catalog_name: str | None
    db_name: str
    table_name: str
    table_comment: str | None
    biz_line: str | None
    domain: str | None
    data_layer: str | None
    owner: str | None
    score: int = 0

    @classmethod
    def from_row(cls, row: dict[str, Any], entities: ExtractedEntities) -> "MetadataCandidate":
        """Convert a MySQL row into a MetadataCandidate.

        原子职责：
        把数据库字段映射成代码里的候选对象，并在进入 Planner 前计算 context score。
        score 用于候选排序，优先展示更符合用户业务线、主题域、数仓分层的表。
        """
        return cls(
            full_table_name=row["full_table_name"],
            catalog_name=row.get("catalog_name"),
            db_name=row["db_name"],
            table_name=row["table_name"],
            table_comment=row.get("table_comment"),
            biz_line=row.get("biz_line"),
            domain=row.get("domain"),
            data_layer=row.get("data_layer"),
            owner=row.get("owner"),
            score=_context_score(row, entities),
        )

    def profile(self) -> dict[str, str | None]:
        """Expose table profile for post slot validation.

        原子职责：
        post_validate_slots 不关心完整数据库 row，只需要 domain/data_layer/biz_line
        等表画像来做跨槽位一致性校验。
        """
        return {
            "catalog_name": self.catalog_name,
            "db_name": self.db_name,
            "table_name": self.table_name,
            "table_comment": self.table_comment,
            "biz_line": self.biz_line,
            "domain": self.domain,
            "data_layer": self.data_layer,
            "owner": self.owner,
        }


class MySQLMetadataRepository:
    """Query table-level metadata from MySQL.

    当前版本只依赖两张表：
    - meta_table: 真实表资产。
    - meta_table_ext: 表级业务术语和别名。
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        """Initialize MySQL connection settings from args or environment variables.

        原子职责：
        让本地 demo 和生产部署都可以通过环境变量切换元数据源地址，不把账号密码写死在代码调用处。
        """
        self.host = host or os.getenv("DATA_AGENT_MYSQL_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("DATA_AGENT_MYSQL_PORT", "3306"))
        self.user = user or os.getenv("DATA_AGENT_MYSQL_USER", "root")
        self.password = password or os.getenv("DATA_AGENT_MYSQL_PASSWORD", "agent123")
        self.database = database or os.getenv("DATA_AGENT_MYSQL_DATABASE", "data_agent")

    def find_by_table_identifier(
        self,
        table: TableIdentifier,
        entities: ExtractedEntities,
    ) -> list[MetadataCandidate]:
        """Resolve one/two/three-part table names against meta_table.

        原子职责：
        处理用户已经给出技术表名的情况：
        - 三段式 catalog.db.table 走 full_table_name 或 catalog/db/table 精确匹配。
        - 两段式 db.table 走 full_table_name 或 db/table 精确匹配。
        - 一段式 table 只按 table_name 找候选，通常会产生多候选。
        """
        if table.parts_count >= 3:
            sql = """
                SELECT * FROM meta_table
                WHERE lifecycle_status = 'online'
                  AND (
                    full_table_name = %s
                    OR (catalog_name = %s AND db_name = %s AND table_name = %s)
                  )
            """
            params = (table.raw, table.catalog, table.schema_name, table.table_name)
        elif table.parts_count == 2:
            sql = """
                SELECT * FROM meta_table
                WHERE lifecycle_status = 'online'
                  AND (
                    full_table_name = %s
                    OR (db_name = %s AND table_name = %s)
                  )
            """
            params = (table.raw, table.schema_name, table.table_name)
        else:
            sql = """
                SELECT * FROM meta_table
                WHERE lifecycle_status = 'online'
                  AND table_name = %s
            """
            params = (table.table_name,)
        return self._fetch_candidates(sql, params, entities)

    def find_by_table_terms(
        self,
        terms: list[str],
        entities: ExtractedEntities,
    ) -> list[MetadataCandidate]:
        """Resolve table-level business terms through meta_table_ext.

        原子职责：
        处理用户没有说物理表名、只说业务叫法的情况，比如“订单信息表”。
        meta_table_ext 同时支持 term_value 和 normalized_term，所以可以用原词和标准词一起查。
        """
        unique_terms = _unique([term for term in terms if term])
        if not unique_terms:
            return []
        placeholders = ",".join(["%s"] * len(unique_terms))
        sql = f"""
            SELECT DISTINCT t.*
            FROM meta_table t
            JOIN meta_table_ext e ON e.table_id = t.id
            WHERE t.lifecycle_status = 'online'
              AND (
                e.normalized_term IN ({placeholders})
                OR e.term_value IN ({placeholders})
              )
        """
        return self._fetch_candidates(sql, (*unique_terms, *unique_terms), entities)

    def find_by_full_table_names(
        self,
        table_names: list[str],
        entities: ExtractedEntities,
    ) -> list[MetadataCandidate]:
        """Validate externally recalled table names against the metadata source of truth.

        原子职责：
        Milvus 只负责召回候选，不负责证明资产真实存在。该方法把向量召回结果重新交给
        meta_table 校验，并补齐主题域、分层、负责人等权威表画像。
        """
        unique_names = _unique([name for name in table_names if name])
        if not unique_names:
            return []
        placeholders = ",".join(["%s"] * len(unique_names))
        sql = f"""
            SELECT * FROM meta_table
            WHERE lifecycle_status = 'online'
              AND full_table_name IN ({placeholders})
        """
        return self._fetch_candidates(sql, tuple(unique_names), entities)

    def _fetch_candidates(
        self,
        sql: str,
        params: tuple[Any, ...],
        entities: ExtractedEntities,
    ) -> list[MetadataCandidate]:
        """Execute SQL and convert rows into sorted MetadataCandidate objects.

        原子职责：
        统一 MySQL 连接、执行、关闭和异常包装。所有查询方法都走这里，避免连接参数、
        超时、cursor 类型、排序规则散落在多个方法里。
        """
        try:
            import pymysql
            from pymysql.cursors import DictCursor
        except ModuleNotFoundError as exc:
            raise MetadataRepositoryError("PyMySQL 未安装，无法查询 MySQL 元数据。") from exc

        try:
            connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset="utf8mb4",
                cursorclass=DictCursor,
                connect_timeout=3,
                read_timeout=5,
                write_timeout=5,
            )
        except Exception as exc:
            raise MetadataRepositoryError(f"MySQL 元数据连接失败: {exc}") from exc

        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
        except Exception as exc:
            raise MetadataRepositoryError(f"MySQL 元数据查询失败: {exc}") from exc
        finally:
            connection.close()

        candidates = [MetadataCandidate.from_row(row, entities) for row in rows]
        return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.full_table_name))


def _context_score(row: dict[str, Any], entities: ExtractedEntities) -> int:
    """Score a candidate by how well it matches user context.

    原子职责：
    候选表可能很多，生产中通常会基于业务线、主题域、分层、热度、权限等排序。
    当前先用 biz_line/domain/data_layer 做最小可解释评分。
    """
    score = 0
    if entities.biz_line and row.get("biz_line") == entities.biz_line:
        score += 3
    if entities.domain and row.get("domain") == entities.domain.value:
        score += 2
    if entities.data_layer and row.get("data_layer") == entities.data_layer.value:
        score += 2
    return score


def _unique(values: list[str]) -> list[str]:
    """Deduplicate values while preserving original order.

    原子职责：
    MySQL IN 查询不需要重复 term；保序可以让 notes 和调试输出更贴近原始识别顺序。
    """
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
