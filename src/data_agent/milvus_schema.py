from __future__ import annotations

import argparse
import os


DEFAULT_COLLECTION = "data_agent_table_assets_v1"


def create_table_asset_collection(drop_existing: bool = False) -> None:
    """Create the table-level hybrid retrieval Collection and its indexes."""
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    uri = os.getenv("DATA_AGENT_MILVUS_URI", "http://127.0.0.1:19531")
    collection_name = os.getenv("DATA_AGENT_MILVUS_COLLECTION", DEFAULT_COLLECTION)
    dimension = int(os.getenv("DATA_AGENT_EMBEDDING_DIM", "1024"))
    client = MilvusClient(uri=uri)

    if client.has_collection(collection_name):
        if not drop_existing:
            print(f"Collection {collection_name} already exists; no changes made.")
            return
        client.drop_collection(collection_name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("asset_key", DataType.VARCHAR, is_primary=True, max_length=512)
    schema.add_field("table_id", DataType.INT64)
    schema.add_field("full_table_name", DataType.VARCHAR, max_length=512)
    schema.add_field("catalog_name", DataType.VARCHAR, max_length=128)
    schema.add_field("db_name", DataType.VARCHAR, max_length=128)
    schema.add_field("table_name", DataType.VARCHAR, max_length=256)
    schema.add_field("biz_line", DataType.VARCHAR, max_length=128)
    schema.add_field("domain", DataType.VARCHAR, max_length=128)
    schema.add_field("data_layer", DataType.VARCHAR, max_length=32)
    schema.add_field("lifecycle_status", DataType.VARCHAR, max_length=32)
    schema.add_field(
        "searchable_text",
        DataType.VARCHAR,
        max_length=8192,
        enable_analyzer=True,
        enable_match=True,
        analyzer_params={"tokenizer": "jieba"},
    )
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("source_updated_at", DataType.INT64)
    schema.add_field("metadata", DataType.JSON)
    schema.add_function(
        Function(
            name="table_text_bm25",
            input_field_names=["searchable_text"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
    )

    indexes = client.prepare_index_params()
    indexes.add_index(
        field_name="dense_vector",
        index_name="dense_vector_idx",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    indexes.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_idx",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )
    for field_name in ["biz_line", "domain", "data_layer", "lifecycle_status"]:
        indexes.add_index(field_name=field_name, index_type="INVERTED")

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=indexes,
        consistency_level="Bounded",
    )
    print(f"Created Collection {collection_name} at {uri}, dense dimension={dimension}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the DataAgent Milvus Collection.")
    parser.add_argument("--drop-existing", action="store_true")
    args = parser.parse_args()
    create_table_asset_collection(drop_existing=args.drop_existing)


if __name__ == "__main__":
    main()
