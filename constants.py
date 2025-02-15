import os
from chromadb.config import Settings  # Ensure correct import

CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="db",
    anonymized_telemetry=False
)
