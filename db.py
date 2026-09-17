import json
import os
from typing import Any, Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "parachuteDB")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")


def get_connection():
    """Establece y retorna una conexión a la base de datos PostgreSQL con soporte para pgvector."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    register_vector(conn)
    return conn


def init_db() -> None:
    """Crea la extensión pgvector y la tabla faqs con su índice vectorial si no existen."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Habilitar la extensión vector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Crear la tabla de FAQs
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS faqs (
                    id SERIAL PRIMARY KEY,
                    faq_id VARCHAR(50) UNIQUE NOT NULL,
                    category VARCHAR(150),
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector(384)
                );
                """
            )

            # Crear índice HNSW para búsqueda por similitud de coseno
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS faqs_embedding_hnsw_idx 
                ON faqs USING hnsw (embedding vector_cosine_ops);
                """
            )
            conn.commit()
            print("Base de datos e índice vectorial inicializados con éxito.")
    finally:
        conn.close()


def upsert_faqs(faqs_data: List[Dict[str, Any]]) -> int:
    """Inserta o actualiza un lote de FAQs con sus respectivos embeddings en PostgreSQL."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            sql = """
                INSERT INTO faqs (faq_id, category, question, answer, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (faq_id) DO UPDATE SET
                    category = EXCLUDED.category,
                    question = EXCLUDED.question,
                    answer = EXCLUDED.answer,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding;
            """
            records = [
                (
                    f["faq_id"],
                    f.get("category", ""),
                    f["question"],
                    f["answer"],
                    json.dumps(f.get("metadata", {})),
                    f["embedding"],
                )
                for f in faqs_data
            ]
            cur.executemany(sql, records)
            conn.commit()
            return len(records)
    finally:
        conn.close()


def search_faqs(query_embedding: List[float], limit: int = 3) -> List[Dict[str, Any]]:
    """Busca en PostgreSQL las FAQs más cercanas semánticamente usando la distancia de coseno."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # embedding-query nos da la similitud de coseno entre 0 y 1
            sql = """
                SELECT 
                    faq_id,
                    category,
                    question,
                    answer,
                    metadata,
                    (1 - (embedding <=> %s::vector)) AS similarity
                FROM faqs
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            cur.execute(sql, (query_embedding, query_embedding, limit))
            results = cur.fetchall()
            return [dict(row) for row in results]
    finally:
        conn.close()
