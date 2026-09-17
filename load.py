import json
import os
import re
import sys
from typing import Any, Dict, List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from db import init_db, upsert_faqs

load_dotenv()

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "Corpus_FAQs_Parachute_SA_2026.txt")
MODEL_NAME = "all-MiniLM-L6-v2"


def parse_corpus(file_path: str) -> List[Dict[str, Any]]:
    """Lee y parsea el archivo de FAQs estructurado."""
    if not os.path.exists(file_path):
        print(f"Error: no se encontró el archivo de corpus en '{file_path}'.")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("------------------------------------------------------------")
    faqs = []

    for block in blocks:
        block = block.strip()
        if not block or "ID:" not in block:
            continue

        id_match = re.search(r"^ID:\s*(FAQ-\d+)", block, re.MULTILINE)
        cat_match = re.search(r"^CATEGOR[IÍ]A:\s*(.+)$", block, re.MULTILINE)
        preg_match = re.search(r"^PREGUNTA:\s*(.+)$", block, re.MULTILINE)
        resp_match = re.search(r"^RESPUESTA:\s*([\s\S]+?)(?=\nMETADATA:|$)", block, re.MULTILINE)
        meta_match = re.search(r"^METADATA:\s*(\{.*\})", block, re.MULTILINE)

        if not (id_match and preg_match and resp_match):
            continue

        faq_id = id_match.group(1).strip()
        category = cat_match.group(1).strip() if cat_match else "General"
        question = preg_match.group(1).strip()
        answer = resp_match.group(1).strip()

        metadata = {}
        if meta_match:
            try:
                metadata = json.loads(meta_match.group(1).strip())
            except json.JSONDecodeError:
                metadata = {}

        faqs.append({
            "faq_id": faq_id,
            "category": category,
            "question": question,
            "answer": answer,
            "metadata": metadata,
        })

    return faqs


def main():
    print(f"--- Iniciando Carga del Corpus de FAQs para Parachute S.A. ---")
    print(f"1. Parseando archivo: {CORPUS_PATH}")
    faqs = parse_corpus(CORPUS_PATH)
    print(f"   Se encontraron {len(faqs)} FAQs estructuradas.")

    if not faqs:
        print("Error: no se extrajeron FAQs. Verifique el formato del archivo.")
        sys.exit(1)

    print(f"2. Cargando modelo de embeddings: '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"3. Generando embeddings para las {len(faqs)} FAQs...")
    # Creamos un texto rico combinando pregunta y respuesta para maximizar la cobertura semántica
    texts_to_embed = [
        f"Pregunta: {f['question']} Respuesta: {f['answer']}"
        for f in faqs
    ]
    embeddings = model.encode(texts_to_embed, show_progress_bar=True, normalize_embeddings=True)

    for i, emb in enumerate(embeddings):
        faqs[i]["embedding"] = emb.tolist()

    print(f"4. Inicializando base de datos PostgreSQL y extensión pgvector...")
    try:
        init_db()
    except Exception as e:
        print(f"\n[ERROR] No se pudo conectar a la base de datos PostgreSQL:")
        print(f"       {e}")
        print("\nAsegúrese de haber iniciado el contenedor con:")
        print("       docker compose up -d\n")
        sys.exit(1)

    print(f"5. Insertando FAQs y vectores en PostgreSQL...")
    total_inserted = upsert_faqs(faqs)
    print(f"\n✓ Carga completada exitosamente. Total de FAQs en base de datos: {total_inserted}")


if __name__ == "__main__":
    main()
