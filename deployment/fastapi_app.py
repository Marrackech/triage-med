from fastapi import FastAPI, HTTPException, Depends
from security import verify_api_key
from pydantic import BaseModel
import requests
import json
import uuid
from datetime import datetime, timezone
import os
import time

app = FastAPI(title="CHSA - Agent de Triage Médical (POC)")

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/completions")
MODEL_NAME = os.environ.get("MODEL_NAME", "UserMarrakech/qwen3-triage-final")
LOG_FILE = os.environ.get("LOG_FILE", "./traceability_log.jsonl")


class TriageRequest(BaseModel):
    contexte_patient: str
    question: str


class TriageResponse(BaseModel):
    interaction_id: str
    timestamp: str
    reponse: str
    latence_secondes: float


def log_interaction(interaction_id, latence):
    """Journalisation minimale : aucune donnée médicale sensible."""
    entry = {
        "interaction_id": interaction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latence_secondes": latence,
        "modele": MODEL_NAME,
        "statut": "success",
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@app.post("/triage", response_model=TriageResponse, dependencies=[Depends(verify_api_key)])
def evaluer_triage(request: TriageRequest):
    interaction_id = str(uuid.uuid4())

    prompt = f"""### Cas clinique :
{request.contexte_patient}

### Question :
{request.question}

Réponds en commençant systématiquement par : [Niveau de priorité estimé : urgence_maximale / urgence_moderee / differee], puis justifie en une ou deux phrases en te basant UNIQUEMENT sur les éléments cités dans le cas clinique ci-dessus.

### Réponse :
[Niveau de priorité estimé :"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.1,
        "repetition_penalty": 1.2,
    }

    debut = time.time()
    try:
        vllm_response = requests.post(VLLM_URL, json=payload, timeout=30)
        vllm_response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Erreur du moteur d'inférence : {e}")
    latence = time.time() - debut

    try:
        vllm_data = vllm_response.json()
        texte_genere = "[Niveau de priorité estimé :" + vllm_data["choices"][0]["text"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="Réponse invalide du moteur d'inférence",
        )

    log_interaction(interaction_id, latence)

    return TriageResponse(
        interaction_id=interaction_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        reponse=texte_genere,
        latence_secondes=round(latence, 2),
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "modele": MODEL_NAME}