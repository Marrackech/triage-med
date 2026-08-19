from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import uuid
from datetime import datetime
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


def log_interaction(interaction_id, contexte, question, reponse, latence):
    """Trace chaque interaction pour audit médical, conformément à la mission."""
    entry = {
        "interaction_id": interaction_id,
        "timestamp": datetime.utcnow().isoformat(),
        "contexte_patient": contexte,
        "question": question,
        "reponse_generee": reponse,
        "latence_secondes": latence,
        "modele": MODEL_NAME,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@app.post("/triage", response_model=TriageResponse)
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

    texte_genere = "[Niveau de priorité estimé :" + vllm_response.json()["choices"][0]["text"]

    log_interaction(interaction_id, request.contexte_patient, request.question, texte_genere, latence)

    return TriageResponse(
        interaction_id=interaction_id,
        timestamp=datetime.utcnow().isoformat(),
        reponse=texte_genere,
        latence_secondes=round(latence, 2),
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "modele": MODEL_NAME}