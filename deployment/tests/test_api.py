import os
from unittest.mock import Mock, patch

os.environ["TRIAGE_API_KEY"] = "test-secret-key"
os.environ["MODEL_NAME"] = "test-model"

from fastapi.testclient import TestClient
from fastapi_app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_triage_requires_api_key():
    response = client.post(
        "/triage",
        json={
            "contexte_patient": "Patient présentant une douleur.",
            "question": "Quel est le niveau de priorité ?",
        },
    )

    assert response.status_code == 401


def test_triage_with_invalid_api_key():
    response = client.post(
        "/triage",
        headers={"X-API-Key": "wrong-key"},
        json={
            "contexte_patient": "Patient présentant une douleur.",
            "question": "Quel est le niveau de priorité ?",
        },
    )

    assert response.status_code == 401


@patch("fastapi_app.requests.post")
def test_triage_success(mock_post):
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [
            {
                "text": " urgence_moderee] Justification de test."
            }
        ]
    }

    mock_post.return_value = mock_response

    response = client.post(
        "/triage",
        headers={"X-API-Key": "test-secret-key"},
        json={
            "contexte_patient": "Patient présentant une douleur.",
            "question": "Quel est le niveau de priorité ?",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "interaction_id" in data
    assert "timestamp" in data
    assert "reponse" in data
    assert "latence_secondes" in data

    mock_post.assert_called_once()


@patch("fastapi_app.requests.post")
def test_triage_vllm_unavailable(mock_post):
    import requests

    mock_post.side_effect = requests.RequestException(
        "vLLM unavailable"
    )

    response = client.post(
        "/triage",
        headers={"X-API-Key": "test-secret-key"},
        json={
            "contexte_patient": "Patient présentant une douleur.",
            "question": "Quel est le niveau de priorité ?",
        },
    )

    assert response.status_code == 502
    assert "Erreur du moteur d'inférence" in response.json()["detail"]


@patch("fastapi_app.requests.post")
def test_triage_sends_correct_payload(mock_post):
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [
            {
                "text": " urgence_moderee] Justification de test."
            }
        ]
    }

    mock_post.return_value = mock_response

    response = client.post(
        "/triage",
        headers={"X-API-Key": "test-secret-key"},
        json={
            "contexte_patient": "Patient avec douleur thoracique.",
            "question": "Quel est le niveau de priorité ?",
        },
    )

    assert response.status_code == 200

    mock_post.assert_called_once()

    call_args = mock_post.call_args

    assert call_args.kwargs["json"]["model"] == "test-model"
    assert call_args.kwargs["json"]["max_tokens"] == 150
    assert call_args.kwargs["json"]["temperature"] == 0.1
    assert call_args.kwargs["json"]["repetition_penalty"] == 1.2
    assert call_args.kwargs["timeout"] == 30


@patch("fastapi_app.requests.post")
def test_triage_prompt_contains_patient_context_and_question(mock_post):
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [
            {
                "text": " urgence_moderee] Justification de test."
            }
        ]
    }

    mock_post.return_value = mock_response

    contexte = "Patient avec douleur thoracique depuis deux heures."
    question = "Quel est le niveau de priorité ?"

    response = client.post(
        "/triage",
        headers={"X-API-Key": "test-secret-key"},
        json={
            "contexte_patient": contexte,
            "question": question,
        },
    )

    assert response.status_code == 200

    payload = mock_post.call_args.kwargs["json"]
    prompt = payload["prompt"]

    assert contexte in prompt
    assert question in prompt
    assert "Niveau de priorité estimé" in prompt


def test_triage_missing_required_field():
    response = client.post(
        "/triage",
        headers={"X-API-Key": "test-secret-key"},
        json={
            "contexte_patient": "Patient avec douleur.",
        },
    )

    assert response.status_code == 422


@patch("fastapi_app.requests.post")
def test_log_does_not_contain_medical_data(mock_post, tmp_path, monkeypatch):
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [
            {
                "text": " urgence_moderee] Justification de test."
            }
        ]
    }

    mock_post.return_value = mock_response

    log_file = tmp_path / "traceability_log.jsonl"
    monkeypatch.setattr("fastapi_app.LOG_FILE", str(log_file))

    contexte = "DONNEE_MEDICALE_TEST_12345"
    question = "QUESTION_MEDICALE_TEST_67890"

    response = client.post(
        "/triage",
        headers={"X-API-Key": "test-secret-key"},
        json={
            "contexte_patient": contexte,
            "question": question,
        },
    )

    assert response.status_code == 200

    contenu_log = log_file.read_text()

    assert contexte not in contenu_log
    assert question not in contenu_log
    assert "DONNEE_MEDICALE_TEST_12345" not in contenu_log
    assert "QUESTION_MEDICALE_TEST_67890" not in contenu_log


@patch("fastapi_app.requests.post")
def test_triage_invalid_vllm_response(mock_post):
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {}

    mock_post.return_value = mock_response

    response = client.post(
        "/triage",
        headers={"X-API-Key": "test-secret-key"},
        json={
            "contexte_patient": "Patient avec douleur.",
            "question": "Quel est le niveau de priorité ?",
        },
    )

    assert response.status_code == 502


@patch("fastapi_app.requests.post")
def test_interaction_ids_are_unique(mock_post):
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [
            {
                "text": " urgence_moderee] Justification de test."
            }
        ]
    }

    mock_post.return_value = mock_response

    headers = {"X-API-Key": "test-secret-key"}
    payload = {
        "contexte_patient": "Patient avec douleur.",
        "question": "Quel est le niveau de priorité ?",
    }

    response_1 = client.post(
        "/triage",
        headers=headers,
        json=payload,
    )

    response_2 = client.post(
        "/triage",
        headers=headers,
        json=payload,
    )

    assert response_1.status_code == 200
    assert response_2.status_code == 200

    id_1 = response_1.json()["interaction_id"]
    id_2 = response_2.json()["interaction_id"]

    assert id_1 != id_2
