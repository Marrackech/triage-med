import os
import secrets

from fastapi import Header, HTTPException, status


# La clé API doit être définie dans une variable d'environnement.
API_KEY = os.getenv("TRIAGE_API_KEY")


def verify_api_key(x_api_key: str = Header(default=None)):
    """
    Vérifie la clé API envoyée dans le header X-API-Key.
    """
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API authentication is not configured",
        )

    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    return True
