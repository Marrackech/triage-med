# Analyse infrastructure — Entraînement et déploiement

## Entraînement (SFT + DPO)

| Métrique | SFT | DPO |
|---|---|---|
| GPU utilisé | Tesla T4 (Kaggle, gratuit) | Tesla T4 (Kaggle, gratuit) |
| Mémoire GPU utilisée | ~9.65 GB / 15.6 GB | ~9-10 GB / 15.6 GB (estimé, config identique) |
| Débit mesuré | ~0.03 steps/s (~1 step / 34s) | ~0.02-0.03 steps/s |
| Steps réalisés | 492 (3 epochs) | 84 (1 epoch, limité par troncature) |
| Temps de calcul total | ~17 017 s (≈ 4h44) | Non chronométré précisément — estimé ≈ 1h30-2h sur la base du débit |
| Coût financier | 0 € (Kaggle GPU gratuit, quota 30h/semaine) | 0 € (idem) |

## Déploiement (inférence)

| Métrique | Valeur |
|---|---|
| GPU utilisé | Tesla T4 (Kaggle) |
| Mémoire GPU (vLLM) | ~3.2 GB (poids du modèle fusionné, fp16) + KV-cache |
| Latence moyenne | 2.41 s / requête |
| Débit estimé | ~0.4 requête/s en séquentiel (non testé en charge concurrente) |
| Coût financier (test) | 0 € (Kaggle GPU gratuit) |

## Estimation de coûts pour un déploiement cloud réel (hors POC)

Ces chiffres sont des estimations basées sur les tarifs publics indicatifs de GPU cloud équivalents à un Tesla T4 (à titre d'ordre de grandeur, tarifs à vérifier au moment du déploiement réel) :

| Poste | Estimation |
|---|---|
| GPU T4 à la demande | ≈ 0.35-0.50 $/heure selon fournisseur |
| Entraînement complet (SFT + DPO, ~6h GPU) | ≈ 2-3 $ |
| Déploiement continu (24/7, 1 mois) | ≈ 250-360 $/mois |
| Alternative : instance à la demande / auto-scaling | Coût réduit si usage non continu (ex. heures d'ouverture des urgences) |

## Limites de cette analyse
- Débit et latence mesurés sur un unique GPU partagé (Kaggle), non représentatif d'une infrastructure de production dédiée.
- Aucun test de charge (requêtes concurrentes) n'a été réalisé — à prévoir avant tout déploiement pilote réel.
- Estimation de coûts non contractuelle, à affiner selon le fournisseur cloud retenu (voir rapport technique, section Roadmap).
