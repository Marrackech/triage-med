# 🏥 POC — Agent IA de Triage Médical

**Fine-tuning SFT + DPO de Qwen3-1.7B pour l'assistance au triage des urgences**

[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-success)](../../actions)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)

> **Verdict du POC : 🔴 NO-GO pour la production clinique / 🟢 GO pour une Phase 2 encadrée**

---

## -- Présentation

Ce projet est un **Proof of Concept (POC)** d'un agent IA destiné à assister le triage des urgences du **Centre Hospitalier Saint-Aurélien (CHSA)**.

L'objectif est d'étudier la faisabilité technique d'un système capable d'analyser des informations cliniques textuelles et de proposer un niveau de priorité parmi trois catégories expérimentales :

* 🔴 **Urgence maximale**
* 🟠 **Urgence modérée**
* 🟢 **Différée**

Le projet couvre l'ensemble de la chaîne AI Engineering / MLOps :

```text
Données médicales
        │
        ▼
Nettoyage & anonymisation
        │
        ▼
Annotation du signal de priorité
        │
        ▼
Dataset SFT
        │
        ▼
Qwen3-1.7B-Base
        │
        ├── SFT + LoRA
        │
        ▼
Modèle SFT
        │
        ├── DPO
        │
        ▼
Modèle aligné
        │
        ▼
Fusion des adaptateurs
        │
        ▼
Modèle final
        │
        ▼
vLLM
        │
        ▼
FastAPI
        │
        ▼
API /triage
        │
        ▼
Journalisation sécurisée
```

---

# ⚠️ Avertissement médical

Ce projet est un **prototype technique et pédagogique**.

Il ne constitue :

* ni un dispositif médical ;
* ni un outil de diagnostic ;
* ni un système autonome de décision médicale ;
* ni une solution destinée à être utilisée directement avec des patients ;
* ni un remplacement d'un professionnel de santé.

Les résultats du POC ont notamment mis en évidence :

* une tendance à la **sur-classification en urgence maximale** ;
* des **hallucinations factuelles** dans certaines justifications ;
* un volume d'annotations cliniques insuffisant ;
* l'absence de validation clinique formelle.

Le système est donc **NO-GO pour un usage clinique réel**.

---

# -- Objectifs --       🎯 

La mission consistait à développer un POC capable de démontrer la faisabilité technique d'un agent de triage médical pouvant, à terme :

1. recueillir des informations cliniques ;
2. analyser les symptômes décrits ;
3. proposer un niveau de priorité ;
4. fournir une justification textuelle ;
5. conserver une trace de chaque interaction ;
6. être exposé via une API ;
7. être servi par un moteur d'inférence optimisé ;
8. être conteneurisé ;
9. être testé automatiquement ;
10. être intégré dans une pipeline CI/CD.

Le POC ne cherche donc pas à démontrer une aptitude clinique suffisante pour une mise en production, mais à construire une **chaîne technique complète et reproductible**.

---

#  Architecture --

L'architecture finale sépare l'entraînement, l'inférence et la couche applicative.

```text
                       ┌──────────────────────┐
                       │       DONNÉES        │
                       │                      │
                       │ MediQAl              │
                       │ FrenchMedMCQA        │
                       │ MedQuAD              │
                       │ UltraMedical         │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Préparation données  │
                       │                      │
                       │ Nettoyage            │
                       │ Anonymisation        │
                       │ Annotation           │
                       │ Structuration       │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │      ENTRAÎNEMENT    │
                       │                      │
                       │ Qwen3-1.7B           │
                       │ SFT + LoRA           │
                       │ DPO                  │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │     MODÈLE FINAL     │
                       │                      │
                       │ Adaptateurs fusionnés│
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │        vLLM          │
                       │                      │
                       │ Inférence LLM        │
                       │ /v1/completions      │
                       └──────────┬───────────┘
                                  │ HTTP
                                  ▼
                       ┌──────────────────────┐
                       │       FastAPI        │
                       │                      │
                       │ /health              │
                       │ /triage              │
                       │ Authentification     │
                       │ Logs sécurisés       │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │     Client / UI      │
                       └──────────────────────┘
```

---

#  -- Structure du projet

L'organisation du dépôt sépare les données, l'entraînement, l'application, les tests et le déploiement.

```text
projetllm/
│
├── app/
│   ├── main.py
│   └── ...
│
├── data_raw/
│   └── ...
│
├── data_processed/
│   └── ...
│
├── data_annotated/
│   └── ...
│
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements-deployment.txt
│   └── ...
│
├── tests/
│   └── ...
│
├── notebooks/
│   ├── notebook_sft.ipynb
│   ├── notebook_dpo.ipynb
│   └── ...
│
├── scripts/
│   └── ...
│
├── README.md
├── go:no-go.txt
└── ...
```

> Les notebooks SFT/DPO, les résultats, les configurations d'entraînement et les éléments de déploiement sont désormais documentés et intégrés au projet.

---

# 📊 Données

## Sources

Le corpus utilisé pour le projet a été constitué à partir de quatre sources médicales ouvertes :

| Source                  | Contenu                                |     Volume exploité | Langue |
| ----------------------- | -------------------------------------- | ------------------: | ------ |
| MediQAl                 | Cas cliniques avec raisonnement        |        3 954 paires | FR     |
| FrenchMedMCQA           | QCM médicaux                           |        1 080 paires | FR     |
| MedQuAD                 | Questions-réponses médicales générales |          700 paires | EN     |
| UltraMedical-Preference | Paires de préférence                   | 800 SFT + 1 500 DPO | EN     |

Le dataset assemblé pour le SFT contient :

**6 534 paires instruction-réponse**

avec environ :

* **77 % de données françaises**
* **23 % de données anglaises**

---

#  -- Sources et licences

Les datasets sont des ressources externes et leurs conditions d'utilisation doivent être respectées indépendamment de la licence du code du projet.

| Dataset                 | Utilisation | Licence                                      |
| ----------------------- | ----------- | -------------------------------------------- |
| MediQAl                 | SFT         | Consulter la licence de la source officielle |
| FrenchMedMCQA           | SFT         | Consulter la licence de la source officielle |
| MedQuAD                 | SFT         | Consulter la licence de la source officielle |
| UltraMedical-Preference | SFT / DPO   | Consulter la licence de la source officielle |

Avant toute redistribution publique des données ou du modèle, il est nécessaire de vérifier :

* la licence exacte de chaque dataset ;
* les conditions de redistribution ;
* les éventuelles restrictions d'utilisation ;
* la licence du modèle Qwen ;
* les conditions applicables au modèle fine-tuné.

> Les données médicales ou potentiellement sensibles ne doivent pas être redistribuées sans vérification préalable de leur statut juridique et de leur licence.

---

#  -- Préparation des données

Le pipeline de données comprend :

1. chargement des sources ;
2. exploration ;
3. nettoyage ;
4. harmonisation ;
5. anonymisation ;
6. constitution du vivier de cas ;
7. annotation ;
8. contrôle des annotations ;
9. création du dataset final ;
10. séparation train / validation / test.

---

# 🔐 Anonymisation

L'anonymisation a été réalisée avec **Microsoft Presidio**, associé à des modèles spaCy français et anglais.

Une première passe a généré des faux positifs sur certains termes médicaux.

Le pipeline a donc été amélioré par :

* recentrage sur les entités réellement sensibles ;
* réduction de la détection de catégories trop bruitées ;
* ajout d'une liste d'exclusion de termes médicaux ;
* ajout de règles spécifiques pour certaines civilités ;
* contrôle des résultats.

Des faux positifs résiduels peuvent néanmoins exister.

Une utilisation réelle nécessiterait donc une validation supplémentaire du pipeline d'anonymisation.

---

# 🏷️ Annotation du signal de priorité

Les datasets sources ne comportaient pas directement les catégories de triage nécessaires au POC.

Un vivier de **355 cas cliniques uniques** a donc été constitué.

Après filtrage et recalibrage :

**214 cas ont été annotés manuellement.**

La distribution finale était :

| Niveau           |  Nombre | Proportion |
| ---------------- | ------: | ---------: |
| Urgence maximale |     101 |     47,2 % |
| Urgence modérée  |      80 |     37,4 % |
| Différée         |      33 |     15,4 % |
| **Total**        | **214** |  **100 %** |

L'annotation manuelle a été réalisée dans le cadre du POC par l'AI Engineer et **ne constitue pas une validation clinique par des professionnels de santé**.

---

# 📦 Structure des données

Chaque exemple suit un schéma unifié comprenant notamment :

```text
id
langue
source
source_id
instruction
contexte_patient
reponse
specialite
priorite
niveau_confiance
split
```

Le champ `niveau_confiance` permet notamment de distinguer les exemples issus des sources originales des exemples annotés manuellement.

---

#  Splits

Le dataset final a été divisé comme suit :

| Split      |    Nombre | Proportion |
| ---------- | --------: | ---------: |
| Train      |     5 227 |       80 % |
| Validation |       653 |       10 % |
| Test       |       654 |       10 % |
| **Total**  | **6 534** |  **100 %** |

La séparation a été stratifiée par source.

Le dataset préparé a été publié sur le Hugging Face Hub dans un dépôt privé afin de faciliter la reproductibilité du pipeline.

---

# 🤖 Modèle

Le modèle de base utilisé est :

**Qwen3-1.7B-Base**

Le fine-tuning est réalisé en deux étapes :

```text
Qwen3-1.7B-Base
       │
       ▼
SFT + LoRA
       │
       ▼
Modèle SFT
       │
       ▼
DPO
       │
       ▼
Modèle aligné
       │
       ▼
Fusion des adaptateurs
       │
       ▼
Modèle final
```

---

# 🧠 SFT — Supervised Fine-Tuning

Le SFT utilise :

* LoRA ;
* quantification 4-bit NF4 ;
* Qwen3-1.7B-Base ;
* Transformers ;
* TRL ;
* PEFT ;
* bitsandbytes ;
* Accelerate ;
* PyTorch.

Configuration LoRA :

| Paramètre               |   Valeur |
| ----------------------- | -------: |
| Rank                    |       16 |
| Alpha                   |       32 |
| Paramètres entraînables | ≈ 17,4 M |
| Proportion              |    ≈ 1 % |

---

# ⚙️ Stabilité de l'entraînement

Plusieurs expérimentations ont rencontré des divergences numériques avec apparition de `NaN`.

Les différentes hypothèses testées ont notamment concerné :

* gradient checkpointing ;
* multi-GPU ;
* optimiseur ;
* paramètres de chargement du modèle ;
* précision numérique.

La configuration finale stable repose sur des calculs en **float32**, avec entraînement sur GPU unique.

Cette configuration a permis d'obtenir un entraînement final stable sur **3 epochs / 492 steps**.

---

# 📈 Résultats SFT

| Métrique             |                      Résultat |
| -------------------- | ----------------------------: |
| Epochs               |                             3 |
| Training loss finale |                 ≈ 2,87 – 2,97 |
| Mean token accuracy  |                        63,8 % |
| Learning rate        |                          5e-5 |
| Configuration stable | float32 intégral / GPU unique |

Les notebooks SFT permettant de reproduire l'entraînement sont désormais présents dans le projet.

---

# 🎯 DPO — Direct Preference Optimization

Le modèle SFT est ensuite aligné par DPO.

Le dataset DPO utilise :

**1 500 paires de préférence**

issues d'UltraMedical-Preference.

Les exemples DPO sont distincts des exemples utilisés pour le SFT.

Configuration finale :

* 1 epoch ;
* 84 steps utiles après filtrage de longueur ;
* configuration float32 ;
* GPU unique.

---

# 📊 Résultats DPO

| Métrique             |   Résultat |
| -------------------- | ---------: |
| Training loss        |      0,542 |
| Validation loss      |      0,522 |
| Rewards / accuracies | **75,3 %** |
| Rewards / margins    |      1,148 |

Le score de **75,3 %** indique que le modèle choisit la réponse préférée dans environ trois cas sur quatre sur le jeu de validation.

---

# 📚 Reproductibilité de l'entraînement

Les notebooks SFT et DPO sont désormais complets.

Les éléments nécessaires à la reproduction sont documentés :

* modèle de base ;
* dataset ;
* splits ;
* seed ;
* hyperparamètres ;
* configuration LoRA ;
* précision ;
* paramètres d'entraînement ;
* configuration DPO ;
* environnement GPU ;
* versions des dépendances.

Les paramètres importants doivent être conservés avec les expériences afin de garantir la reproductibilité.

---

# 📦 Dépendances

Deux environnements sont séparés :

```text
requirements-training.txt
requirements-deployment.txt
```

## Environnement d'entraînement

Les versions correspondent à l'environnement GPU utilisé pour le projet :

```text
accelerate==1.14.0
bitsandbytes==0.50.1
datasets==5.0.1
huggingface_hub==1.28.0
peft==0.20.0
pytorch-ignite==0.5.4
pytorch-lightning==2.6.5
sentence-transformers==5.4.1
sentencepiece==0.2.2
tensorflow-datasets==4.9.9
torch==2.10.0+cu128
torchao==0.10.0
torchaudio==2.10.0+cu128
torchcodec==0.10.0+cu128
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.9.0
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.25.0+cu128
transformers==5.15.1
trl==1.10.0
vega-datasets==0.9.0
```

Installation :

```bash
pip install -r requirements-training.txt
```

## Environnement de déploiement

```text
fastapi==0.141.1
httpx==0.28.1
pydantic==2.13.4
pydantic_core==2.46.4
requests==2.34.2
requests-file==3.0.1
uvicorn==0.52.4
```

Installation :

```bash
pip install -r requirements-deployment.txt
```

La séparation permet de ne pas mélanger les dépendances lourdes de l'entraînement avec l'environnement de l'API.

---

# ☁️ Entraînement sur GPU

L'entraînement a été réalisé dans un environnement GPU, notamment Kaggle.

Installation des dépendances :

```bash
pip install -r requirements-training.txt
```

Les notebooks SFT et DPO contiennent les étapes d'exécution nécessaires :

```text
Préparation du dataset
        ↓
Chargement du tokenizer
        ↓
Chargement du modèle
        ↓
Configuration LoRA
        ↓
SFT
        ↓
Évaluation
        ↓
Sauvegarde
        ↓
DPO
        ↓
Évaluation
        ↓
Sauvegarde du modèle final
```

Le modèle final est ensuite préparé pour l'inférence et publié/stocké selon le workflow du projet.

---

# 🤗 Hugging Face

Hugging Face Hub est utilisé pour le stockage du dataset préparé et du modèle.

Avant publication d'un modèle ou d'un dataset, vérifier :

* licence ;
* droits de redistribution ;
* absence de secrets ;
* absence de données sensibles ;
* documentation du modèle ;
* version du modèle ;
* reproductibilité de l'expérience.

Les tokens Hugging Face ne doivent jamais être commités dans le dépôt.

---

# ⚡ vLLM

Le modèle final est servi par **vLLM**.

vLLM constitue le moteur d'inférence tandis que FastAPI constitue la couche applicative.

```text
FastAPI
   │
   │ HTTP
   ▼
vLLM
   │
   ▼
Qwen3-1.7B final
```

L'architecture utilise l'endpoint compatible :

```text
/v1/completions
```

L'inférence GPU a été validée dans l'environnement GPU du POC.

---

# 🚀 FastAPI

FastAPI expose la couche applicative.

Endpoints principaux :

```text
GET  /health
POST /triage
```

## `/health`

Permet de vérifier que l'API est disponible.

Exemple :

```bash
curl http://localhost:8000/health
```

## `/triage`

Permet de transmettre une requête de triage au système.

Le flux est :

```text
Requête
   │
   ▼
FastAPI
   │
   ▼
Validation
   │
   ▼
vLLM
   │
   ▼
Modèle
   │
   ▼
Réponse
   │
   ▼
Journalisation
```

---

# 🔐 Authentification

L'API dispose désormais d'un mécanisme d'authentification.

L'authentification fait partie de la version actuelle du projet et doit être utilisée pour contrôler l'accès aux endpoints protégés.

Les secrets et informations d'authentification ne doivent jamais être écrits directement dans le code source.

Ils doivent être fournis via l'environnement de déploiement ou le système de secrets utilisé.

---

# 📝 Logs et traçabilité

Chaque interaction avec `/triage` est journalisée de manière structurée.

Les informations de traçabilité comprennent notamment :

```text
interaction_id
timestamp
contexte_patient
question
réponse
latence
modèle utilisé
```

La journalisation permet :

* l'audit a posteriori ;
* l'analyse des performances ;
* le suivi des erreurs ;
* la traçabilité des interactions.

Les logs doivent être traités comme des données potentiellement sensibles et ne doivent pas contenir inutilement de données personnelles ou de secrets.

---

# 🐳 Docker

L'API est conteneurisée avec Docker.

Construire l'image :

```bash
docker build -t triage-api .
```

Lancer le conteneur :

```bash
docker run --rm -p 8000:8000 triage-api
```

Tester le health check :

```bash
curl http://localhost:8000/health
```

---

# 🐳 Docker Compose

Docker Compose est désormais intégré au projet afin de faciliter le lancement reproductible de l'environnement de déploiement.

Lancer :

```bash
docker compose up --build
```

En arrière-plan :

```bash
docker compose up -d --build
```

Arrêter :

```bash
docker compose down
```

Le fichier Compose permet de centraliser la configuration des services nécessaires à l'architecture.

> L'inférence GPU dépend de l'infrastructure utilisée. Le conteneur FastAPI peut être validé indépendamment dans un environnement local ne disposant pas de GPU CUDA.

---

# 🧪 Tests Pytest

Les tests automatisés sont intégrés au projet.

Lancer localement :

```bash
pytest
```

Les tests couvrent notamment les comportements de l'API et les fonctionnalités nécessaires au pipeline CI/CD.

Les cas importants comprennent :

* disponibilité de l'API ;
* `/health` ;
* validation des requêtes ;
* comportement de `/triage` ;
* gestion des erreurs.

---

# 🔄 CI/CD

Une pipeline GitHub Actions est intégrée au projet.

Le pipeline permet notamment de :

```text
Push
 │
 ▼
GitHub Actions
 │
 ├── Installation
 │
 ├── Tests Pytest
 │
 ├── Build Docker
 │
 ├── Démarrage du conteneur
 │
 ├── Health check
 │
 ├── Publication de l'image
 │
 └── Nettoyage
```

La pipeline a été testée et vérifiée.

L'image Docker est également publiée automatiquement dans le cadre du workflow CI/CD.

---

# 📈 Résultats d'inférence

Cinq scénarios représentatifs ont été testés :

* cas grave ;
* cas bénin ;
* traumatologie ;
* pédiatrie ;
* prompt minimal.

## Latence

| Métrique         | Résultat |
| ---------------- | -------: |
| Requêtes         |        5 |
| Taux de succès   |    100 % |
| Latence moyenne  |   2,41 s |
| Latence médiane  |   2,41 s |
| Latence minimale |   2,34 s |
| Latence maximale |   2,47 s |

La latence observée est relativement stable sur l'environnement de test GPU.

---

# 🧪 Robustesse de l'API

Les tests réalisés ont notamment vérifié :

| Scénario                       | Résultat                                 |
| ------------------------------ | ---------------------------------------- |
| Champ requis manquant          | `422`                                    |
| Champs vides                   | comportement contrôlé par l'API actuelle |
| Contexte dépassant 2048 tokens | erreur gérée par l'API                   |

Les tests Pytest et la CI/CD permettent désormais de vérifier automatiquement le comportement de l'application.

---

# 📊 Résultats qualitatifs

## Cas grave

Le modèle est capable de produire une réponse cohérente avec un cas présentant des signes de gravité.

Cependant, des artefacts d'anonymisation et certaines dérives terminologiques ont été observés.

## Cas bénin

Le modèle a également montré une limite critique :

un cas bénin a été classé comme **urgence maximale** avec une justification faisant référence à des pathologies ou complications absentes du contexte fourni.

Cet exemple illustre les deux principales limites du POC :

* sur-classification ;
* hallucination factuelle.

---

# ⚠️ Limites

## 1. Signal de priorité insuffisant

Seulement **214 cas** ont été annotés manuellement pour le signal de priorité.

Cela représente environ **4 % du corpus d'entraînement**.

Ce volume est insuffisant pour apprendre de manière robuste un véritable signal de triage clinique.

---

## 2. Sur-classification

Le modèle présente une tendance à classer certains cas bénins comme :

**urgence maximale**.

Cette limite est particulièrement importante dans un contexte de triage.

---

## 3. Hallucinations

Certaines réponses introduisent des informations cliniques absentes du contexte fourni.

Pour un système médical, ce comportement représente un risque majeur.

---

## 4. Validation clinique

Les annotations du signal de priorité n'ont pas été réalisées par un comité de professionnels de santé.

Une validation clinique formelle est indispensable avant toute utilisation réelle.

---

## 5. Anonymisation

Le pipeline d'anonymisation a été amélioré mais des faux positifs résiduels peuvent subsister.

Une revue humaine serait nécessaire pour une utilisation réelle.

---

## 6. Contextes longs

Le modèle possède une fenêtre de contexte limitée dans la configuration du POC.

Une stratégie de dégradation gracieuse doit être prévue pour les contextes dépassant la limite.

---

# 🚦 Go / No-Go

## État technique actuel

| Critère                    | Statut |
| -------------------------- | ------ |
| Modèle SFT                 | ✅      |
| Modèle DPO                 | ✅      |
| Inference vLLM             | ✅      |
| FastAPI                    | ✅      |
| Authentification           | ✅      |
| Logs sécurisés             | ✅      |
| Tests Pytest               | ✅      |
| CI/CD                      | ✅      |
| Docker                     | ✅      |
| Docker Compose             | ✅      |
| Publication d'image CI/CD  | ✅      |
| Requirements figés         | ✅      |
| Notebooks SFT/DPO          | ✅      |
| Résultats / courbes / logs | ✅      |

---

## État clinique

| Critère                                     | Statut  |
| ------------------------------------------- | ------- |
| Fiabilité du signal de priorité             | ❌ NO-GO |
| Validation par des professionnels de santé  | ❌ NO-GO |
| Absence d'hallucinations dangereuses        | ❌ NO-GO |
| Corpus de triage suffisamment représentatif | ❌ NO-GO |

---

# 🔴 Verdict

## NO-GO pour la production clinique

Le POC ne doit pas être utilisé pour prendre des décisions médicales réelles.

Les limites observées sur le signal de priorité et les hallucinations empêchent actuellement toute utilisation clinique autonome.

## 🟢 GO pour la Phase 2

Le POC atteint néanmoins son objectif technique :

> démontrer la faisabilité d'une chaîne complète de développement, entraînement, alignement, serving, API, sécurité, tests, conteneurisation et CI/CD d'un agent LLM spécialisé.

---

# 🗺️ Roadmap Phase 2

## 1. Construire un véritable corpus de triage

Collaborer avec des professionnels de santé afin de constituer un corpus de cas de triage représentatif.

## 2. Augmenter fortement le nombre d'annotations

Le signal de priorité doit être appris à partir d'un volume beaucoup plus important de cas.

## 3. Validation clinique

Mettre en place une validation formelle avec un comité médical.

## 4. Ajouter un mécanisme d'incertitude

Le système devrait pouvoir retourner une réponse du type :

```text
Indéterminé — évaluation humaine requise.
```

plutôt que de forcer une catégorie lorsque les informations sont insuffisantes.

## 5. Réduire les hallucinations

Explorer :

* grounding ;
* retrieval ;
* règles de validation ;
* vérification des faits ;
* contraintes de génération ;
* supervision humaine.

## 6. Gestion des contextes longs

Ajouter :

* troncature contrôlée ;
* résumé ;
* segmentation ;
* ou réponse `indéterminée`.

## 7. Renforcer l'anonymisation

Combiner l'anonymisation automatique avec une revue humaine ciblée.

---

# 🚀 Projection Phase 3

Après validation de la Phase 2, une montée en charge pourrait envisager :

* modèles plus importants ;
* datasets cliniques beaucoup plus larges ;
* données structurées ;
* symptômes ;
* antécédents ;
* constantes vitales ;
* protocoles de triage ;
* monitoring ;
* infrastructure GPU de production ;
* mécanismes de bascule ;
* surveillance post-déploiement.

---

# 🧑‍💻 Commandes principales

## Installer les dépendances d'entraînement

```bash
pip install -r requirements-training.txt
```

## Installer les dépendances de déploiement

```bash
pip install -r requirements-deployment.txt
```

## Lancer les tests

```bash
pytest
```

## Lancer FastAPI en développement

```bash
uvicorn app.main:app --reload
```

## Vérifier l'API

```bash
curl http://127.0.0.1:8000/health
```

## Construire l'image Docker

```bash
docker build -t triage-api .
```

## Lancer Docker

```bash
docker run --rm -p 8000:8000 triage-api
```

## Lancer Docker Compose

```bash
docker compose up --build
```

## Arrêter Docker Compose

```bash
docker compose down
```

---

# 🔐 Variables d'environnement

Les secrets doivent être fournis via l'environnement de déploiement.

Exemples de paramètres susceptibles d'être nécessaires selon la configuration :

```text
HF_TOKEN
VLLM_URL
API_KEY
```

Les valeurs réelles ne doivent jamais être ajoutées au dépôt.

Le fichier `.env` doit être exclu de Git :

```gitignore
.env
```

---

# 🔬 Reproductibilité

Pour reproduire le projet :

```text
1. Installer les dépendances
        ↓
2. Préparer les données
        ↓
3. Exécuter le notebook SFT
        ↓
4. Évaluer le modèle SFT
        ↓
5. Exécuter le notebook DPO
        ↓
6. Évaluer le modèle DPO
        ↓
7. Fusionner / sauvegarder le modèle
        ↓
8. Démarrer vLLM
        ↓
9. Démarrer FastAPI
        ↓
10. Exécuter Pytest
        ↓
11. Construire l'image Docker
        ↓
12. Valider le déploiement
```

Les seeds et hyperparamètres utilisés sont documentés dans les notebooks et les configurations associées.

---

# 📚 Technologies utilisées

| Technologie      | Rôle                 |
| ---------------- | -------------------- |
| Python           | Langage principal    |
| PyTorch          | Deep Learning        |
| Qwen3-1.7B       | Modèle de base       |
| Transformers     | LLM                  |
| TRL              | SFT / DPO            |
| PEFT             | Fine-tuning efficace |
| LoRA             | Adaptation du modèle |
| bitsandbytes     | Quantification       |
| Datasets         | Gestion des données  |
| Accelerate       | Entraînement GPU     |
| Presidio         | Anonymisation        |
| spaCy            | NLP / NER            |
| Kaggle           | Environnement GPU    |
| Hugging Face Hub | Modèles et datasets  |
| vLLM             | Serving / inférence  |
| FastAPI          | API REST             |
| Pydantic         | Validation           |
| Uvicorn          | Serveur ASGI         |
| Pytest           | Tests                |
| Docker           | Conteneurisation     |
| Docker Compose   | Orchestration        |
| GitHub Actions   | CI/CD                |

---

# 📄 Documentation technique

Le projet est accompagné du rapport technique :

```text
Rapport_Technique_POC_Triage_CHSA.docx
```

Ce document détaille :

* le contexte ;
* les objectifs ;
* la préparation des données ;
* l'anonymisation ;
* l'annotation ;
* le SFT ;
* le DPO ;
* l'infrastructure ;
* les résultats ;
* les limites ;
* la checklist Go/No-Go ;
* la roadmap Phase 2.

---

# 📋 État final du projet

|  # | Point                                | Statut            |
| -: | ------------------------------------ | ----------------- |
|  1 | Nettoyage du dépôt                   | ✅ Fait            |
|  2 | Notebooks SFT/DPO complets           | ✅ Fait            |
|  3 | Résultats d'exécution, courbes, logs | ✅ Fait            |
|  4 | Graines aléatoires + hyperparamètres | ✅ Fait            |
|  5 | README complet                       | ✅ Fait            |
|  6 | Versions des dépendances figées      | ✅ Fait            |
|  7 | Analyse infrastructure               | ✅ Fait            |
|  8 | Tests Pytest CI/CD                   | ✅ Fait et vérifié |
|  9 | Docker Compose                       | ✅ Fait            |
| 10 | Publication d'image CI/CD            | ✅ Fait            |
| 11 | Authentification + logs sécurisés    | ✅ Fait et vérifié |

**État du projet : 🟢 POC techniquement complet**

**État pour une utilisation clinique : 🔴 NO-GO**

**Prochaine étape : 🟢 Phase 2 — validation clinique et amélioration du signal de priorité**

---

# 👤 Auteur

**Haroun Tanane**

Projet réalisé dans le cadre du parcours **AI Engineer**.

---

# 📌 Conclusion

Ce POC démontre qu'il est possible de construire une chaîne complète autour d'un LLM spécialisé :

```text
Données
  ↓
Nettoyage
  ↓
Anonymisation
  ↓
Annotation
  ↓
SFT
  ↓
DPO
  ↓
Évaluation
  ↓
Fusion du modèle
  ↓
vLLM
  ↓
FastAPI
  ↓
Authentification
  ↓
Logs sécurisés
  ↓
Pytest
  ↓
Docker
  ↓
Docker Compose
  ↓
CI/CD
  ↓
Publication de l'image
```

Le POC atteint ainsi son objectif d'ingénierie : **démontrer la faisabilité technique de bout en bout d'un agent LLM spécialisé pour le triage**.

Les résultats expérimentaux montrent toutefois que la faisabilité technique ne suffit pas à garantir la sécurité clinique. La sur-classification, les hallucinations et l'absence de validation médicale imposent un **NO-GO pour la production**.

La suite logique du projet est donc une **Phase 2 encadrée avec des professionnels de santé**, un corpus de triage beaucoup plus représentatif et une validation clinique rigoureuse.


📚 Project Repository

Le code complet du projet, les notebooks d'entraînement, les tests, le déploiement Docker et la CI/CD sont disponibles dans le dépôt GitHub associé au projet.

GitHub : https://github.com/Marrackech/triage-med

👤 Author

Haroun Tanane

AI Engineer — Projet réalisé dans le cadre du parcours AI Engineer.