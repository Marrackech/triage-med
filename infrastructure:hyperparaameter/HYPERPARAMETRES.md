# Hyperparamètres finaux — Reproductibilité

## Seed
Toutes les phases d'entraînement utilisent `seed=42` (random, numpy, torch, cuda).

## SFT (Supervised Fine-Tuning)

| Paramètre | Valeur |
|---|---|
| Modèle de base | Qwen/Qwen3-1.7B-Base |
| Méthode | LoRA |
| Rang LoRA (r) | 16 |
| Alpha LoRA | 32 |
| Dropout LoRA | 0.05 |
| Modules ciblés | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Quantization | 4-bit NF4, double quant |
| Compute dtype | float32 (choix de stabilité, voir rapport section 3.1) |
| Optimiseur | paged_adamw_8bit |
| Learning rate | 5e-5 |
| Scheduler | cosine |
| Warmup steps | 20 |
| Max grad norm | 0.3 |
| Batch size (device) | 8 |
| Gradient accumulation | 2 |
| Batch effectif | 16 |
| Epochs | 3 |
| Max sequence length | 1024 |
| Seed | 42 |

## DPO (Direct Preference Optimization)

| Paramètre | Valeur |
|---|---|
| Modèle de départ | Modèle SFT ci-dessus (adaptateurs fusionnés) |
| Méthode | LoRA (nouveaux adaptateurs) |
| Rang LoRA (r) | 16 |
| Alpha LoRA | 32 |
| Beta (force de préférence) | 0.1 |
| Optimiseur | paged_adamw_8bit |
| Learning rate | 5e-5 |
| Max grad norm | 0.3 |
| Batch size (device) | 1 |
| Gradient accumulation | 8 |
| Batch effectif | 8 |
| Epochs | 1 |
| Max sequence length | 512 |
| Seed | 42 |
| Volume de paires | 1500 (train), 150 (validation) |

## Reproductibilité — limites connues
Le déterminisme complet sur GPU n'est pas garanti à 100 % même avec une seed fixée (opérations CUDA non-déterministes par défaut). Pour un déterminisme strict, ajouter `torch.use_deterministic_algorithms(True)` — non activé ici pour préserver la vitesse d'entraînement sur un POC à délai contraint.
