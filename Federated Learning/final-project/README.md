# Federated Fine-tuning of ViT-B/16 on EuroSAT

**Final Project — Federated Learning · MADI · Tecnun, Universidad de Navarra**  
**Autor:** Enrique Lahuerta · Febrero 2026

---

## Descripción

Fine-tuning federado de **ViT-B/16** (Vision Transformer, preentrenado en ImageNet) sobre **EuroSAT RGB** — un dataset de 27.000 imágenes de satélite Sentinel-2 con 10 clases de uso del suelo.

El proyecto analiza cuatro dimensiones:
1. **Head-only vs Full fine-tuning** — impacto en accuracy y overhead de comunicación
2. **Número de nodos** — 5 vs 10 clientes
3. **Heterogeneidad de datos** — IID vs Dirichlet α=0.3
4. **Trade-off comunicación/accuracy** — 30 KB vs 344 MB por ronda por cliente

### Experimentos

| Federación | Nodos | Modo | Particionado | LR | Rondas |
|---|---|---|---|---|---|
| `head-5nodes` | 5 | Head-only | IID | 1e-3 | 10 |
| `head-10nodes` | 10 | Head-only | IID | 1e-3 | 10 |
| `head-noniid` | 5 | Head-only | Dirichlet α=0.3 | 1e-3 | 10 |
| `full-finetune` | 5 | Full FT | IID | 1e-4 | 5 |

---

## Setup

```bash
conda activate flwr
cd final-project
pip install -e .
```

---

## Ejecución

Lanza los experimentos en orden. Cada uno guarda automáticamente resultados en `outputs/<timestamp>/`.

```bash
# Exp 1: Head-only, 5 nodos, IID  (~3-5 min/ronda)
flwr run .

# Exp 2: Head-only, 10 nodos, IID
flwr run . head-10nodes

# Exp 3: Head-only, 5 nodos, non-IID
flwr run . head-noniid

# Exp 4: Full fine-tuning, 5 nodos (más lento, análisis de comunicación)
flwr run . full-finetune
```

---

## Resultados

Cada experimento genera en `outputs/<timestamp>/`:
- `metrics.json` — accuracy/loss por ronda + info de bytes transmitidos
- `run_config.json` — configuración exacta del experimento
- `final_model.pt` — pesos del modelo global final

Abre `analysis.ipynb` para generar todos los plots y la tabla de comunicación.

---

## Por qué head-only como default

Con `freeze-backbone=true` solo se entrenan los **7.690 parámetros** de la cabeza clasificadora (768→10), en lugar de los **86 millones** del modelo completo. Esto supone:

- **Reducción de comunicación del 99.99%**: 30 KB vs 344 MB por cliente por ronda
- **Velocidad**: ~3-5 min/ronda vs >30 min/ronda en GPU de 8GB
- **Justificación técnica real**: en FL la comunicación es el cuello de botella principal, no el cómputo local

El experimento `full-finetune` (5 rondas) sirve como comparativa para cuantificar el trade-off.

---

## Estructura del proyecto

```
final-project/
  vitexample/
    __init__.py
    task.py          # ViT-B/16 + EuroSAT + train/test
    client_app.py    # ClientApp federado
    server_app.py    # ServerApp con evaluación centralizada
  pyproject.toml     # 4 federaciones configuradas
  analysis.ipynb     # Análisis y plots
  README.md
  outputs/           # Resultados (generados al ejecutar)
```
