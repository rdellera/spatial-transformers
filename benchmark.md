# Benchmark – costruzione degli embedding

Registro dei test sulle risorse (tempo, memoria GPU, RAM) necessarie per costruire l'embedding di un dataset `z_D` con il modello del paper, al crescere del numero di vertici per poligono e di geometrie per dataset. Si misura solo la costruzione dell'embedding, non il training.

- Script: `benchmark_embedding.py` (stesso percorso di `compute_frozen_embeddings`: lettura → limiti di campionamento → polygon encoder → dataset encoder).
- Dati: generati con AIDA (`Generator.py`, distribuzione `uniform`), in `data/benchmark/g<geometrie>_v<maxseg>/`. Ogni poligono ha un numero casuale di vertici tra 4 e `maxseg`. Ci sono 3 repliche per combinazione.
- Dati grezzi: `outputs_benchmark/embedding_benchmark.csv`, una riga per misura.

## Hardware di riferimento ("computer normale")

| Componente | Valore |
|---|---|
| CPU | AMD Ryzen 7 5800H |
| RAM | 16 GB |
| GPU | NVIDIA RTX 3050 Ti Laptop, 4 GB |
| Software | Windows 11, Python 3.11.9, torch 2.15.0+cu132 |

## Parametri che influiscono sul calcolo

| Gruppo | Parametri (valore nel paper) | Impatto sul costo | Nei test |
|---|---|---|---|
| Dimensione dell'input | V vertici per poligono, G geometrie per dataset | Variabili indipendenti. Attenzione full: memoria O(V²) nello stadio 1 e O(G²) nello stadio 2 | **Variate** |
| Strategia di attenzione | `ATTENTION_STRATEGY`: `sampling` (attenzione full) o `sparse` | *sparse* guarda solo K vicini: memoria O(V·K) e O(G·K) | **Variata** |
| Limiti di campionamento | *sampling*: lettura 256 vertici / 512 geometrie, embedding 128 / 512. *sparse*: lettura 4096 / 20 000, embedding 512 / 1024 | Oltre la soglia i dati in più vengono scartati a caso e il costo resta costante | **Due configurazioni**: `paper` e `none` (disattivati) |
| Architettura | `D_MODEL` 128, `NUM_HEADS` 4, `POLYGON_LAYERS` 3, `DATASET_LAYERS` 2, `mlp_ratio` 4, bias MLP 128, bande di Fourier 8/6 | Definisce il modello | Fissi |
| Vicinato sparse | finestra sul bordo 8, k vertici 16, k geometrie 64 | Fissa K | Fissi |
| Esecuzione | `POLYGON_BATCH_SIZE` 1, float32, `eval()` + `no_grad` | Un poligono alla volta | Fissi |

Esiti possibili:
- `ok`;
- `oom_gpu`: memoria GPU esaurita;
- `oom_ram`: memoria RAM esaurita;
- `crash`;
- `timeout`: oltre 30 minuti per una misura;
- `skipped`: la stessa configurazione è già fallita su un input più piccolo.

---

## Test 1 – Griglia fino a 10 000 geometrie

- **Data:** 2026-11-07
- **Codice:** commit `70e3274` (branch `feat/resource-metric`)
- **Griglia:** G ∈ {100, 1 000, 10 000} × `maxseg` ∈ {5, 10, 20, 200, 2000} × 3 repliche, cioè 45 dataset × 4 configurazioni = 180 misure
- **Esiti:** 162 `ok`, 1 `timeout`, 1 `oom_gpu`, 16 `skipped`

Ogni cella riporta **tempo totale medio (s) / picco GPU (MB)**, con media sulle 3 repliche. Il tempo totale comprende lettura del file, stadio 1 e stadio 2.

### `sampling` con i limiti del paper
| G \ maxseg | 5 | 10 | 20 | 200 | 2000 |
|---|---|---|---|---|---|
| 100 | 0.83 / 19 | 0.74 / 19 | 0.74 / 19 | 0.75 / 38 | 0.89 / 38 |
| 1 000 | 3.63 / 358 | 3.59 / 358 | 3.73 / 358 | 3.85 / 358 | 4.98 / 358 |
| 10 000 | 4.88 / 358 | 4.68 / 358 | 4.70 / 358 | 6.38 / 358 | 17.50 / 358 |

### `sparse` con i limiti del paper
| G \ maxseg | 5 | 10 | 20 | 200 | 2000 |
|---|---|---|---|---|---|
| 100 | 1.11 / 15 | 1.05 / 15 | 1.01 / 15 | 1.07 / 19 | 1.16 / 40 |
| 1 000 | 9.58 / 108 | 9.67 / 108 | 9.61 / 108 | 9.40 / 108 | 10.45 / 108 |
| 10 000 | 11.80 / 108 | 13.94 / 108 | 11.37 / 108 | 13.71 / 108 | 23.47 / 108 |

### `sampling` senza limiti (attenzione full su tutti i dati)
| G \ maxseg | 5 | 10 | 20 | 200 | 2000 |
|---|---|---|---|---|---|
| 100 | 0.75 / 19 | 0.77 / 19 | 0.76 / 19 | 0.84 / 86 | 812 / 7 972 ⚠ |
| 1 000 | 8.08 / 1 347 | 7.27 / 1 347 | 7.42 / 1 347 | 7.30 / 1 348 | `timeout` |
| 10 000 | `oom_gpu` | `skipped` | `skipped` | `skipped` | `skipped` |

⚠ Picco GPU oltre i 4 GB fisici, con RAM di processo a 7,4 GB: vedi osservazione 3.

### `sparse` senza limiti
| G \ maxseg | 5 | 10 | 20 | 200 | 2000 |
|---|---|---|---|---|---|
| 100 | 1.11 / 15 | 1.03 / 15 | 1.03 / 15 | 1.14 / 19 | 1.66 / 140 |
| 1 000 | 9.63 / 108 | 9.15 / 108 | 9.15 / 108 | 9.53 / 108 | 16.95 / 141 |
| 10 000 | 105.32 / 1 017 | 106.81 / 1 017 | 105.91 / 1 017 | 115.04 / 1 017 | 153.20 / 1 018 |

### Osservazioni
1. **Con i limiti del paper il costo è limitato in partenza.** Oltre 512 geometrie (`sampling`) o 1024 (`sparse`) i dati in più vengono scartati: il picco GPU resta fermo a 358 MB e 108 MB, e il tempo cresce solo per la lettura del file. Su `g10000_v2000` circa 13 dei 17,5 s sono lettura. Quindi queste configurazioni non costruiscono l'embedding di *tutto* il dataset.
2. **L'attenzione full senza limiti è la prima a cedere, su entrambi gli assi.**
   - *Geometrie:* con 10 000 geometrie va in `oom_gpu` anche con poligoni da ≤5 vertici, perché lo stadio 2 crea tensori G×G. A 1 000 geometrie servono già 1,35 GB.
   - *Vertici:* con poligoni da ≤2000 vertici un solo poligono richiede circa 8 GB, perché lo stadio 1 crea tensori V×V. Con 100 geometrie la misura riesce, ma in 812 s; con 1 000 supera i 30 minuti.
   - Limite del portatile per `sampling` senza limiti: tra 1 000 e 10 000 geometrie, e tra 200 e 2000 vertici per poligono.
3. **Su Windows la GPU non va subito in errore quando la memoria finisce.** Il driver NVIDIA sposta la parte in eccesso nella RAM condivisa: il calcolo prosegue ma diventa centinaia di volte più lento (812 s contro 0,84 s della fascia ≤200). Il limite pratico appare quindi come un tempo enorme, non come un errore. L'`oom_gpu` compare solo quando una singola richiesta è troppo grande anche per la memoria condivisa.
4. **`sparse` senza limiti regge tutta la griglia**, fino a 10 000 geometrie × ≤2000 vertici, cioè circa 10 milioni di vertici, in 153 s con 1 GB di GPU.
   - Il tempo cresce circa in modo lineare con G: 1,1 → 9,5 → 105 s.
   - Fino a 200 vertici il tempo dipende poco da V.
   - Il picco GPU cresce in modo lineare con G, circa 0,1 MB per geometria, per i tensori dei vicini dello stadio 2 (G × 64). Estrapolando, a 100 000 geometrie servirebbero circa 10 GB, oltre i 4 GB disponibili: da verificare nel Test 2.
5. **Il tempo è dominato dallo stadio 1 (oltre l'85%) a causa di `POLYGON_BATCH_SIZE = 1`.** Ogni poligono viene codificato da solo, con un costo fisso di circa 10 ms indipendente da V fino a 200 vertici. Quindi oggi il tempo misura il numero di chiamate più che il calcolo.
6. **La RAM non è mai stata il limite.** Il picco è tra 1,15 e 1,45 GB, di cui circa 1,1 GB dovuti a torch e CUDA. L'unica eccezione è il caso del punto 3 (7,4 GB), dovuto alla memoria GPU spostata in RAM.
