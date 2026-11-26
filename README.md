# Benchmark delle risorse per gli spatial embedding

Questo repository estende il codice del paper *Spatial Transformers* con un benchmark delle risorse computazionali (tempo, memoria GPU, RAM) richieste dalla costruzione dell'embedding di un dataset `z_D`, al variare del numero di geometrie e di vertici per poligono. Il benchmark misura la sola costruzione dell'embedding, non il training.

L'architettura del modello, gli esperimenti OPT-1/OPT-2, il formato dei dati e l'installazione sono descritti nel README del progetto originale: <https://github.com/smigliorini/spatial-transformers>.

## Struttura del progetto

```text
benchmark_embedding.py              Script del benchmark
train_selectivity_opt1_opt2.py      Codice del paper: build_encoders, encode_dataset_differentiable, limiti per fase
spatial_data_loading.py             Codice del paper: read_polygons
data/benchmark/g<G>_v<maxseg>/      Un gruppo di dataset per combinazione (G, maxseg)
    sum_g<G>_v<maxseg>.csv          Sommario AIDA del gruppo
    commands.log                    Comando Generator.py di ogni replica (riga n -> uniform_<n>.wkt)
    uniform_<n>.wkt                 Dataset generati (non versionati)
outputs_benchmark/
    embedding_benchmark.csv         Risultati grezzi, una riga per misura
    run.log                         Log dell'esecuzione
```

Il benchmark non modifica le funzioni del paper: le richiama dallo script.

## Impianto del benchmark

**Percorso misurato.** È lo stesso di `compute_frozen_embeddings`:

```text
read_polygons -> limiti di campionamento -> PolygonEncoder (stadio 1) -> DatasetEncoder (stadio 2, vicini kNN inclusi) -> z_D
```

**Griglia.** Le due variabili indipendenti sono il numero di geometrie per dataset G e il numero massimo di vertici per poligono `maxseg`. Per ogni combinazione si generano 10 repliche, per un totale di 450 dataset e 1800 misure.

| Variabile | Valori |
|---|---|
| G | 100, 200, 400, 1000, 2000, 4000, 10000, 20000, 40000 (passi ×2 o ×2,5) |
| `maxseg` | 5, 10, 20, 200, 2000 (ogni poligono ha da 4 a `maxseg` vertici) |
| Configurazione | `sampling` / `sparse` × limiti del paper (`paper`) / senza limiti (`none`) |

**Parametri che influiscono sul calcolo.**

| Gruppo | Parametri (valore del paper) | Effetto sul costo | Nel benchmark |
|---|---|---|---|
| Dimensione dell'input | V vertici per poligono, G geometrie per dataset | Con attenzione full: memoria O(V²) nello stadio 1 e O(G²) nello stadio 2 | Variati |
| Strategia di attenzione | `ATTENTION_STRATEGY`: `sampling` (full) o `sparse` | Con `sparse`, K vicini: memoria O(V·K) e O(G·K) | Variata |
| Limiti di campionamento | `sampling`: lettura 256 vertici / 512 geometrie, embedding 128 / 512. `sparse`: lettura 4096 / 20000, embedding 512 / 1024 | Oltre la soglia, i dati in eccesso vengono scartati in modo casuale | `paper` o `none` |
| Architettura | `D_MODEL` 128, `NUM_HEADS` 4, `POLYGON_LAYERS` 3, `DATASET_LAYERS` 2 | Definisce il modello | Fissi |
| Vicinato `sparse` | finestra sul bordo 8, k vertici 16, k geometrie 64 | Fissa K | Fissi |
| Esecuzione | `POLYGON_BATCH_SIZE` 1, float32, `eval()` + `no_grad` | Un poligono alla volta | Fissi |

**Grandezze misurate** (colonne di `embedding_benchmark.csv`):
- identificazione: `group`, `dataset`, `geometries`, `max_vertices`, `strategy`, `caps`, `status`;
- dati effettivamente usati: `polygons_read`, `polygons_used`, `vertices_used`, `max_vertices_used`;
- tempi in secondi: `read_s`, `stage1_s`, `stage2_s`, `total_s`;
- memoria in MB:
  - `gpu_stage1_mb`: picco fino all'avvio del `DatasetEncoder`;
  - `gpu_stage2_mb`: picco da quel punto alla fine, inclusa la memoria ancora occupata dallo stadio 1; il campo è vuoto se lo stadio 2 non è iniziato;
  - `gpu_peak_mb`: il massimo dei due;
  - `ram_peak_mb`.

**Esiti.**
- `ok`;
- `oom_gpu` / `oom_ram`: memoria esaurita;
- `crash`: processo terminato senza risultati;
- `timeout`: oltre 30 minuti;
- `skipped`: la stessa configurazione è già fallita su un input non più grande.

## Scelte progettuali

- **Pesi casuali con la configurazione del paper.** Le risorse dipendono dalle dimensioni dei tensori, non dai valori dei pesi, quindi il training non è necessario.
- **Un processo per misura.** Il picco di RAM è attribuibile alla singola misura, e un errore di memoria non interrompe il benchmark.
- **Warm-up preliminare.** L'inizializzazione del contesto CUDA e dei kernel resta esclusa dalla misura.
- **Picco GPU per stadio.** Un hook sull'avvio del `DatasetEncoder` separa la memoria dei due stadi, così si individua lo stadio che determina il limite.
- **Regola di salto.** Una configurazione fallita non può riuscire su un input più grande, quindi le misure successive vengono saltate per contenere la durata.
- **Ripresa dopo interruzione.** Ogni misura viene scritta nel CSV appena conclusa; al riavvio quelle già presenti non vengono ripetute.
- **Dati AIDA `uniform`, 10 repliche senza seed.** La variabilità tra repliche rende misurabile la dispersione dei risultati.
- **Griglia a raddoppi.** I rapporti tra passi consecutivi descrivono la crescita delle risorse soltanto con valori misurati, senza estrapolazioni.
- **CUDA Sysmem Fallback disattivato.** Su Windows il driver sposterebbe nella RAM la memoria GPU in eccesso, trasformando l'errore in un rallentamento di diversi ordini di grandezza.

## Riproduzione

Requisiti: Python 3.11, `requirements.txt` e una build CUDA di PyTorch. Per i dettagli si rimanda al README originale.

**Generazione dei dati.** Si usa `Generator.py` dell'[AIDA tool](https://github.com/smigliorini/aida-augumentation-tool), branch `main`, commit `e4a2548`. Lo strumento importa `cgi`, quindi richiede Python ≤ 3.12. I file occupano circa 30 GB. Dalla radice del repository:

```bash
AIDA=/path/to/aida-augumentation-tool
for dir in data/benchmark/*/; do
  n=1
  tr -d '\r' < "$dir/commands.log" | while read -r command; do
    (cd "$AIDA" && $command) > "$dir/uniform_$n.wkt"
    n=$((n + 1))
  done
done
```

In alternativa si può eseguire `GeneratorCSV.py` su ciascun `sum_<gruppo>.csv`. AIDA non fissa un seed, quindi i poligoni rigenerati sono diversi, ma il numero di geometrie e di vertici, che determina il costo, segue gli stessi parametri.

**Esecuzione.** Prima del lancio, su Windows, disattivare il CUDA Sysmem Fallback per l'eseguibile Python.

```bash
mkdir -p outputs_benchmark
python -u benchmark_embedding.py 2>&1 | tee -a outputs_benchmark/run.log
```

Sul portatile di riferimento (RTX 3050 Ti Laptop, 4 GB) la griglia completa richiede circa 10 ore. Per ripartire da zero occorre cancellare `outputs_benchmark/embedding_benchmark.csv`.

## Risultati

I risultati grezzi sono in `outputs_benchmark/embedding_benchmark.csv`, una riga per misura, e il log dell'esecuzione in `outputs_benchmark/run.log`. I test precedenti, eseguiti su una griglia ridotta e con il fallback attivo, restano consultabili nella cronologia git.
