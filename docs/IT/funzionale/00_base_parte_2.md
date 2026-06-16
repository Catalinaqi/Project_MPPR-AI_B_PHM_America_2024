
---

# 📑 Indice

- [🔍 Piano di Audit Profondo dei Dati (Fase di Validazione Fisica)](#piano-di-audit-profondo-dei-dati-fase-di-validazione-fisica)
- [💡 La Regola d'Oro](#la-regola-doro)
- [1. Criteri Tecnici e Intervalli Consentiti](#1-criteri-tecnici-e-intervalli-consentiti)
- [2. Definizione del Pipeline di Controllo Qualità](#2-definizione-del-pipeline-di-controllo-qualità)
  - [A. Anomalie di Sensore (Dati da Scartare)](#a-anomalie-di-sensore-dati-da-scartare)
  - [B. Contesto Operativo (Dati da Conservare)](#b-contesto-operativo-dati-da-conservare)

---

# 🔍 Piano di Audit Profondo dei Dati (Fase di Validazione Fisica)

Basato sulle specifiche fisiche del motore e sulla dinamica di volo, abbiamo progettato questo piano di audit per garantire che i dati di telemetria siano **fisicamente realistici**.

L'obiettivo principale è garantire che i dati di telemetria siano **fisicamente realistici** prima di introdurli nei modelli predittivi,
assicurando che rispettino i limiti operativi nominali di un motore a turbina e gli standard di **PHM (Prognostics and Health Management)**.

- Nota: calcolo del **Margine di Coppia (Target vs. Realtà)** dettagliato in
  [`00_base_parte_1.md`](00_base_parte_1.md)


### 💡 La Regola d'Oro

Quando si esegue l'audit di dataset industriali, applichiamo una distinzione fondamentale:

> *"Un'anomalia di sensore è una violazione della fisica termodinamica; un'anomalia operativa è una violazione dell'inviluppo di crociera standard."*

L'obiettivo è scartare i **falsi positivi** (dati che sembrano errori ma sono comportamenti reali) e conservare solo la **spazzatura tecnica** (letture impossibili che degraderebbero il modello).

---

## 1. Criteri Tecnici e Intervalli Consentiti

Abbiamo regolato le soglie per riflettere la realtà operativa rilevata nel dataset, consentendo stati come *Ground Idle* e pressioni atmosferiche estreme.

| Campo | Significato Fisico | Intervallo Consentito / Previsto | Giustificazione Tecnica |
| --- | --- | --- | --- |
| **`oat`** | *Outside Air Temperature* (Celsius) | `-50.0 a 60.0` | Limiti climatici estremi per operazione sicura. |
| **`ias`** | *Indicated Airspeed* (Nodi) | `0.0 a 250.0` | `0.0` rappresenta stato di *hover* o in pista; è un dato valido e critico. |
| **`pa`** | *Pressure Altitude* (Piedi) | `-1000.0 a 20000.0` | Altitudini negative sono fisicamente possibili a causa di alta pressione o depressioni geografiche. |
| **`mgt`** | *Measured Gas Temp* (°C) | `200.0 a 1000.0` | Soglia termica; valori `< 200°C` implicano motore spento o sensore morto. |
| **`ng`** | *Gas Generator Speed* (%) | `60.0 a 110.0` | Il nucleo non può girare al di sotto di questo regime in operazione; `0` indica guasto totale. |
| **`np`** | *Power Turbine Speed* (%) | `80.0 a 115.0` | Include *Ground Idle* e transitori di accensione; l'intervallo `>= 80%` è operativamente valido. |
| **`trq_measured`** | Coppia Reale Misurata (%) | `0.0 a 125.0` | Valori negativi non hanno senso fisico; implicano frattura dell'albero o errore del sensore. |
| **`trq_margin`** | Margine di Coppia (%) | `-100.0 a 100.0` | Variabile obiettivo. Deviazioni estreme denotano guasti critici di telemetria. |

---

## 2. Definizione del Pipeline di Controllo Qualità

Il pipeline di ingegneria dei dati classifica i record in due categorie di validazione:

### A. Anomalie di Sensore (Dati da Scartare)

Sono violazioni della fisica termodinamica e meccanica. Se questi limiti vengono infranti, il dato è tecnicamente spazzatura.

* **`ng < 60.0`**: Indica un guasto del sensore, poiché il nucleo non può girare al di sotto di questo regime quando acceso.
* **`trq_measured < 0.0`**: Fisicamente impossibile; se c'è coppia negativa, il sensore o l'albero sono fratturati.
* **`mgt < 200.0`**: Soglia di "motore freddo". Se si verifica durante il volo (`ias > 0`), il sensore è morto.

### B. Contesto Operativo (Dati da Conservare)

Sono stati di volo validi che il modello deve imparare a distinguere. Non devono essere eliminati.

* **Altitudine di Pressione (`pa`) tra `-1000` e `0**`: Validazione di condizioni atmosferiche reali (alta pressione) o depressioni geografiche, non errori dell'altimetro.
* **RPM del Rotore (`np`) tra `80` e `90**`: Classificato come *Ground Idle* (ralenti a terra). È uno stato transitorio legittimo per l'addestramento.
* **Velocità (`ias`) = `0.0****: Identifica la manovra di *hover* (stazionario). Eliminare questo dato significherebbe perdere la fase di volo più critica dell'elicottero.
