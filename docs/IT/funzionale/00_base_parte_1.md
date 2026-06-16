
---

# 📑 Indice

- [1. Il concetto base: cos'è la Coppia / Torque?](#1-il-concetto-base-cos%C3%A8-la-coppia--torque)
- [2. I due valori in gioco (Teoria vs. Realtà)](#2-i-due-valori-in-gioco-teoria-vs-realt%C3%A0)
  - [2.1. Coppia desiderata / Torque Target / Target](#21-coppia-desiderata--torque-target--target)
  - [2.2. Coppia fornita / Coppia misurata](#22-coppia-fornita--coppia-misurata)
- [3. L'indicatore di salute: Torque Margin (Margine di Coppia)](#3-lindicatore-di-salute-torque-margin-margine-di-coppia)
  - [3.1. Come si interpreta nella manutenzione preventiva?](#31-come-si-interpreta-nella-manutenzione-preventiva)
- [4. Riepilogo con un esempio pratico in pista di volo](#4-riepilogo-con-un-esempio-pratico-in-pista-di-volo)
- [5. Quadro Teorico: La Coppia come Grandezza Fisica](#5-quadro-teorico-la-coppia-come-grandezza-fisica)
  - [5.1. Come si calcola fisicamente?](#51-come-si-calcola-fisicamente)
  - [5.2. Quali sono le sue Unità di Misura?](#52-quali-sono-le-sue-unit%C3%A0-di-misura)
  - [5.3. Come si calcola nel tuo progetto di Machine Learning?](#53-come-si-calcola-nel-tuo-progetto-di-machine-learning)
- [6. Concetti Fondamentali di Ingegneria del Motore](#6-concetti-fondamentali-di-ingegneria-del-motore)
  - [6.1. Il Concetto Base: cos'è la Coppia / Torque?](#61-il-concetto-base-cos%C3%A8-la-coppia--torque)
  - [6.2. I Due Valori in Gioco (Teoria vs. Realtà)](#62-i-due-valori-in-gioco-teoria-vs-realt%C3%A0)
  - [6.3. L'Indicatore Critico: *Torque Margin* (Margine di Coppia)](#63-lindicatore-critico-torque-margin-margine-di-coppia)
  - [6.4. Scenario Pratico di Operazione (Flusso di Inferenza)](#64-scenario-pratico-di-operazione-flusso-di-inferenza)

---

## 1. Il concetto base: cos'è la Coppia / Torque?

**Torque (inglese) / Coppia (italiano):**
È la forza di rotazione o torsione generata dal motore.
Nel caso dell'elicottero, è la forza con cui il motore fa ruotare l'albero che muove
le pale del rotore principale. Si misura in Newton-metro (N·m) o in percentuale (%)
della capacità massima del motore.

> **Esempio analogico:**
> Immagina di usare una chiave inglese per stringere un bullone.
> La forza con cui giri la chiave è la coppia (o torque).

---

## 2. I due valori in gioco (Teoria vs. Realtà)

Qui entra in gioco la sfida del PHM. Per capire se il motore è sano o danneggiato,
confrontiamo ciò che dovrebbe accadere in teoria con ciò che sta accadendo nella realtà.

### 2.1. Coppia desiderata / Torque Target / Target

* **Concetto:**
* Questi tre termini si riferiscono alla stessa cosa. È il valore di coppia ideale, teorico e di riferimento tecnico.
* È ciò che un motore perfetto, appena uscito di fabbrica, dovrebbe erogare in determinate
* condizioni ambientali specifiche (ad esempio, a una temperatura esterna di 15°C,
* al livello del mare e a una determinata velocità del compressore).
*
* **Nel Machine Learning:**
* Viene chiamato Target (obiettivo - output) perché è la variabile continua che il tuo modello di regressione
* cercherà di stimare/prevedere basandosi sui sensori puliti.
* **Esempio:** Il manuale dell'elicottero dice che oggi, con questo clima freddo, il motore
* dovrebbe erogare un 100% di coppia in modo ottimale. Quel 100% è il tuo Torque Target (Coppia desiderata).

### 2.2. Coppia fornita / Coppia misurata

* **Concetto:**
* Entrambi i termini si riferiscono alla realtà fisica attuale.
* È la coppia reale che i sensori (torquimetri) installati sul motore stanno misurando in quel preciso secondo.
* È ciò che il motore sta effettivamente riuscendo a fornire (fornita).
*
* **Nel Machine Learning:**
* Questo **non** viene previsto dal tuo modello; è un dato di input che arriva direttamente dalla telemetria dell'elicottero
* (la colonna `trq_measured`).
* **Esempio:**
* Guardi lo schermo della cabina e il sensore dice che il motore sta erogando un 95% di coppia. Quel 95% è la coppia misurata.

---

## 3. L'indicatore di salute: Torque Margin (Margine di Coppia)

* **Concetto:**
* È la differenza matematica tra la coppia ideale che ci aspettavamo e la coppia reale che il motore può offrire.
* È il KPI (indicatore chiave) principale per la diagnosi di guasti nelle turbine.

$$\text{Torque Margin} = \text{Torque Target (Coppia Desiderata)} - \text{Torque Misurato (Coppia Fornita)}$$

### 3.1. Come si interpreta nella manutenzione preventiva?

* **Margine vicino a zero o positivo controllato:**
* Il motore è sano. Ciò che gli viene richiesto teoricamente è ciò che sta erogando nella realtà.
*
* **Margine negativo o molto deviato:**
* Il motore ha un problema (degradazione per sporcizia nel compressore, usura delle palette, perdita di gas).
* Il motore cerca di dare potenza, ma a causa del danno, la coppia misurata cala o si discosta drasticamente
* rispetto al target teorico.

---

## 4. Riepilogo con un esempio pratico in pista di volo

Immagina che l'elicottero stia effettuando una missione di soccorso in montagna:

1. Il sistema legge le condizioni ambientali: la temperatura dell'aria (`oat`),
   la velocità dell'aria (`ias`) e i giri del motore (`ng`).
2. Con questi dati, il tuo algoritmo calcola la **Coppia desiderata (Torque Target):** *"Per quest'aria così rarefatta di montagna,
   un motore sano dovrebbe dare un 88% di coppia"*.
3. Il sensore dell'elicottero legge la **Coppia misurata (Coppia fornita):** registra che l'albero si muove solo con un 82% di coppia reale.
4. Il sistema calcola il **Torque Margin:** 88% - 82% = 6%. Uno scostamento di 6 punti.

**Diagnosi:** Poiché il margine si è allontanato dai limiti nominali, il classificatore di guasti (LightGBM) si attiva e marca lo stato del
motore come `faulty = 1`. È il momento di fare manutenzione preventiva prima che il motore si guasti in pieno volo.

---

## 5. Quadro Teorico: La Coppia come Grandezza Fisica

La coppia (o torque) non è un concetto astratto di software, è una grandezza fisica reale, misurabile e con unità molto chiare
nell'ingegneria. Ecco la suddivisione tecnica di come funziona come dimensione fisica:

### 5.1. Come si calcola fisicamente?

Nella fisica classica, la coppia è il risultato dell'applicazione di una forza a una determinata distanza da un punto di rotazione (un asse).
La sua formula base è:

$$\text{Torque} = \text{Forza} \times \text{Distanza}$$

* **Nell'elicottero:**
* La "forza" è generata dai gas caldi della combustione che spingono le palette della turbina,
* e la "distanza" è il raggio dell'albero motore che fa ruotare il rotore delle pale.

### 5.2. Quali sono le sue Unità di Misura?

Nella pratica e nei dataset di ingegneria (come quello di questa sfida), la coppia si esprime solitamente in due modi:

* **Unità del Sistema Internazionale (SI):** Il Newton-metro (N·m). Rappresenta la forza di 1 Newton applicata a 1 metro di distanza dall'asse.
* **Percentuale (%):** Nell'aviazione e nella robotica industriale, i sensori solitamente normalizzano la misura.
* Invece di mostrare al pilota un numero enorme come 3500 N·m, il computer di bordo lo mostra come percentuale della coppia massima consentita
* (ad esempio: 95% di coppia). È così che appare nel tuo dataset PHM.

### 5.3. Come si calcola nel tuo progetto di Machine Learning?

Qui è dove la fisica si combina con il tuo codice:

* **La Misura (Telemetria):** L'elicottero ha un sensore fisico reale (un torquimetro) sull'albero. Questo sensore misura la torsione del metallo mentre ruota e salva quel dato nel tuo dataset. Questa è la *coppia misurata* (`trq_measured`).
* **Il Calcolo (Il Tuo Modello):** La *coppia desiderata* (`trq_target`) **non** può essere misurata con un sensore diretto perché è un valore ideale. Viene calcolata tramite un'equazione fisica (o tramite il tuo modello di regressione lineare/polinomiale) basata sullo stato ambientale:

$$\text{trq\_target} = f(\text{oat}, \text{mgt}, \text{pa}, \text{ias}, \text{np}, \text{ng})$$

> **Conclusione:** Alla fine, sottrai le due dimensioni fisiche (Teorica meno Reale) e ottieni il **Torque Margin**, che ti dice quanta forza sta perdendo il robot o l'aeromobile a causa dell'usura meccanica.

---

## 6. Concetti Fondamentali di Ingegneria del Motore

### 6.1. Il Concetto Base: cos'è la Coppia / Torque?

Il **Torque** (termine inglese) o ***Coppia*** (termine italiano) rappresenta la forza di rotazione o torsione generata da una pianta motrice.

* **Applicazione Aeronautica:** Nel contesto di un elicottero, definisce la forza angolare con cui il motore impulsa l'albero di trasmissione per far ruotare le pale del rotore principale.
* **Unità di Misura:** Si esprime formalmente in Newton-metro (N·m) o in modo relativo come percentuale (**10%** - **100%**) rispetto alla capacità operativa massima di progetto del motore.

> **Analogia Meccanica:** Se usi una chiave inglese per regolare un bullone, la forza fisica applicata sul manico per farlo ruotare equivale esattamente alla coppia (o *torque*).

### 6.2. I Due Valori in Gioco (Teoria vs. Realtà)

Il cuore della sfida PHM (*Prognostics and Health Management*) consiste nel contrastare il comportamento teorico atteso rispetto alle prestazioni fisiche reali dell'attivo per diagnosticare con precisione il suo stato di salute.

**A. Coppia Desiderata / Torque Target / Target (Teoria)**

* **Concetto:** Definisce il valore di coppia ideale e di riferimento tecnico. Rappresenta la potenza che un motore nominale (in condizioni perfette di fabbrica) dovrebbe fornire sotto parametri ambientali e cinematici specifici.
* **Formula Ambientale:** Il suo valore è determinato da variabili come la temperatura esterna (`oat`), la pressione atmosferica (`pa`) e il regime di giri del compressore.
* **Ruolo nel Machine Learning:** Agisce come variabile continua obiettivo (**Target**). Il modello di regressione addestrato si occupa di stimare analiticamente questo valore teorico a partire dai dati puliti dei sensori.
* **Esempio:** Se i manuali di ingegneria stabiliscono che con il clima attuale il motore opererebbe ottimamente al **100%** della sua capacità, quel valore costituisce la *Coppia Desiderata*.

**B. Coppia Fornita / Coppia Misurata (Realtà)**

* **Concetto:** Rappresenta la realtà fisica del sistema in tempo di esecuzione. È la coppia reale istantanea misurata dai torquimetri fisici integrati nella trasmissione dell'elicottero.
* **Ruolo nel Machine Learning:** Questo valore **non viene previsto**. Agisce strettamente come una variabile di input primaria (`input`) proveniente dalla telemetria dell'attivo (la colonna `trq_measured`).
* **Esempio:** Se la strumentazione della cabina registra che l'albero di trasmissione eroga un **95%** di forza torsionale, quel dato rappresenta la *coppia misurata*.

### 6.3. L'Indicatore Critico: *Torque Margin* (Margine di Coppia)

Il **Torque Margin** è l'indicatore chiave di prestazione (KPI) fondamentale utilizzato nell'ingegneria di manutenzione predittiva per valutare la degradazione delle turbine a gas. Si calcola mediante la differenza matematica tra l'aspettativa teorica e l'erogazione reale:

$$\text{Torque Margin} = \text{Torque Target (Coppia Desiderata)} - \text{Torque Misurato (Coppia Fornita)}$$

**Criteri di Diagnosi nella Manutenzione Preventiva**

* **Margine prossimo a zero o positivo controllato:** Indica uno stato di salute **Nominale (Sano)**. La risposta fisica del motore è allineata con le curve di progetto teorico della pianta motrice.
* **Margine negativo o fortemente deviato:** Indica uno stato di **Guasto (Anomalo)**. Evidenzia una perdita di efficienza termodinamica dovuta a cause fisiche come la degradazione per sporcizia nel compressore, usura delle palette della turbina o perdite nel percorso del gas. Il motore consuma più risorse cercando di soddisfare la domanda, ma la *coppia misurata* scende al di sotto del riferimento di progetto.

### 6.4. Scenario Pratico di Operazione (Flusso di Inferenza)

Per illustrare il funzionamento del pipeline durante una missione di volo in condizioni di alta esigenza (volo in montagna), il sistema esegue la seguente elaborazione sequenziale:

```text
[ Variabili Ambientali ] ────> [ Modello di Regressione ] ────> Calcola: Coppia Desiderata (88%)
  (oat, pa, ias, ng, np)                                                  │
                                                                          ▼
[ Telemetria Sensore ]   ────> Legge: Coppia Misurata (82%)   ────> [ Calcolo del Margine ]
                                                                          │
                                                                          ▼
                                                                  Torque Margin = 6%
                                                                          │
                                                                          ▼
                                 [ Classificatore LightGBM ] ───> Diagnosi: FAULTY = 1

```

1. **Lettura delle Condizioni:** Il sistema acquisisce le variabili ambientali: temperatura dell'aria (`oat`), velocità relativa (`ias`) e i giri del nucleo del motore (`ng`).
2. **Stima Teorica:** L'algoritmo di regressione elabora i dati ambientali e determina: *"Per questa altitudine e densità dell'aria, un motore sano deve generare una Coppia Desiderata del **88%**."*
3. **Misura della Telemetria:** Il sensore fisico registra una *Coppia Misurata* reale del **82%**.
4. **Valutazione del KPI:** Il sistema calcola lo scostamento: 88% - 82% = 6% di perdita di margine.
5. **Emissione della Diagnosi:** Rilevando che il margine ha violato i limiti geometrici tollerabili di operazione nominale, il modello classificatore si attiva, parametrizzando lo stato dell'attivo come `faulty = 1`. Ciò consente di programmare un'ispezione preventiva prima di compromettere la sicurezza del volo.
