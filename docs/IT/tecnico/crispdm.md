
## Panoramica delle Fasi CRISP-DM

1. **Comprensione del Business**: Definire gli obiettivi
2. **Comprensione dei Dati**: EDA, profilazione
3. **Preparazione dei Dati**: Feature engineering, pulizia
4. **Modellazione**: Addestrare modelli ML
5. **Valutazione**: Metriche, validazione
6. **Distribuzione**: API, monitoraggio

# Metodologia del Progetto di Data Science

| FASE | SOTTOFASE | REGOLA MENTALE | OBIETTIVO / PROBLEMA |
| :--- | :--- | :--- | :--- |
| FASE 2: COMPRENSIONE DEI DATI | **2.1 ACQUISIZIONE DATI** | Ottenere i dati e identificarne la fonte | Dati non disponibili, sparsi o isolati in più fonti. |
| FASE 2: COMPRENSIONE DEI DATI | **2.2 DESCRIZIONE DATI** | Comprendere struttura e contenuto di base | Mancanza di chiarezza su colonne, tipi di dati e distribuzioni. |
| FASE 2: COMPRENSIONE DEI DATI | **2.3 verifica_qualità_dati** | Individuare problemi di qualità (senza correggere ancora) | Presenza di errori, outlier o incongruenze di scala sconosciuta. |
| FASE 2: COMPRENSIONE DEI DATI | **2.4 ESPLORAZIONE DATI** | Scoprire pattern e relazioni | Correlazioni o comportamenti sconosciuti tra le variabili. |
| FASE 3: PREPARAZIONE DEI DATI | **3.1 SELEZIONE DATI** | Decidere COSA includere e COSA escludere | Definire il perimetro; evitare data leakage e rumore strutturale. |
| FASE 3: PREPARAZIONE DEI DATI | **3.2 PULIZIA DATI** | Correggere errori e rumore senza aggiungere informazioni | Migliorare la qualità intrinseca correggendo le incongruenze. |
| FASE 3: PREPARAZIONE DEI DATI | **3.3 TRASFORMAZIONE DATI** | Cambiare rappresentazione per un migliore apprendimento | Riformattare le feature per un apprendimento efficiente del modello. |
| FASE 3: PREPARAZIONE DEI DATI | **3.4 INTEGRAZIONE DATI** | Unire più fonti in un dataset coerente | Unificare più fonti in un dataset coerente e allineato. |
| FASE 3: PREPARAZIONE DEI DATI | **3.5 FORMATTAZIONE DATI** | Ultima rifinitura per l'algoritmo | Preparare i dati per il consumo ML senza alterarne il significato. |
| FASE 4: MODELLAZIONE DATI | **4.1 SELEZIONE TECNICA** | Scegliere l'algoritmo giusto | Selezionare il modello in base al tipo di problema e alle assunzioni. |
| FASE 4: MODELLAZIONE DATI | **4.2 COSTRUZIONE MODELLO** | Addestrare il modello | Addestrare il modello e trovare il set di parametri ottimale. |
| FASE 4: MODELLAZIONE DATI | **4.3 PROGETTAZIONE TEST** | Definire come dimostrare che funziona | Progettare una strategia di valutazione per prevenire il leakage. |
| FASE 4: MODELLAZIONE DATI | **4.4 VALUTAZIONE MODELLO** | Valutare le prestazioni | Misurare obiettivamente la qualità in base al problema specifico. |
| FASE 5: VALUTAZIONE E INTERPRETAZIONE | **5.1 ESTRAZIONE CONOSCENZA** | Comprendere COSA e PERCHÉ ha imparato | Interpretare i risultati, l'importanza delle feature o il significato dei cluster. |
| FASE 5: VALUTAZIONE E INTERPRETAZIONE | **5.2 VALUTAZIONE BUSINESS** | Valutare il VERO valore per il business | Passare dalle metriche tecniche alla riduzione dei costi o al profitto. |
| FASE 5: VALUTAZIONE E INTERPRETAZIONE | **5.3 REVISIONE PROCESSO** | Revisionare la metodologia | Identificare errori metodologici o decisioni ingiustificate. |
| FASE 5: VALUTAZIONE E INTERPRETAZIONE | **5.4 DECISIONE PROSSIMI PASSI** | Decidere il futuro del modello | Determinare se distribuire, iterare o abbandonare il progetto. |

