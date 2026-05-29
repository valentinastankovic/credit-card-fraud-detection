# Credit Card Fraud Detection

Projekat iz predmeta Duboko učenje i neuronske mreže.  
Autori: Valentina Stanković, Nikola Todorović

---

## 1. Opis problema

Detekcija prevara kreditnim karticama predstavlja jedan od ključnih problema u finansijskoj industriji. Cilj projekta je izgradnja modela koji može da razlikuje legitimne transakcije od prevara na osnovu istorijskih podataka.

Glavni izazov je ekstremna neravnoteža klasa — prevare čine svega 0.17% svih transakcija, što znači da model koji uvek predviđa "legitimna transakcija" postiže tačnost od 99.83%, ali ne hvata nijednu prevaru. Zbog toga su potrebne posebne tehnike za rad sa neuravnoteženim datasetima.

---

## 2. Podaci

**Izvor:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Struktura dataseta:**
- 284,807 transakcija kreditnim karticama
- 31 kolona: `Time`, `V1`–`V28`, `Amount`, `Class`
- `Class`: 0 = legitimna transakcija, 1 = prevara
- 284,315 legitimnih transakcija (99.83%) i 492 prevara (0.17%)

**Napomena:** Kolone V1–V28 su rezultat PCA transformacije originalnih podataka zbog zaštite privatnosti korisnika. Jedine netransformisane kolone su `Time` (vreme transakcije u sekundama) i `Amount` (iznos transakcije).

**Preprocesiranje:**
- Provera nedostajućih vrednosti — nije pronađena nijedna
- Skaliranje kolona `Amount` i `Time` pomoću `StandardScaler` (mean ≈ 0, std ≈ 1)
- Podela na trening (80%) i test (20%) set uz stratifikaciju
- Balansiranje klasa na trening setu pomoću **SMOTE** tehnike

**SMOTE (Synthetic Minority Oversampling Technique):**  
SMOTE generiše sintetičke primere prevara interpolacijom između postojećih primera, čime izjednačava broj klasa. Nakon SMOTE-a trening set sadrži 227,451 legitimnih i 227,451 sintetičkih primera prevara. SMOTE se primenjuje **isključivo na trening setu** kako bi test set ostao realan.

---

## 3. Arhitektura modela

Korišćen je **MLPClassifier** (Multi-Layer Perceptron) iz biblioteke scikit-learn.

**Arhitektura baseline modela:**
- Ulazni sloj: 30 feature-a
- Skriveni sloj 1: 64 neurona, aktivacija ReLU
- Skriveni sloj 2: 32 neurona, aktivacija ReLU
- Skriveni sloj 3: 16 neurona, aktivacija ReLU
- Izlazni sloj: 1 neuron (binarna klasifikacija)
- Learning rate: 0.001
- Maksimalan broj iteracija: 50

---

## 4. Trening

Model je treniran na SMOTE-balansiranom trening setu (454,902 primera).  
Korišćen je Adam optimizer sa early stopping mehanizmom — trening se zaustavlja kada loss ne poboljša za više od `tol=0.0001` tokom 10 uzastopnih epoha.

---

## 5. Analiza osetljivosti i hiperparametarska optimizacija

Ispitano je 24 kombinacije hiperparametara:

| Hiperparametar | Testirane vrednosti |
|---|---|
| `hidden_layer_sizes` | (32,16), (64,32), (64,32,16), (128,64,32) |
| `activation` | relu, tanh |
| `learning_rate_init` | 0.001, 0.01, 0.0001 |

**Rezultati optimizacije:**

| Model | hidden_layer_sizes | activation | lr | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Baseline | (64, 32, 16) | relu | 0.001 | 0.80 | 0.965 |
| Optimizovani | (128, 64, 32) | relu | 0.001 | 0.82 | 0.970 |

Optimizovani model sa arhitekturom (128, 64, 32) postiže blago poboljšanje F1 skora uz zadržavanje visoke vrednosti AUC-ROC.

---

## 6. Rezultati evaluacije

**MLP neuronska mreža (optimizovani model):**

| Metrika | Legitimna | Prevara |
|---|---|---|
| Precision | 1.00 | 0.77 |
| Recall | 1.00 | 0.84 |
| F1-score | 1.00 | 0.80 |

- **AUC-ROC:** 0.965
- **Propuštenih prevara (False Negative):** 14 od 98
- **Lažnih uzbuna (False Positive):** 36

**Poređenje sa logističkom regresijom:**

| Model | Precision (prevara) | Recall (prevara) | F1 (prevara) | AUC-ROC |
|---|---|---|---|---|
| Logistička regresija | 0.06 | 0.92 | 0.11 | 0.970 |
| MLP neuronska mreža | 0.77 | 0.84 | 0.80 | 0.965 |

Iako logistička regresija postiže sličan AUC-ROC, njen precision od svega 0.06 znači da od svakih 100 transakcija označenih kao prevara, samo 6 su stvarne prevare — što bi u praksi značilo ogromne troškove za banku zbog blokiranja legitimnih transakcija.

---

## 7. Diskusija

Model postiže odlične rezultate sa AUC-ROC od 0.965 i F1 skorom 0.80 za klasu prevara. Ključni izazovi bili su:

- **Neuravnoteženost klasa** — rešena SMOTE tehnikom
- **Izbor metrike** — accuracy nije relevantna metrika za ovaj problem; fokus je na F1 skoru i AUC-ROC
- **Kompromis precision/recall** — u praksi je recall važniji (ne propustiti prevaru), ali previsok recall uz nizak precision dovodi do previše lažnih uzbuna

Logistička regresija postiže visok recall ali izrazito nizak precision, što je nepraktično za realne sisteme. MLP neuronska mreža pronalazi bolji balans između ove dve metrike.

---

## 8. Zaključak

Uspešno je izgrađen model za detekciju prevara kreditnih kartica koji postiže AUC-ROC 0.965 i F1 skor 0.80. MLP neuronska mreža pokazuje značajno bolje performanse od logističke regresije kada se uzme u obzir balans između precision i recall.

Moguća poboljšanja uključuju primenu naprednijih arhitektura poput LSTM mreža ili XGBoost algoritma koji se često koristi u industriji za ovaj tip problema.

---

## Licenca

MIT License