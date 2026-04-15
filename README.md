# Intelligente Verwaltungssteuerung

Eine KI-gestützte Webanwendung zur automatischen Klassifikation von Bürgeranfragen, automatisierten E-Mail-Weiterleitung an zuständige Ämter sowie datenbasierter Auslastungsprognose für öffentliche Verwaltungen.

**Live Demo:** [smartgov.streamlit.app](https://smartgov-uign8hchnsxbtz93xpxxri.streamlit.app) 

---

## Problemstellung

Öffentliche Verwaltungen erhalten täglich hunderte Bürgeranfragen per E-Mail, die manuell gelesen, bewertet und an das richtige Amt weitergeleitet werden müssen. Das kostet Zeit, führt zu Fehlweiterleitungen und belastet das Personal.

## Lösung

Prozesse werden vollständig automatisiert

- Eingehende Anfrage wird per Machine Learning klassifiziert
- Das zuständige Amt wird in Echtzeit identifiziert
- Eine fertige Weiterleitungs-E-Mail wird automatisch generiert
- Auslastungsprognosen helfen bei der Personalplanung

---

## Features

| Feature | Beschreibung |
|---|---|
| **Automatische Klassifikation** | TF-IDF + Logistic Regression ordnet jede Anfrage einem von 10 Ämtern zu |
| **E-Mail-Generierung** | Fertige Weiterleitungsmail wird automatisch erstellt und kann exportiert werden |
| **Auslastungsprognose** | Ridge-Regression prognostiziert Besucheraufkommen für jede Stunde der kommenden Woche |
| **Live-Training** | Beide Modelle können ohne Code-Kenntnisse direkt in der App nachtrainiert werden |
| **Interaktive Visualisierung** | Heatmap und Liniendiagramm (Altair) mit Ampelfarbsystem |
| **Anfragen-Log** | Alle Vorgänge werden protokolliert und können als CSV exportiert werden |

---

## Technologie

```
Python · Streamlit · scikit-learn · Altair · Pandas · NumPy
```

**Machine Learning Modelle:**
- Klassifikation: TF-IDF Vectorizer (Bigrams) + Logistic Regression
- Auslastungsprognose: Ridge Regression mit Saison- und Stundengewichtung

---

## Installation & Start

```bash
# Repository klonen
git clone https://github.com/christianm38/smartgov-ai.git
cd smartgov-ai

# Abhängigkeiten installieren
pip install -r requirements.txt

# App starten
streamlit run smartgov_app.py
```

Die App öffnet sich automatisch unter `http://localhost:8501`

---

## Zuständige Ämter

Die App klassifiziert Anfragen in folgende Kategorien:

`Bürgeramt` · `Tiefbauamt` · `Abfallwirtschaft` · `Umweltamt` · `Bauamt` · `Bildung und Betreuung` · `Ordnungsamt` · `Digitalisierung` · `Gebäudewirtschaft` · `Sozialamt`

---

## Projektstruktur

```
smartgov-ai/
├── smartgov_app.py      # Hauptanwendung
├── requirements.txt     # Abhängigkeiten
└── README.md            # Projektbeschreibung
```

---

## Hintergrund

Dieses Projekt entstand als Prototyp für den Einsatz von Machine Learning in der öffentlichen Verwaltung. Ziel war es, einen realen Anwendungsfall mit minimalem technischen Aufwand zu lösen und gleichzeitig eine intuitive Benutzeroberfläche zu schaffen, die auch ohne technische Vorkenntnisse bedienbar ist.

---

## Autor

**Christian Mann**
· [E-Mail](mailto:mannchristian38@gmail.com)
