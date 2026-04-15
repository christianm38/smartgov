import streamlit as st
import pandas as pd
import numpy as np
import datetime
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

# ══════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════

st.set_page_config(page_title="Smarte Verwaltung", page_icon=None, layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .main { background-color: #f5f6fa; }
    .gov-header {
        background: linear-gradient(135deg, #0B1F3A 0%, #1a3a5c 100%);
        color: white; padding: 32px 40px 24px;
        border-radius: 12px; margin-bottom: 28px;
    }
    .gov-header h1 { margin: 0; font-size: 28px; font-weight: 700; letter-spacing: -0.5px; }
    .gov-header p  { margin: 6px 0 0; opacity: 0.65; font-size: 14px; }
    .card {
        background: white; border-radius: 10px; padding: 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 16px;
    }
    .card-title { font-size: 13px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.8px; color: #666; margin-bottom: 6px; }
    .card-value { font-size: 32px; font-weight: 700; color: #0B1F3A; }
    .result-box {
        background: #f0f7ff; border-left: 4px solid #1a6fc4;
        border-radius: 6px; padding: 18px 20px; margin: 16px 0;
    }
    .result-box .amt  { font-size: 20px; font-weight: 700; color: #0B1F3A; }
    .result-box .conf { font-size: 13px; color: #555; margin-top: 2px; }
    .email-box {
        background: #ffffff; border: 1px solid #dde3ec;
        border-radius: 8px; padding: 20px 24px;
        font-size: 14px; line-height: 1.7; color: #222;
    }
    .email-box .subject {
        font-weight: 700; font-size: 15px;
        border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 14px;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 14px; }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# KONSTANTEN
# ══════════════════════════════════════════════════════════════

ÄMTER = [
    "Bürgeramt",
    "Tiefbauamt",
    "Abfallwirtschaft",
    "Umweltamt",
    "Bauamt",
    "Bildung und Betreuung",
    "Ordnungsamt",
    "Digitalisierung",
    "Gebäudewirtschaft",
    "Sozialamt",
]

AMT_EMAIL = {
    "Bürgeramt":            "buergeramt@stadt.de",
    "Tiefbauamt":           "tiefbau@stadt.de",
    "Abfallwirtschaft":     "abfall@stadt.de",
    "Umweltamt":            "umwelt@stadt.de",
    "Bauamt":               "bauamt@stadt.de",
    "Bildung und Betreuung":"bildung@stadt.de",
    "Ordnungsamt":          "ordnung@stadt.de",
    "Digitalisierung":      "digital@stadt.de",
    "Gebäudewirtschaft":    "gebaeude@stadt.de",
    "Sozialamt":            "sozial@stadt.de",
}

DAYS   = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"]
MONATE = ["Januar","Februar","März","April","Mai","Juni",
          "Juli","August","September","Oktober","November","Dezember"]
HOURS  = list(range(8, 19))

# Saisonfaktoren pro Monat
SAISON = {1:0.95, 2:1.0, 3:1.1, 4:1.1, 5:1.05, 6:1.0,
          7:0.85, 8:0.80, 9:1.1, 10:1.1, 11:1.0, 12:0.90}

# Stundenfaktoren
def stunden_faktor(h):
    if h in [8, 9]:     return 1.35
    elif h in [10, 11]: return 1.15
    elif h == 12:       return 0.75
    elif h in [13, 14]: return 0.90
    elif h in [15, 16]: return 1.05
    else:               return 0.65

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════

DEFAULT_TEXTS = [
    "Personalausweis beantragen", "Ausweis verloren", "Reisepass neu",
    "Ummeldung Wohnsitz", "Führungszeugnis", "Meldebescheinigung",
    "Straßenlaterne defekt", "Schlagloch melden", "Gehweg beschädigt",
    "Bordstein reparieren", "Ampel defekt",
    "Müll nicht abgeholt", "Sperrmüll anmelden", "Recyclinghof Öffnungszeiten",
    "Gelbe Tonne fehlt", "Containerstandort",
    "Baum umgestürzt", "Baumfällung beantragen", "Schädlingsbefall",
    "Gewässerverschmutzung", "Lärmschutz",
    "Bauantrag stellen", "Baugenehmigung", "Abbruchgenehmigung",
    "Nutzungsänderung", "Bebauungsplan",
    "Kita Platz gesucht", "Schulanmeldung", "Hortplatz beantragen",
    "Schülerbeförderung", "Schulbezirk",
    "Falschparker melden", "Lärmbelästigung", "Graffiti melden",
    "Versammlungsanmeldung", "Hundesteuern",
    "Online-Dienste Verwaltung", "Digitaler Ausweis", "E-Government Portal",
    "App Stadtservices", "Digitale Ummeldung",
    "Rathaus Heizung defekt", "Schulgebäude Reparatur", "Stadthaus Aufzug defekt",
    "Reinigung öffentliche Gebäude", "Mietvertrag Stadtimmobilie",
    "Sozialhilfe beantragen", "Wohngeld Antrag", "Grundsicherung",
    "Pflegeberatung", "Obdachlosenunterkunft",
]
DEFAULT_LABELS = [
    "Bürgeramt","Bürgeramt","Bürgeramt","Bürgeramt","Bürgeramt","Bürgeramt",
    "Tiefbauamt","Tiefbauamt","Tiefbauamt","Tiefbauamt","Tiefbauamt",
    "Abfallwirtschaft","Abfallwirtschaft","Abfallwirtschaft","Abfallwirtschaft","Abfallwirtschaft",
    "Umweltamt","Umweltamt","Umweltamt","Umweltamt","Umweltamt",
    "Bauamt","Bauamt","Bauamt","Bauamt","Bauamt",
    "Bildung und Betreuung","Bildung und Betreuung","Bildung und Betreuung",
    "Bildung und Betreuung","Bildung und Betreuung",
    "Ordnungsamt","Ordnungsamt","Ordnungsamt","Ordnungsamt","Ordnungsamt",
    "Digitalisierung","Digitalisierung","Digitalisierung","Digitalisierung","Digitalisierung",
    "Gebäudewirtschaft","Gebäudewirtschaft","Gebäudewirtschaft","Gebäudewirtschaft","Gebäudewirtschaft",
    "Sozialamt","Sozialamt","Sozialamt","Sozialamt","Sozialamt",
]

# Auslastungs-Trainingsdaten: Features [tag(0-4), monat(1-12)] -> Besucher
BASE_LOAD = {0: 130, 1: 90, 2: 70, 3: 110, 4: 150}
DEFAULT_LOAD_X, DEFAULT_LOAD_Y = [], []
for tag, base in BASE_LOAD.items():
    for monat in range(1, 13):
        val = int(base * SAISON[monat] + np.random.randint(-6, 6))
        DEFAULT_LOAD_X.append([tag, monat])
        DEFAULT_LOAD_Y.append(val)

def init():
    for k, v in {
        "cls_texts":  list(DEFAULT_TEXTS),
        "cls_labels": list(DEFAULT_LABELS),
        "load_X":     list(DEFAULT_LOAD_X),
        "load_Y":     list(DEFAULT_LOAD_Y),
        "log":        [],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()

# ══════════════════════════════════════════════════════════════
# MODELLE
# ══════════════════════════════════════════════════════════════

def train_classifier():
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    X   = vec.fit_transform(st.session_state.cls_texts)
    clf = LogisticRegression(max_iter=2000, C=2.0)
    clf.fit(X, st.session_state.cls_labels)
    return vec, clf

def train_load_model():
    X = np.array(st.session_state.load_X)
    y = np.array(st.session_state.load_Y)
    m = Ridge(alpha=1.0)
    m.fit(X, y)
    return m

vec, clf   = train_classifier()
load_model = train_load_model()

def classify(text):
    X     = vec.transform([text])
    label = clf.predict(X)[0]
    conf  = round(float(max(clf.predict_proba(X)[0])) * 100, 1)
    return label, conf

def predict_day(tag_idx, monat):
    return max(10, int(load_model.predict([[tag_idx, monat]])[0]))

def predict_hour(base, hour):
    return max(1, int(base * stunden_faktor(hour)))

# ══════════════════════════════════════════════════════════════
# HEADER + KPIs
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="gov-header">
    <h1>Smarte Verwaltung — Intelligente Verwaltungssteuerung</h1>
    <p>Automatische Zuordnung von Bürgeranfragen &middot; E-Mail-Weiterleitung &middot; Auslastungsprognose</p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
next_monday = datetime.date.today() + datetime.timedelta(days=(7 - datetime.date.today().weekday()))
today_log   = [a for a in st.session_state.log if a["datum"].startswith(str(datetime.date.today()))]
with k1: st.markdown(f'<div class="card"><div class="card-title">Anfragen gesamt</div><div class="card-value">{len(st.session_state.log)}</div></div>', unsafe_allow_html=True)
with k2: st.markdown(f'<div class="card"><div class="card-title">Anfragen heute</div><div class="card-value">{len(today_log)}</div></div>', unsafe_allow_html=True)
with k3: st.markdown(f'<div class="card"><div class="card-title">Trainingsdaten</div><div class="card-value">{len(st.session_state.cls_texts)}</div></div>', unsafe_allow_html=True)
with k4: st.markdown(f'<div class="card"><div class="card-title">Prognose KW</div><div class="card-value">{next_monday.isocalendar()[1]}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "Anfrage & E-Mail",
    "Auslastungsprognose",
    "Live-Training",
    "Anfragen-Log",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — ANFRAGE & E-MAIL
# ══════════════════════════════════════════════════════════════

with tab1:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("#### Bürgeranfrage eingeben")
        anfrage  = st.text_area("Anfrage", label_visibility="collapsed",
                                placeholder="Beschreiben Sie Ihr Anliegen...", height=130)
        absender = st.text_input("Absender E-Mail (optional)", placeholder="buerger@mail.de")

        if st.button("Analyse starten", type="primary", use_container_width=True):
            if anfrage.strip():
                amt, conf = classify(anfrage)
                st.session_state.update({
                    "last_amt": amt, "last_conf": conf,
                    "last_anfrage": anfrage, "last_sender": absender,
                })
                st.session_state.log.append({
                    "datum":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "anfrage":  anfrage,
                    "amt":      amt,
                    "absender": absender or "—",
                })
            else:
                st.warning("Bitte eine Anfrage eingeben.")

        if "last_amt" in st.session_state:
            amt = st.session_state["last_amt"]
            st.markdown(f"""
            <div class="result-box">
                <div class="amt">{amt}</div>
                <div class="conf">Zuständig &middot; {AMT_EMAIL[amt]}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown("#### E-Mail Vorschau")
        if "last_amt" in st.session_state:
            amt    = st.session_state["last_amt"]
            anf    = st.session_state["last_anfrage"]
            sender = st.session_state["last_sender"] or "nicht angegeben"
            today  = datetime.date.today().strftime("%d.%m.%Y")
            body   = (
                f"Sehr geehrte Damen und Herren,\n\n"
                f"hiermit leiten wir Ihnen eine eingegangene Bürgeranfrage zur weiteren Bearbeitung weiter.\n\n"
                f"Datum: {today}\nAbsender: {sender}\n\nAnfrage:\n{anf}\n\n"
                f"Bitte nehmen Sie Kontakt mit dem Bürger auf und bearbeiten Sie das Anliegen "
                f"gemäß der internen Richtlinien.\n\nMit freundlichen Grüßen\nSmarte Verwaltung Eingangssteuerung"
            )
            st.markdown(f"""
            <div class="email-box">
                <div class="subject">An: {AMT_EMAIL[amt]}<br>Betreff: Neue Bürgeranfrage — {today}</div>
                {body.replace(chr(10), "<br>")}
            </div>
            """, unsafe_allow_html=True)
            st.download_button(
                "E-Mail exportieren (.txt)",
                data=f"An: {AMT_EMAIL[amt]}\nBetreff: Neue Bürgeranfrage — {today}\n\n{body}",
                file_name=f"weiterleitung_{amt}_{today}.txt",
                mime="text/plain", use_container_width=True,
            )
        else:
            st.markdown('<div class="email-box" style="color:#aaa;min-height:200px;">Nach der Analyse wird hier die E-Mail Vorschau angezeigt.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — AUSLASTUNGSPROGNOSE
# ══════════════════════════════════════════════════════════════

with tab2:
    st.markdown("#### Auslastungsprognose — Kommende Woche")

    col_sel, _ = st.columns([1, 2])
    with col_sel:
        monat_name = st.selectbox("Monat", MONATE, index=next_monday.month - 1)
    monat = MONATE.index(monat_name) + 1

    # Tagesprognosen
    tages_werte = {day: predict_day(i, monat) for i, day in enumerate(DAYS)}

    # ── Stundenheatmap: alle Tage × alle Stunden ──────────────
    rows = []
    for i, day in enumerate(DAYS):
        base = tages_werte[day]
        for h in HOURS:
            rows.append({
                "Tag":     day,
                "Stunde":  f"{h:02d}:00",
                "Stunde_n": h,
                "Besucher": predict_hour(base, h),
            })
    df_heat = pd.DataFrame(rows)

    heatmap = (
        alt.Chart(df_heat)
        .mark_rect(cornerRadius=3)
        .encode(
            x=alt.X("Stunde:O",
                    sort=[f"{h:02d}:00" for h in HOURS],
                    axis=alt.Axis(title="Uhrzeit", labelAngle=0, titleFontSize=12)),
            y=alt.Y("Tag:O",
                    sort=DAYS,
                    axis=alt.Axis(title=None, titleFontSize=12)),
            color=alt.Color(
                "Besucher:Q",
                scale=alt.Scale(scheme="redyellowgreen", reverse=True),
                legend=alt.Legend(title="Besucher"),
            ),
            tooltip=["Tag", "Stunde", "Besucher"],
        )
        .properties(height=230, title="Besucherprognose — Stunde × Wochentag")
        .configure_title(fontSize=14, anchor="start", color="#0B1F3A")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(heatmap, use_container_width=True)

    st.markdown("---")

    # ── Liniendiagramm: alle Tage übereinander ────────────────
    line = (
        alt.Chart(df_heat)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("Stunde:O",
                    sort=[f"{h:02d}:00" for h in HOURS],
                    axis=alt.Axis(title="Uhrzeit", labelAngle=0)),
            y=alt.Y("Besucher:Q", axis=alt.Axis(title="Besucher")),
            color=alt.Color("Tag:N",
                            sort=DAYS,
                            scale=alt.Scale(scheme="tableau10"),
                            legend=alt.Legend(title="Wochentag")),
            tooltip=["Tag", "Stunde", "Besucher"],
        )
        .properties(height=300, title="Tagesverlauf nach Wochentag")
        .configure_title(fontSize=14, anchor="start", color="#0B1F3A")
        .configure_view(strokeWidth=0)
        .interactive()
    )
    st.altair_chart(line, use_container_width=True)

    st.markdown("---")

    # ── Tagessummen-Tabelle ───────────────────────────────────
    col_t, col_i = st.columns([2, 1])
    with col_t:
        df_prog = pd.DataFrame({
            "Tag":      DAYS,
            "Besucher": [tages_werte[d] for d in DAYS],
        })
        def style_row(row):
            if row["Besucher"] > 140: return ["background-color:#f87171; color:#7f1d1d; font-weight:600"] * len(row)
            elif row["Besucher"] > 100: return ["background-color:#fbbf24; color:#78350f; font-weight:600"] * len(row)
            else: return ["background-color:#4ade80; color:#14532d; font-weight:600"] * len(row)
        st.dataframe(df_prog.style.apply(style_row, axis=1),
                     use_container_width=True, hide_index=True)
    with col_i:
        peak = max(tages_werte, key=tages_werte.get)
        low  = min(tages_werte, key=tages_werte.get)
        st.markdown(f"""
        <div class="card" style="margin-top:0">
            <div class="card-title">Spitzentag</div>
            <div class="card-value" style="font-size:20px">{peak}</div>
            <div style="color:#666;font-size:13px">{tages_werte[peak]} Besucher erwartet</div>
        </div>
        <div class="card">
            <div class="card-title">Ruhigster Tag</div>
            <div class="card-value" style="font-size:20px">{low}</div>
            <div style="color:#666;font-size:13px">{tages_werte[low]} Besucher erwartet</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — LIVE-TRAINING
# ══════════════════════════════════════════════════════════════

with tab3:
    t_cls, t_load = st.tabs(["Klassifikation trainieren", "Auslastung trainieren"])

    with t_cls:
        col_a, col_b = st.columns([1, 1], gap="large")
        with col_a:
            st.markdown("#### Neuen Eintrag hinzufügen")
            new_text  = st.text_input("Anfrage-Beispiel", placeholder="z.B. Parkausweis beantragen")
            new_label = st.selectbox("Zuständiges Amt", ÄMTER, key="cls_label")
            if st.button("Hinzufügen", type="primary", key="add_cls"):
                if new_text.strip():
                    st.session_state.cls_texts.append(new_text.strip())
                    st.session_state.cls_labels.append(new_label)
                    vec, clf = train_classifier()
                    st.success(f"Eintrag hinzugefügt: {new_label}")
                    st.rerun()
                else:
                    st.warning("Bitte Text eingeben.")
            st.markdown("---")
            df_exp = pd.DataFrame({"Anfrage": st.session_state.cls_texts, "Amt": st.session_state.cls_labels})
            st.download_button("Trainingsdaten exportieren",
                data=df_exp.to_csv(index=False).encode("utf-8"),
                file_name="cls_training.csv", mime="text/csv", use_container_width=True)
        with col_b:
            st.markdown("#### Aktuelle Trainingsdaten")
            filt = st.selectbox("Filtern", ["Alle"] + ÄMTER, key="filt_cls")
            df_show = pd.DataFrame({"Anfrage": st.session_state.cls_texts, "Amt": st.session_state.cls_labels})
            if filt != "Alle":
                df_show = df_show[df_show["Amt"] == filt]
            st.dataframe(df_show, use_container_width=True, height=340, hide_index=True)

    with t_load:
        col_c, col_d = st.columns([1, 1], gap="large")
        with col_c:
            st.markdown("#### Neuen Auslastungswert hinzufügen")
            l_day = st.selectbox("Tag", DAYS, key="l_day")
            l_mon = st.selectbox("Monat", MONATE, key="l_mon",
                                  index=datetime.date.today().month - 1)
            l_vis = st.number_input("Tatsächliche Besucherzahl",
                                     min_value=0, max_value=500, value=100, key="l_vis")
            if st.button("Hinzufügen", type="primary", key="add_load"):
                mon_idx = MONATE.index(l_mon) + 1
                st.session_state.load_X.append([DAYS.index(l_day), mon_idx])
                st.session_state.load_Y.append(l_vis)
                load_model = train_load_model()
                st.success(f"Eintrag hinzugefügt: {l_day}, {l_mon}, {l_vis} Besucher")
        with col_d:
            st.markdown("#### Auslastungsdaten Übersicht")
            df_load = pd.DataFrame(st.session_state.load_X, columns=["Tag-Index", "Monat"])
            df_load["Besucher"] = st.session_state.load_Y
            df_load["Tag"]      = df_load["Tag-Index"].apply(lambda x: DAYS[x])
            df_load["Monat"]    = df_load["Monat"].apply(lambda x: MONATE[x - 1])
            st.dataframe(df_load[["Tag", "Monat", "Besucher"]].tail(20),
                         use_container_width=True, height=340, hide_index=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — ANFRAGEN-LOG
# ══════════════════════════════════════════════════════════════

with tab4:
    if not st.session_state.log:
        st.info("Noch keine Anfragen verarbeitet.")
    else:
        df_log   = pd.DataFrame(st.session_state.log)
        filt_amt = st.selectbox("Filtern nach Amt", ["Alle"] + ÄMTER, key="log_filt")
        df_filt  = df_log if filt_amt == "Alle" else df_log[df_log["amt"] == filt_amt]
        st.dataframe(df_filt, use_container_width=True, hide_index=True)
        st.download_button("Log exportieren",
            data=df_filt.to_csv(index=False).encode("utf-8"),
            file_name=f"log_{datetime.date.today()}.csv",
            mime="text/csv", use_container_width=False)
        st.markdown("---")
        st.markdown("#### Anfragen nach Amt")
        st.bar_chart(df_log["amt"].value_counts(), height=260)
