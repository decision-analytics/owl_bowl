"""
workshop_demo.py
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", layout_file="layouts/workshop.slides.json")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from src.data import simulate_true_demand_for_week_context, simulate_history
    from src.costs import NewsvendorCosts, newsvendor_cost, newsvendor_profit
    from src.simulation import run_backtest_historical_data

    return (
        NewsvendorCosts,
        mo,
        newsvendor_profit,
        np,
        pd,
        plt,
        run_backtest_historical_data,
        simulate_history,
        simulate_true_demand_for_week_context,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="public/logo_wiwi_en_.png" width="30%" align="left">

    <img src="public/decision_analytics_logo.png" width="17%" align="right">

    <br>
    <br>

    <br><br><br>
    # DenkBar 2026 Löhne / Kreis Herford



    ## KI-gestützte Planung unter Unsicherheit: Wieso bessere Prognosen nicht unbedingt zu besseren Entscheidungen führen


    <br>

    <br>
    <br>


    **Prof. Dr. Michael Römer**

    **Decision and Operation Technologies / Decision Analytics Group  | Bielefeld University**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Ziele


    Wir entwickeln ein (intuitives) Verständnis dafür
    - dass ein naiver Einsatz von datengetriebenen Vorhersagen oft zu suboptimalen Entscheidungen führt
    - es bei Entscheidungen unter Unsicherheit nicht auf Vorhersagegenauigkeit, sondern auf den richtigen Ansatz zur Entscheidungsfindung ankommt

    Wir entdecken, dass
    - wir oft sehr gut darin sind, Entscheidungen unter Unsicherheit zu treffen
    - zu einfache (formale, computer-gestützte) Modelle und Ansätze oftmals zu schlechten Entscheidungen führen
    - es aber auch gar nicht ganz so komplizierte Modelle gibt, die mit Unsicherheit gut umgehen können und mit unserer Intuition in Einklang gebracht werden können

    Wir identifizieren Siuationen aus der Praxis, in denen Entscheidungen unter Unsicherheit von besseren Ansätzen profitieren würden


    -> Was sind Ihre Erwartungen?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Ablauf des Workshops

    1. Vorstellung
    2. OWL-Bowl: Eine Fallstudie zur Entscheidung unter Unsicherheit
    3. Wo haben Sie Entscheidungen unter Unsicherheit?
    5. Perspektiven für mögliche Kooperationen
    6. Dinge zum mitnehmen
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Kennenlernen

    - Bitte stellen Sie sich kurz vor!
    - Was führt Sie zur DenkBar und zu diesem Workshop?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # OWL-Bowl: Eine Fallstudie zur Entscheidungsfindung unter Unsicherheit


    Stellen Sie sich vor, Sie betreiben einen Food-Truck auf dem Wochenmarkt in Ostwestfalen-Lippe. Sie verkaufen jeden Samstag eine begehrte regionale 'OWL-Bowl'.

    Das **Problem** ist, dass

    - die frischen Zutaten für die Bowl **Samstag früh** vorbereiten und einkaufen müssen
    - die **Nachfrage** nach den Bowls **unsicher** ist

    Folgende Informationen sind gegeben:

    - **Materialkosten:** 3.00 € pro Bowl
    - **Verkaufspreis:** 10.00 € pro Bowl
    - **Restwert der Zutaten für eine Bowl:** 2.50 € (Einige Zutaten können am nächsten Tag weiterverwendet werden)

    Die Frage ist: Wie viele Bowls sollten Sie für ein gegebes Wochenende bestellen?
    """)
    return


@app.cell
def _(mo):
    n_years = mo.ui.slider(5, 12, value=8, label="Historische Jahre")
    seed = mo.ui.number(value=42, label="Seed")
    overage_cost = mo.ui.number(value=0.5, label="Kosten bei Überbestand")
    underage_cost = mo.ui.number(value=7.0, label="Kosten bei Unterbestand")
    include_spikes = mo.ui.checkbox(value=False, label="Seltene Nachfragespitzen (Spikes) simulieren")

    controls = mo.vstack(
        [
            mo.md("## Einstellungen \nHier können Sie die Parameter der Simulation anpassen."),
            n_years,
            seed,
            overage_cost,
            underage_cost,
            include_spikes,
        ]
    )
    return controls, include_spikes, n_years, overage_cost, seed, underage_cost


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(
    include_spikes,
    n_years,
    overage_cost,
    run_backtest_historical_data,
    seed,
    underage_cost,
):
    summary, detail, q_scan_example, df, q_scan_all_weeks = run_backtest_historical_data(
        n_years=int(n_years.value),
        seed=int(seed.value),
        overage_cost=float(overage_cost.value),
        underage_cost=float(underage_cost.value),
        n_samples_ngb=1500,
        tau_low=0.30,
        tau_high=0.99,
        tau_steps=50,
        include_spikes=bool(include_spikes.value),
    )
    return detail, df, q_scan_all_weeks, q_scan_example, summary


@app.cell
def _(df, mo, plt):
    # Histogramm der historischen Nachfrage erstellen
    fig_dist, ax_dist = plt.subplots(figsize=(11, 4))
    ax_dist.hist(df["demand"], bins=30, color='#1f77b4', alpha=0.7, edgecolor='white')
    ax_dist.set_title("Verteilung der historischen Nachfrage")
    ax_dist.set_xlabel("Nachfrage pro Woche")
    ax_dist.set_ylabel("Häufigkeit")
    ax_dist.grid(alpha=0.3)

    # Layout für die "Histogramm-Folie"
    slide_distribution = mo.vstack([
        mo.md("## Histogramm der historischen Daten"),
        mo.md("Diese Darstellung zeigt uns, wie häufig welche Nachfragemengen in der Vergangenheit aufgetreten sind. "
              "Dies hilft uns zu verstehen, ob die Nachfrage eher normalverteilt ist oder "
              "ob es 'fat tails' (viele kleine oder große Ausreißer) gibt."),
        fig_dist
    ])
    return (slide_distribution,)


@app.cell
def _(slide_distribution):
    slide_distribution
    return


@app.cell
def _(
    NewsvendorCosts,
    df,
    mo,
    newsvendor_profit,
    np,
    overage_cost,
    pd,
    underage_cost,
):
    # Berechnung der Performance-Kennzahlen basierend auf dem einfachen Mittelwert-Forecast
    mean_demand = df["demand"].mean()
    mae_mean = np.mean(np.abs(df["demand"] - mean_demand))
    rmse_mean = np.sqrt(np.mean((df["demand"] - mean_demand)**2))

    # Erwarteter Gewinn (Newsvendor-Prinzip) für die Mittelwert-Strategie
    costs_simple = NewsvendorCosts(overage_cost=float(overage_cost.value), underage_cost=float(underage_cost.value))
    expected_profit_mean = newsvendor_profit(mean_demand, df["demand"], costs_simple).mean()

    # Darstellung als Tabelle und Markdown
    summary_table = pd.DataFrame({
        "Metrik": ["Mittlere Nachfrage", "MAE (Mittlerer Absoluter Fehler)", "RMSE", "Mittlerer Wochengewinn"],
        "Wert": [f"{mean_demand:.2f}", f"{mae_mean:.2f}", f"{rmse_mean:.2f}", f"{expected_profit_mean:.2f} €"]
    })

    slide_naive_analysis = mo.vstack([
        mo.md("## Analyse: Naive Strategie (Historischer Mittelwert)"),
        mo.md(f"Wir betrachten hier die Baseline-Strategie, bei der wir jeden Samstag konstant **{mean_demand:.0f}** Bowls bestellen, basierend auf dem Durchschnitt der historischen Daten."),
        summary_table
    ])
    return (slide_naive_analysis,)


@app.cell
def _(slide_naive_analysis):
    slide_naive_analysis
    return


@app.cell
def _(df, mo, plt):
    fig_hist, ax_hist = plt.subplots(figsize=(11, 4))
    ax_hist.plot(df.index, df["demand"], label="Historische Nachfrage", alpha=0.85, color="#1f77b4")
    ax_hist.set_title("Historische Daten über vergangene Jahre")
    ax_hist.set_xlabel("Wochen (fortlaufend)")
    ax_hist.set_ylabel("Nachfrage")
    ax_hist.grid(alpha=0.3)

    slide_timeseries = mo.vstack([
        mo.md("## Jetzt kommt Data Science ins Spiel: Historische Daten im Zeitverlauf\n\nWir sehen hier die (simulierte) Historie der Nachfrage über alle vergangenen Jahre. Die Daten unterliegen saisonalen Schwankungen, Wettereinflüssen und Einflüssen wie Ferien"),
        fig_hist
    ])
    return (slide_timeseries,)


@app.cell
def _(slide_timeseries):
    slide_timeseries
    return


@app.cell
def _(detail, mo, pd, plt, summary):
    # Berechnung für die Visualisierung des Testjahrs (52 Wochen)
    fig_Mittelwert, ax_Mittelwert = plt.subplots(figsize=(11, 3))
    ax_Mittelwert.plot(detail["week"], detail["demand"], label="Tatsächliche Nachfrage", marker='o', alpha=0.6, color='black')
    ax_Mittelwert.plot(detail["week"], detail["forecast_mean_based"], color='red', linestyle='--', label=f"Mittelwert Forecast ({summary['mean_based_mean']:.0f})")
    ax_Mittelwert.set_title("Mittelwert-Vorhersage (Historischer Mittelwert) vs. Realität (Evaluation im Testjahr)")
    ax_Mittelwert.set_xlabel("Woche")
    ax_Mittelwert.set_ylabel("Nachfrage")
    ax_Mittelwert.legend()
    ax_Mittelwert.grid(alpha=0.3)
    fig_Mittelwert = plt.gca()

    slide_mean_forecast = mo.vstack([
        mo.md("## Statische Vorhersage mittels Mittelwert\n\nEine einfache Vorhersage mittels Mittelwert ignoriert Schwankungen, ignoriert Kontextinformationen und nimmt einfach den historischen Gesamtmittelwert der Trainingsjahre (alles außer das Testjahr) als Forecast und zugleich als Bestellmenge. Wir evaluieren diesen Ansatz auf dem letzten Jahr (Testjahr)."),
        mo.md(f"**Geschätzter Mittelwert:** `{summary['mean_based_mean']:.0f}`"),
        fig_Mittelwert,
        pd.DataFrame({
            "Metrik": ["MAE", "RMSE", "Mittlerer Wochengewinn"],
            "Mittelwert": [
                summary["mae_mean_based"], 
                summary["rmse_mean_based"], 
                f"{summary['avg_weekly_profit_mean_based_point']:.2f} €"
            ]
        })
    ])
    return (slide_mean_forecast,)


@app.cell
def _(slide_mean_forecast):
    slide_mean_forecast
    return


@app.cell
def _(df, mo):
    FEATURES = ["week_sin", "week_cos", "temp", "holiday", "rain", "event"]
    sample_df = df[["demand"] + FEATURES].head(6).copy()
    sample_df = sample_df.rename(columns={
        "demand": "Nachfrage (Label)",
        "week_sin": "Saisonfaktor 1",
        "week_cos": "Saisonfaktor 2",
        "temp": "Temperatur (°C)",
        "holiday": "Ferien (1/0)",
        "rain": "Regen (mm)",
        "event": "Event (1/0)"
    })
    context_table = mo.ui.table(sample_df.round(3))

    slide_context = mo.vstack([
        mo.md("## Data Science und Maschinelles Lernen: Kontext-Informationen nutzen\n\nBevor wir ein Machine-Learning-Modell trainieren, müssen wir verstehen, welche Daten ihm zur Verfügung stehen. Im Gegensatz zum einfachen Mittelwert-Forecast nutzen wir nun ein fortgeschrittenes ML-Modell (XGBoost), das folgende zusätzlichen Einflussfaktoren (Features) nutzt, um die Nachfrage (Label) vorherzusagen:"),
        mo.md("""
    ### Unsere Datenstruktur:
    *   **Das Label (vorherzusagende Größe)**: `Nachfrage` (wie viele Einheiten tatsächlich nachgefragt wurden).
    *   **Die Features (Einflussfaktoren)**:
        *   **Saisonalität (zwei Werte)**: Erfasst wöchentliche Zyklen über das Jahr hinweg.
        *   **Wetter (`Temperatur` & `Regen`)**: Erfasst den Einfluss von Hitze oder Niederschlag auf den Stadionbesuch.
        *   **Stadion-Events (`Event`)**: Ob ein großes Spiel/Event stattfindet, das die Nachfrage sprunghaft ansteigen lässt.
        *   **Ferienzeit (`Ferien`)**: Ob Ferien oder Feiertage die Freizeitaktivitäten beeinflussen.
        """),
        mo.md("### Beispielhafte Datenzeilen (Features und Label):"),
        context_table
    ])
    return (slide_context,)


@app.cell
def _(slide_context):
    slide_context
    return


@app.cell
def _(detail, mo, pd, plt, summary):
    # Erstellung der Vergleichsgrafik für XGBoost im Testjahr
    fig_xgb, ax_xgb = plt.subplots(figsize=(11, 3))

    ax_xgb.plot(detail["week"], detail["demand"], label="Tatsächliche Nachfrage", marker='o', alpha=0.6, color='black')
    ax_xgb.plot(detail["week"], detail["forecast_xgb"], label="XGBoost-Forecast", marker='x', linestyle='--', color='orange')
    ax_xgb.set_title("XGBoost-Vorhersage vs. Realität (Evaluation im Testjahr)")
    ax_xgb.set_xlabel("Woche")
    ax_xgb.set_ylabel("Nachfrage")
    ax_xgb.legend()
    ax_xgb.grid(alpha=0.3)

    # Berechnung des Metrik-DataFrames mit Rundung
    metrics_df = pd.DataFrame({
        "Metrik": ["MAE", "RMSE"],
        "Mittelwert": [summary["mae_mean_based"], summary["rmse_mean_based"]],
        "XGBoost": [summary["mae_xgb"], summary["rmse_xgb"]]
    }).set_index("Metrik").round(2)

    slide_xgb_forecast = mo.vstack([
        mo.md("## Data Science: Vorhersage mittels Features (XGBoost)\n\nJetzt trainieren wir ein moderneres Modell (`XGBoost`) auf den historischen Trainingsjahren. Dieses Modell lernt aus Kontextdaten (Wetter, Saisonalität, etc.). Wir evaluieren die Vorhersage-Güte ebenfalls auf dem letzten Jahr (Testjahr)."),
        fig_xgb,
        mo.md("### Vergleich der Vorhersage-Güte"),
        metrics_df
    ])
    return (slide_xgb_forecast,)


@app.cell
def _(slide_xgb_forecast):
    slide_xgb_forecast
    return


@app.cell
def _(detail, mo, pd, plt, summary):
    fig_eval1_xgb, ax_eval1_xgb = plt.subplots(figsize=(11, 4))
    ax_eval1_xgb.plot(detail["week"], detail["demand"], label="Realisierte Nachfrage", linewidth=2, color='black')
    ax_eval1_xgb.plot(detail["week"], detail["q_mean_based"], label="Mittelwert-Bestellmenge", alpha=0.85)
    ax_eval1_xgb.plot(detail["week"], detail["q_xgb"], label="XGBoost-Bestellmenge", alpha=0.85)
    ax_eval1_xgb.set_xlabel("Woche")
    ax_eval1_xgb.set_ylabel("Menge")
    ax_eval1_xgb.set_title("Nachfrage und Bestellentscheidungen im Testjahr")
    ax_eval1_xgb.legend()
    ax_eval1_xgb.grid(alpha=0.25)

    fig_eval2_xgb, ax_eval2_xgb = plt.subplots(figsize=(11, 4))
    ax_eval2_xgb.plot(detail["week"], detail["cum_gewinn_mean_based"], label="Mittelwert Strategie", linewidth=2)
    ax_eval2_xgb.plot(detail["week"], detail["cum_gewinn_xgb"], label="XGBoost Strategie", linewidth=2)
    ax_eval2_xgb.set_xlabel("Woche")
    ax_eval2_xgb.set_ylabel("Kumulierter Gewinn")
    ax_eval2_xgb.set_title("Kumulierter Gewinn im Testjahr")
    ax_eval2_xgb.legend()
    ax_eval2_xgb.grid(alpha=0.25)

    decision_table_xgb = pd.DataFrame({
        "Strategie": ["Mittelwert (Naive)", "XGBoost (ML)"],
        "Jahresgewinn": [f"{summary['annual_profit_mean_based_point']:.2f} €", f"{summary['annual_profit_xgb_point']:.2f} €"],
        "Ø Wochengewinn": [f"{summary['avg_weekly_profit_mean_based_point']:.2f} €", f"{summary['avg_weekly_profit_xgb_point']:.2f} €"],
    })

    better_forecast_xgb = summary["mae_xgb"] < summary["mae_mean_based"]
    better_decision_xgb = summary["annual_profit_xgb_point"] > summary["annual_profit_mean_based_point"]

    slide_decision_comparison = mo.vstack([
        mo.md("## Entscheidungs-Evaluation: Mittelwert vs. ML-Modell\n\nFührt das bessere Machine-Learning-Vorhersagemodell automatisch zu besseren Entscheidungen und höheren Gewinnen? Wir vergleichen den Gewinn der Bestellentscheidungen über das Testjahr."),
        decision_table_xgb,
        mo.md(f"""
    ### Kernaussage
    - Hat XGBoost die besseren Vorhersagen (geringerer Fehler)? **{better_forecast_xgb}**
    - Führt XGBoost auch zu höheren Gewinnen / besseren Entscheidungen? **{better_decision_xgb}**

    Obwohl das ML-Modell die Nachfrage im Durchschnitt genauer trifft, kann es in der Praxis durch die **asymmetrischen Kosten** (z.B. hohe Kosten bei Unterbestand) zu schlechteren Entscheidungen (und damit geringeren Gewinnen) führen als der naive Ansatz!
        """),
        fig_eval1_xgb,
        fig_eval2_xgb
    ])
    return (slide_decision_comparison,)


@app.cell
def _(slide_single_week_evaluation):
    slide_single_week_evaluation
    return


@app.cell
def _(slide_decision_comparison):
    slide_decision_comparison
    return


@app.cell
def _(detail):
    better_mean_based_weeks = detail[detail["gewinn_mean_based"] > detail["gewinn_xgb"]]
    return (better_mean_based_weeks,)


@app.cell
def _(better_mean_based_weeks, detail, mo):
    default_week = int(better_mean_based_weeks.iloc[0]["week"]) if not better_mean_based_weeks.empty else 1

    week_selector = mo.ui.dropdown(
        options={f"Woche {w}": w for w in detail["week"]},
        value=f"Woche {default_week}",
        label="Zu evaluierende Woche auswählen:"
    )
    return (week_selector,)


@app.cell
def _(
    NewsvendorCosts,
    detail,
    mo,
    newsvendor_profit,
    overage_cost,
    pd,
    plt,
    simulate_true_demand_for_week_context,
    underage_cost,
    week_selector,
):
    selected_week = week_selector.value
    row = detail[detail["week"] == selected_week].iloc[0]

    costs = NewsvendorCosts(float(overage_cost.value), float(underage_cost.value))

    q_mean_based = row["q_mean_based"]
    q_xgb = row["q_xgb"]
    q_ngb = row["q_ngb"]

    true_samples = simulate_true_demand_for_week_context(row, n_samples=10000, seed=42)

    true_profit_mean_based = newsvendor_profit(q_mean_based, true_samples, costs).mean()
    true_profit_xgb = newsvendor_profit(q_xgb, true_samples, costs).mean()
    true_profit_ngb = newsvendor_profit(q_ngb, true_samples, costs).mean()

    fig_single, ax_single = plt.subplots(figsize=(9, 4))
    ax_single.hist(true_samples, bins=range(200), alpha=0.5, color='gray', density=True, label="Wahre Verteilung (10k Samples)")
    ax_single.axvline(q_mean_based, color='blue', linestyle='--', label=f"Mittelwert (q={q_mean_based:.0f})")
    ax_single.axvline(q_xgb, color='orange', linestyle='--', label=f"XGBoost (q={q_xgb:.0f})")
    ax_single.axvline(q_ngb, color='green', linestyle='--', label=f"NGBoost (q={q_ngb:.0f})")
    ax_single.set_xlabel("Nachfrage")
    ax_single.set_ylabel("Dichte")
    ax_single.set_title(f"Wahre bedingte Nachfrageverteilung für Woche {selected_week}")
    ax_single.legend()
    ax_single.grid(alpha=0.3)

    eval_table = pd.DataFrame({
        "Policy": ["Mittelwert", "XGBoost", "NGBoost"],
        "Bestellmenge": [q_mean_based, q_xgb, q_ngb],
        "Tatsächlicher Gewinn (im Backtest)": [f"{row['gewinn_mean_based']:.2f} €", f"{row['gewinn_xgb']:.2f} €", f"{row['gewinn_ngb']:.2f} €"],
        "Wahrer erwarteter Gewinn": [f"{true_profit_mean_based:.2f} €", f"{true_profit_xgb:.2f} €", f"{true_profit_ngb:.2f} €"]
    })

    slide_single_week_evaluation = mo.vstack([
        mo.md(f"## Evaluation der Politik in Woche {selected_week}"),
        week_selector,
        mo.md("Hier evaluieren wir die Entscheidungen aus der Backtest-Simulation gegen die **wahre datengenerierende bedingte Verteilung** dieser spezifischen Woche."),
        fig_single,
        eval_table
    ])
    return (slide_single_week_evaluation,)


@app.cell
def _(mo):
    slide_ngboost_intro = mo.vstack([
        mo.md("## Ein Ausweg: Probabilistische Prognosen (Verteilungsprognose)"),
        mo.md("""
        In der Praxis ist die Nachfrage nie eine feste Zahl, sondern eine Verteilung mit Unsicherheit. 
        Ein Punktmodell wie XGBoost versucht, den "durchschnittlichen" Wert vorherzusagen (Punktprognose). Bei asymmetrischen Kosten (z.B. wenn Unterbestand viel teurer ist als Überbestand) reicht das nicht: Wir wollen uns gegen Risiken absichern und nicht stur den Mittelwert bestellen!

        **Die Idee von NGBoost:**
        - NGBoost macht keine Punktprognose, sondern prognostiziert für jede Woche eine **ganze Wahrscheinlichkeitsverteilung** (z.B. eine Normalverteilung mit einem Mittelwert und einer Standardabweichung).
        - Dadurch kennen wir nicht nur den wahrscheinlichsten Wert, sondern auch die **Spannbreite** und die Wahrscheinlichkeiten für Ausreißer.
        - Mit dieser Verteilung können wir mathematisch genau ermitteln, welche Bestellmenge das Risiko minimiert.
        """)
    ])
    return (slide_ngboost_intro,)


@app.cell
def _(slide_ngboost_intro):
    slide_ngboost_intro
    return


@app.cell
def _(mo, plt, q_scan_all_weeks, summary):
    fig_quant_all, ax_quant_all = plt.subplots(figsize=(9, 4))
    ax_quant_all.plot(q_scan_all_weeks["tau"], q_scan_all_weeks["avg_profit"], marker="o", color="#2ca02c")

    # Finde das beste Quantil im Mittel
    best_row_all = q_scan_all_weeks.loc[q_scan_all_weeks["avg_profit"].idxmax()]

    # Berechne das theoretisch optimale Quantil (Critical Fractile)
    crit_fractile = summary["critical_fractile_reference"]

    ax_quant_all.axvline(best_row_all["tau"], linestyle="--", alpha=0.7, color="red", 
                         label=f"Empirisch bestes Quantil: {best_row_all['tau']:.2f}")
    ax_quant_all.axvline(crit_fractile, linestyle=":", alpha=0.7, color="blue", 
                         label=f"Theoretisches Optimum (Critical Fractile): {crit_fractile:.2f}")

    ax_quant_all.set_xlabel("Sicherheitsstufe / Quantil (tau)")
    ax_quant_all.set_ylabel("Ø wöchentlicher Gewinn (€)")
    ax_quant_all.set_title("Durchschnittlicher Gewinn über alle Wochen im Testjahr je nach Quantil")
    ax_quant_all.legend()
    ax_quant_all.grid(alpha=0.25)

    slide_quantile_optimization = mo.vstack([
        mo.md("## Optimierung über Quantile (Im Mittel über alle Wochen)"),
        mo.md("""
        Wir können für jede Woche simulieren, wie hoch der erwartete Gewinn für verschiedene Quantile (Sicherheitsstufen) wäre. 
        Wenn wir diesen Gewinn über alle Wochen im Testjahr mitteln, sehen wir einen klaren Verlauf (umgekehrtes U):
        - Ein zu niedriges Quantil (z.B. links) führt zu häufigem Unterbestand und entgangenem Gewinn.
        - Ein zu hohes Quantil (z.B. rechts) führt zu großem Überbestand und Abschreibungskosten.
        - Das Gewinnmaximum liegt exakt beim theoretischen Optimum (Critical Fractile), das auf unserem Kostenverhältnis basiert.
        """),
        fig_quant_all
    ])
    return (slide_quantile_optimization,)


@app.cell
def _(slide_quantile_optimization):
    slide_quantile_optimization
    return


@app.cell
def _(detail, mo, pd, plt, summary):
    fig_eval1, ax_eval1 = plt.subplots(figsize=(11, 4))
    ax_eval1.plot(detail["week"], detail["demand"], label="Realisierte Nachfrage", linewidth=2, color='black')
    ax_eval1.plot(detail["week"], detail["q_mean_based"], label="Mittelwert-Bestellmenge", alpha=0.85)
    ax_eval1.plot(detail["week"], detail["q_xgb"], label="XGBoost-Bestellmenge", alpha=0.85)
    ax_eval1.plot(detail["week"], detail["q_ngb"], label="NGBoost-Bestellmenge", alpha=0.85)
    ax_eval1.set_xlabel("Woche")
    ax_eval1.set_ylabel("Menge")
    ax_eval1.set_title("Nachfrage und Bestellentscheidungen im Backtest-Jahr")
    ax_eval1.legend()
    ax_eval1.grid(alpha=0.25)

    fig_eval2, ax_eval2 = plt.subplots(figsize=(11, 4))
    ax_eval2.plot(detail["week"], detail["cum_gewinn_mean_based"], label="Mittelwert point", linewidth=2)
    ax_eval2.plot(detail["week"], detail["cum_gewinn_xgb"], label="XGBoost point", linewidth=2)
    ax_eval2.plot(detail["week"], detail["cum_gewinn_ngb"], label="NGBoost probabilistic", linewidth=2)
    ax_eval2.set_xlabel("Woche")
    ax_eval2.set_ylabel("Kumulierter Gewinn")
    ax_eval2.set_title("Kumulierter Gewinn im Backtest")
    ax_eval2.legend()
    ax_eval2.grid(alpha=0.25)

    decision_table = pd.DataFrame({
        "Policy": ["Mittelwert point", "XGBoost point", "NGBoost probabilistic"],
        "Jahresgewinn": [summary["annual_profit_mean_based_point"], summary["annual_profit_xgb_point"], summary["annual_profit_ngb_prob"]],
        "Ø Wochengewinn": [summary["avg_weekly_profit_mean_based_point"], summary["avg_weekly_profit_xgb_point"], summary["avg_weekly_profit_ngb_prob"]],
    })

    better_forecast = summary["mae_xgb"] < summary["mae_mean_based"]
    better_decision = summary["annual_profit_xgb_point"] > summary["annual_profit_mean_based_point"]
    ngb_beats_xgb = summary["annual_profit_ngb_prob"] > summary["annual_profit_xgb_point"]

    slide_backtest_evaluation = mo.vstack([
        mo.md("## Evaluation der Politik anhand der historischen Simulation\n\nWir bewerten nun die Politiken mit der asymmetrischen Kostenstruktur (Überbestand vs. Unterbestand) im Hinblick auf den erzielten Gewinn. Wir nehmen hier zusätzlich die NGBoost probabilistische Methode hinzu, welche durch Simulation das beste Quantil ermittelt."),
        decision_table,
        mo.md(f"""
    ### Kernaussage
    - Ist `XGBoost` der bessere Punkt-Forecaster? **{better_forecast}**
    - Führt `XGBoost` (als Punktprognose) auch zu einem höheren Gewinn? **{better_decision}**
    - Schlägt `NGBoost` mit probabilistischer brute-force-Policy das Punktmodell im Hinblick auf den Gewinn? **{ngb_beats_xgb}**

    Ein besserer Punkt-Forecast gewinnt zwar bei der MAE/RMSE-Betrachtung, aber kann in der eingebetteten Entscheidung (wegen asymmetrischer Kosten) verlieren!
        """),
        fig_eval1,
        fig_eval2
    ])
    return (slide_backtest_evaluation,)


@app.cell
def _(slide_backtest_evaluation):
    slide_backtest_evaluation
    return


@app.cell
def _(
    include_spikes,
    overage_cost,
    run_backtest_historical_data,
    seed,
    underage_cost,
):
    summary_longterm, _, _, _, _ = run_backtest_historical_data(
        n_years=40,
        seed=int(seed.value),
        overage_cost=float(overage_cost.value),
        underage_cost=float(underage_cost.value),
        n_samples_ngb=1500,
        tau_low=0.30,
        tau_high=0.99,
        tau_steps=50,
        include_spikes=bool(include_spikes.value),
        n_test_years=10,
    )
    return (summary_longterm,)


@app.cell
def _(mo, pd, summary_longterm):
    decision_table_longterm = pd.DataFrame({
        "Policy": ["Mittelwert", "XGBoost Punktschätzer", "NGBoost Probabilistisch"],
        "Ø Jahresgewinn (10 J. Test)": [
            f"{summary_longterm['annual_profit_mean_based_point']:.2f} €",
            f"{summary_longterm['annual_profit_xgb_point']:.2f} €",
            f"{summary_longterm['annual_profit_ngb_prob']:.2f} €"
        ],
        "Ø Wochengewinn (10 J. Test)": [
            f"{summary_longterm['avg_weekly_profit_mean_based_point']:.2f} €",
            f"{summary_longterm['avg_weekly_profit_xgb_point']:.2f} €",
            f"{summary_longterm['avg_weekly_profit_ngb_prob']:.2f} €"
        ],
    })

    better_forecast_longterm = summary_longterm["mae_xgb"] < summary_longterm["mae_mean_based"]
    better_decision_longterm = summary_longterm["annual_profit_xgb_point"] > summary_longterm["annual_profit_mean_based_point"]
    ngb_beats_xgb_longterm = summary_longterm["annual_profit_ngb_prob"] > summary_longterm["annual_profit_xgb_point"]

    slide_longterm_evaluation = mo.vstack([
        mo.md("## Langzeit-Evaluation der Politiken (40 Jahre Simulation)"),
        mo.md("Um die Robustheit der gelernten Politiken über einen langen Zeithorizont und mehrere Testjahre hinweg zu prüfen, simulieren wir hier eine Welt mit **40 Jahren** Datenhistorie. Wir teilen diese in **30 Jahre Training** und **10 Jahre Testing** auf. Hierbei sind keine Abbildungen enthalten, um den Fokus rein auf den langfristigen Gewinn zu legen."),
        decision_table_longterm,
        mo.md(f"""
    ### Langzeit-Erkenntnisse (30 J. Training / 10 J. Test)
    - Ist `XGBoost` langfristig der bessere Punkt-Forecaster? **{better_forecast_longterm}**
    - Erzielt `XGBoost` (Punktprognose) langfristig auch den höheren Gewinn? **{better_decision_longterm}**
    - Schlägt `NGBoost` mit der probabilistischen Quantil-Policy das Punktmodell langfristig im Hinblick auf den Gewinn? **{ngb_beats_xgb_longterm}**

    Auch über einen sehr langen Zeithorizont (10 Testjahre) und bei reichlich Trainingsdaten (30 Jahre) zeigt sich: Der probabilistische Ansatz von NGBoost erzielt signifikant höhere Gewinne, da er die asymmetrische Kostenstruktur explizit in die Entscheidung einbezieht!
        """)
    ])
    return (slide_longterm_evaluation,)


@app.cell
def _(slide_longterm_evaluation):
    slide_longterm_evaluation
    return


@app.cell
def _(mo, plt, q_scan_example):
    fig_quant, ax_quant = plt.subplots(figsize=(9, 4))
    ax_quant.plot(q_scan_example["tau"], q_scan_example["avg_profit"], marker="o")
    best_row = q_scan_example.iloc[0]
    ax_quant.axvline(best_row["tau"], linestyle="--", alpha=0.7, color="red", label=f"Bestes Quantil: {best_row['tau']:.2f}")
    ax_quant.set_xlabel("Getestetes Quantil (tau)")
    ax_quant.set_ylabel("Simulierter Durchschnittsgewinn")
    ax_quant.set_title("Brute-force-Scan über Quantile (Beispiel Woche 1 - Erwarteter Gewinn)")
    ax_quant.legend()
    ax_quant.grid(alpha=0.25)

    slide_quantile_scan = mo.vstack([
        mo.md("## Manuelles Durchspielen von Quantilen\n\nDie NGBoost-Policy ermittelt für jede Woche eine prädiktive Verteilung der Nachfrage. \nAnstatt analytisch vorzugehen, ziehen wir Samples aus der Verteilung und berechnen für eine Reihe von Quantilen (tau) den simulierten erwarteten Gewinn. Das Maximum wird als Bestellmenge gewählt."),
        fig_quant,
        mo.md("Tabelle der Quantile in Woche 1:"),
        q_scan_example
    ])
    return


@app.cell
def _():
    #slide_quantile_scan
    return


@app.cell
def _(mo):
    get_game_state, set_game_state = mo.state({
        "phase": "start",
        "week_idx": 0,
        "df": None,
        "history": [],
        "last_result": None
    })
    return get_game_state, set_game_state


@app.cell
def _(
    NewsvendorCosts,
    get_game_state,
    include_spikes,
    mo,
    seed,
    set_game_state,
    simulate_history,
):
    # Static UI widgets and callbacks defined ONCE.
    # It depends on seed, so it will update only when seed parameter is tweaked.

    def on_start_click(*args):
        df_full = simulate_history(
            n_years=1,
            seed=int(seed.value),
            include_spikes=bool(include_spikes.value),
        )
        df_game = df_full.head(10).copy()
        set_game_state({
            "phase": "play",
            "week_idx": 0,
            "df": df_game,
            "history": [],
            "last_result": None
        })

    start_btn = mo.ui.button(label="Spiel starten", on_click=on_start_click, kind="success")

    q_input = mo.ui.number(value=50, step=1, label="Ihre Bestellmenge für diese Woche:")

    def on_submit(*args):
        curr = get_game_state()
        if curr["phase"] != "play": return

        sub_q = q_input.value
        if sub_q is None: return

        week_idx = curr["week_idx"]
        df_play = curr["df"]
        row_play = df_play.iloc[week_idx]
        sub_d = row_play["demand"]

        costs_obj = NewsvendorCosts(overage_cost=0.5, underage_cost=7.0)
        profit_val = 10.00 * min(sub_q, sub_d) - 3.00 * sub_q + 2.50 * max(0, sub_q - sub_d)

        sub_result = {
            "Woche": week_idx + 1,
            "Temperatur": float(row_play['temp']),
            "Regen": "Ja" if row_play['rain'] else "Nein",
            "Event": "Ja" if row_play['event'] else "Nein",
            "Ferien": "Ja" if row_play['holiday'] else "Nein",
            "Bestellmenge": int(sub_q),
            "Nachfrage": int(sub_d),
            "Gewinn (€)": float(profit_val)
        }

        new_history = curr["history"] + [sub_result]

        set_game_state({
            **curr,
            "phase": "feedback",
            "history": new_history,
            "last_result": sub_result
        })

    submit_btn = mo.ui.button(label="Entscheidung abschicken", on_click=on_submit, kind="success")

    def on_next_click(*args):
        curr = get_game_state()
        fb_week_idx = curr["week_idx"]
        if fb_week_idx + 1 >= 10:
            set_game_state({**curr, "phase": "end"})
        else:
            set_game_state({**curr, "phase": "play", "week_idx": fb_week_idx + 1})

    next_btn = mo.ui.button(label="Nächste Woche", on_click=on_next_click, kind="info")

    def on_restart_click(*args):
        set_game_state({
            "phase": "start",
            "week_idx": 0,
            "df": None,
            "history": [],
            "last_result": None
        })

    restart_btn = mo.ui.button(label="Neues Spiel", on_click=on_restart_click)
    return next_btn, q_input, restart_btn, start_btn, submit_btn


@app.cell
def _(
    get_game_state,
    mo,
    next_btn,
    pd,
    q_input,
    restart_btn,
    start_btn,
    submit_btn,
):
    # Pure rendering cell for the interactive game tab.
    curr_state = get_game_state()
    slide_game = None


    # Helper to generate history table display
    history_table = None
    if curr_state["history"]:
        history_df = pd.DataFrame(curr_state["history"])
        display_df = history_df.copy()
        display_df["Temperatur"] = display_df["Temperatur"].apply(lambda t: f"{t:.1f} °C")
        display_df["Gewinn (€)"] = display_df["Gewinn (€)"].apply(lambda p: f"{p:.2f} €")

        history_table = mo.vstack([
            mo.md("---"),
            mo.md("### Bisheriger Spielverlauf (Historie)"),
            display_df
        ])

    if curr_state["phase"] == "start":
        slide_game = mo.vstack([
            mo.md("## Das OWL-Bowl Spiel\n\nStellen Sie sich vor, Sie betreiben einen Food-Truck auf dem Wochenmarkt in Ostwestfalen-Lippe. Sie verkaufen jeden Samstag eine begehrte regionale **'OWL-Bowl'**.\n\n"
                  "**Die harten Fakten:**\n"
                  "- **Materialkosten:** 3.00 € pro Bowl\n"
                  "- **Verkaufspreis:** 10.00 € pro Bowl\n"
                  "- **Restwert:** 2.50 € (Einige Zutaten können am nächsten Tag weiterverwendet werden)\n\n"
                  "**Ihre resultierende Kostenstruktur:**\n"
                  "- Kosten bei Überbestand: **0.50 €** (3.00€ - 2.50€)\n"
                  "- Entgangener Gewinn bei Unterbestand: **7.00 €** (10.00€ - 3.00€)\n\n"
                  "Spielen Sie 10 Wochen durch. Vor jeder Woche sehen Sie die Wetterprognose und ob ein lokales Event ansteht. Versuchen Sie, den Gesamtgewinn so hoch wie möglich zu halten!"
                 ),
            start_btn
        ])

    elif curr_state["phase"] == "play":
        week_idx = curr_state["week_idx"]
        df_play = curr_state["df"]
        row_play = df_play.iloc[week_idx]

        slide_game = mo.vstack([
            mo.md(f"## Woche {week_idx + 1} von 10"),
            mo.md(f"**Wettervorhersage & Kontext:**\n"
                  f"- Temperatur: **{row_play['temp']:.1f} °C**\n"
                  f"- Regen wahrscheinlich: **{'Ja' if row_play['rain'] else 'Nein'}**\n"
                  f"- Lokales Event: **{'Ja' if row_play['event'] else 'Nein'}**\n"
                  f"- Ferien/Feiertag: **{'Ja' if row_play['holiday'] else 'Nein'}**"
                 ),
            mo.md("Wie viele Bowls bereiten Sie vor?"),
            q_input,
            submit_btn,
            history_table if history_table is not None else mo.md("")
        ])

    elif curr_state["phase"] == "feedback":
        res = curr_state["last_result"]
        fb_week_idx = curr_state["week_idx"]

        q = res["Bestellmenge"]
        d = res["Nachfrage"]
        profit = res["Gewinn (€)"]

        if q > d:
            calc_text = (
                f"**Überbestand:** Sie haben **{q}** Bowls vorbereitet, aber die Nachfrage lag nur bei **{d}**.\n"
                f"- **{q - d} Bowls** wurden nicht verkauft.\n"
                f"- Kosten pro übrig gebliebener Bowl: **0.50 €** (Material 3.00€ - Restwert 2.50€).\n"
                f"- **Umsatz:** 10.00€ * {d} = **{10.00*d:.2f} €**.\n"
                f"- **Materialkosten:** 3.00€ * {q} = **{3.00*q:.2f} €**.\n"
                f"- **Restwert:** 2.50€ * {q-d} = **{2.50*(q-d):.2f} €**.\n"
                f"- **Gewinn:** 10.00€ * {d} (Umsatz) - 3.00€ * {q} (Kosten) + 2.50€ * {q-d} (Restwert) = **{profit:.2f} €**."
            )
        elif q < d:
            calc_text = (
                f"**Unterbestand:** Sie haben **{q}** Bowls vorbereitet bei einer Nachfrage von **{d}**.\n"
                f"- **{d - q} Kunden** konnten nicht bedient werden.\n"
                f"- Entgangener Gewinn pro nicht verkaufter Bowl: **7.00 €** (Verkauf 10.00€ - Material 3.00€).\n"
                f"- **Umsatz:** 10.00€ * {q} = **{10.00*q:.2f} €**.\n"
                f"- **Materialkosten:** 3.00€ * {q} = **{3.00*q:.2f} €**.\n"
                f"- **Gewinn:** 10.00€ * {q} (Umsatz) - 3.00€ * {q} (Kosten) = **{profit:.2f} €**."
            )
        else:
            calc_text = (
                f"**Perfekte Punktlandung!** Sie haben exakt **{q}** Bowls vorbereitet und alle **{d}** Bowls verkauft.\n"
                f"- **Umsatz:** 10.00€ * {q} = **{10.00*q:.2f} €**.\n"
                f"- **Materialkosten:** 3.00€ * {q} = **{3.00*q:.2f} €**.\n"
                f"- **Gewinn:** 10.00€ * {q} (Umsatz) - 3.00€ * {q} (Kosten) = **{profit:.2f} €**."
            )

        slide_game = mo.vstack([
            mo.md(f"## Ergebnis für Woche {res['Woche']}"),
            mo.md(f"- Ihre Bestellmenge: **{q}**\n"
                  f"- Tatsächliche Nachfrage: **{d}**\n"
                  f"### Erzielter Gewinn: {profit:.2f} €"
                 ),
            mo.md("---"),
            mo.md("### Gewinn-Berechnung:"),
            mo.md(calc_text),
            mo.md(""),
            next_btn,
            history_table if history_table is not None else mo.md("")
        ])

    elif curr_state["phase"] == "end":
        history_df = pd.DataFrame(curr_state["history"])
        total_profit = history_df["Gewinn (€)"].sum()

        slide_game = mo.vstack([
            mo.md("## Spielende!"),
            mo.md(f"Ihr erzielter Gesamtgewinn: **{total_profit:.2f} €**"),
            history_table,
            mo.md(""),
            restart_btn
        ])
    return (slide_game,)


@app.cell
def _(slide_game):
    slide_game
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dinge zum Mitnehmen

    - Entscheiden unter Unsicherheit bedeutet, Wahrscheinlichkeitsverteilungen zu gestalten
    - Bessere Prognosen führen nicht automatische zu besseren Entscheidungen
    - Man sollte Unsicherheit bis zum letzen Analyseschritt "durchhalten"
    - ML kann helfen, Unsicherheit zu reduzieren, aber eine Restunsicherheit bleibt und sollte in Entscheidungsansatz explizit einfließen
    - Ein Unsicherheitssimulator ist oftmals die Grundlage für die Bewertung von Entscheidungen und Politiken unter Unsicherheit
    - Entscheidungen, die auf Mittelwerten basieren, sind im Mittel suboptimal (das selbe gilt für Entscheidungen, die auf Punktschätzern basieren)
    """)
    return


if __name__ == "__main__":
    app.run()
