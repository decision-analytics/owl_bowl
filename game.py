import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from src.data import simulate_history
    from src.costs import newsvendor_cost, NewsvendorCosts
    return mo, pd, np, simulate_history, newsvendor_cost, NewsvendorCosts


@app.cell
def _(mo):
    params = mo.query_params()
    include_spikes = params.get("spikes", "false").lower() in ("true", "1", "yes")
    game_seed = int(params.get("seed", "42"))
    return game_seed, include_spikes


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
    game_seed,
    get_game_state,
    include_spikes,
    mo,
    newsvendor_cost,
    set_game_state,
    simulate_history,
):
    # Static UI widgets and callbacks defined ONCE.
    # This cell does not depend on get_game_state(), so it never re-runs on state changes!
    
    spikes_switch = mo.ui.checkbox(value=include_spikes, label="Seltene Nachfragespitzen (Spikes) im Spiel simulieren")
    seed_input = mo.ui.number(value=game_seed, step=1, label="Spiel-Seed (Zufallszahlen-Startwert):")
    
    def on_start_click(*args):
        df_full = simulate_history(n_years=1, seed=int(seed_input.value), include_spikes=spikes_switch.value)
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
        
        costs_obj = NewsvendorCosts(overage_cost=1.0, underage_cost=5.5)
        cost_val = float(newsvendor_cost(sub_q, sub_d, costs_obj))
        
        # Profit = 8.50 * min(q, d) - 3.00 * q + 2.00 * max(0, q - d)
        profit_val = 8.50 * min(sub_q, sub_d) - 3.00 * sub_q + 2.00 * max(0, sub_q - sub_d)
        
        sub_result = {
            "Woche": week_idx + 1,
            "Temperatur": float(row_play['temp']),
            "Regen": "Ja" if row_play['rain'] else "Nein",
            "Event": "Ja" if row_play['event'] else "Nein",
            "Ferien": "Ja" if row_play['holiday'] else "Nein",
            "Bestellmenge": int(sub_q),
            "Nachfrage": int(sub_d),
            "Kosten (€)": float(cost_val),
            "Profit (€)": float(profit_val)
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
    
    return start_btn, q_input, submit_btn, next_btn, restart_btn, spikes_switch, seed_input


@app.cell
def _(
    get_game_state,
    mo,
    next_btn,
    pd,
    q_input,
    restart_btn,
    seed_input,
    spikes_switch,
    start_btn,
    submit_btn,
):
    # Pure rendering cell. Re-runs on every state change, but only assembles the static elements.
    curr_state = get_game_state()
    game_ui = None
    
    # Helper to generate history table display
    history_table = None
    if curr_state["history"]:
        history_df = pd.DataFrame(curr_state["history"])
        display_df = history_df.copy()
        display_df["Temperatur"] = display_df["Temperatur"].apply(lambda t: f"{t:.1f} °C")
        display_df["Kosten (€)"] = display_df["Kosten (€)"].apply(lambda c: f"{c:.2f} €")
        display_df["Profit (€)"] = display_df["Profit (€)"].apply(lambda p: f"{p:.2f} €")
        
        history_table = mo.vstack([
            mo.md("---"),
            mo.md("### Bisheriger Spielverlauf (Historie)"),
            display_df
        ])
    
    if curr_state["phase"] == "start":
        game_ui = mo.vstack([
            mo.md("## Das OWL-Bowl Newsvendor Spiel\n\nStellen Sie sich vor, Sie betreiben einen Food-Truck auf dem Wochenmarkt in Ostwestfalen-Lippe. Sie verkaufen jeden Samstag eine begehrte regionale **'OWL-Bowl'**.\n\n"
                  "**Die harten Fakten:**\n"
                  "- **Materialkosten:** 3.00 € pro Bowl\n"
                  "- **Verkaufspreis:** 8.50 € pro Bowl\n"
                  "- **Restwert:** 2.00 € (Einige Zutaten können am nächsten Tag weiterverwendet werden)\n\n"
                  "**Ihre resultierende Kostenstruktur:**\n"
                  "- Kosten bei Überbestand: **1.00 €** (3.00€ - 2.00€)\n"
                  "- Entgangener Gewinn bei Unterbestand: **5.50 €** (8.50€ - 3.00€)\n\n"
                  "Spielen Sie 10 Wochen durch. Vor jeder Woche sehen Sie die Wetterprognose und ob ein lokales Event ansteht. Versuchen Sie, die Gesamtkosten so gering wie möglich zu halten!"
                 ),
            mo.md("---"),
            mo.md("### Spielmodus einstellen:"),
            spikes_switch,
            seed_input,
            mo.md("---"),
            start_btn
        ])
        
    elif curr_state["phase"] == "play":
        week_idx = curr_state["week_idx"]
        df_play = curr_state["df"]
        row_play = df_play.iloc[week_idx]
        
        game_ui = mo.vstack([
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
        cost = res["Kosten (€)"]
        profit = res["Profit (€)"]
        
        if q > d:
            calc_text = (
                f"**Überbestand:** Sie haben **{q}** Bowls vorbereitet, aber die Nachfrage lag nur bei **{d}**.\n"
                f"- **{q - d} Bowls** wurden nicht verkauft.\n"
                f"- Kosten pro übrig gebliebener Bowl: **1.00 €** (Material 3.00€ - Restwert 2.00€).\n"
                f"- **Berechnung:** {q - d} * 1.00 € = **{cost:.2f} €** Kosten.\n"
                f"- **Profit:** 8.50€ * {d} (Umsatz) - 3.00€ * {q} (Kosten) + 2.00€ * {q-d} (Restwert) = **{profit:.2f} €**."
            )
        elif q < d:
            calc_text = (
                f"**Unterbestand:** Sie haben **{q}** Bowls vorbereitet bei einer Nachfrage von **{d}**.\n"
                f"- **{d - q} Kunden** konnten nicht bedient werden.\n"
                f"- Entgangener Gewinn pro nicht verkaufter Bowl: **5.50 €** (Verkauf 8.50€ - Material 3.00€).\n"
                f"- **Berechnung:** {d - q} * 5.50 € = **{cost:.2f} €** Kosten.\n"
                f"- **Profit:** 8.50€ * {q} (Umsatz) - 3.00€ * {q} (Kosten) = **{profit:.2f} €**."
            )
        else:
            calc_text = (
                f"**Perfekte Punktlandung!** Sie haben exakt **{q}** Bowls vorbereitet und alle **{d}** Bowls verkauft.\n"
                f"- Keine zusätzlichen Kosten: **0.00 €**.\n"
                f"- **Profit:** 8.50€ * {q} (Umsatz) - 3.00€ * {q} (Kosten) = **{profit:.2f} €**."
            )
        
        game_ui = mo.vstack([
            mo.md(f"## Ergebnis für Woche {res['Woche']}"),
            mo.md(f"- Ihre Bestellmenge: **{q}**\n"
                  f"- Tatsächliche Nachfrage: **{d}**\n"
                  f"### Entstandene Kosten: {cost:.2f} €\n"
                  f"### Erzielter Profit: {profit:.2f} €"
                 ),
            mo.md("---"),
            mo.md("### Kosten- und Profit-Berechnung:"),
            mo.md(calc_text),
            mo.md(""),
            next_btn,
            history_table if history_table is not None else mo.md("")
        ])
        
    elif curr_state["phase"] == "end":
        history_df = pd.DataFrame(curr_state["history"])
        total_costs = history_df["Kosten (€)"].sum()
        total_profit = history_df["Profit (€)"].sum()
        
        game_ui = mo.vstack([
            mo.md("## Spielende!"),
            mo.md(f"Ihre Gesamtkosten über 10 Wochen: **{total_costs:.2f} €**\n\n"
                  f"Ihr erzielter Gesamtprofit: **{total_profit:.2f} €**"
                 ),
            history_table,
            mo.md(""),
            restart_btn
        ])
        
    return game_ui,

@app.cell
def _(game_ui):
    game_ui
