import pandas as pd
from src.data import simulate_history
from src.costs import newsvendor_cost, NewsvendorCosts

def render_game(mo, state, set_state):
    costs = NewsvendorCosts(overage_cost=1.0, underage_cost=5.5)
    
    if state["phase"] == "start":
        seed_input = mo.ui.number(value=state["seed"], label="Spiele-Seed (Zahl)", step=1)
        
        def start_game(*args):
            # Generate 10 weeks of data
            # We use simulate_history for a single year (52 weeks) and take the first 10
            df_full = simulate_history(n_years=1, seed=int(seed_input.value))
            df_game = df_full.head(10).copy()
            set_state({
                "phase": "play",
                "seed": int(seed_input.value),
                "week_idx": 0,
                "df": df_game,
                "history": [],
                "last_result": None
            })
            
        start_btn = mo.ui.button(label="Spiel starten", on_click=start_game, kind="success")
        
        return mo.vstack([
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
            seed_input,
            start_btn
        ], align="center")
        
    elif state["phase"] == "play":
        week_idx = state["week_idx"]
        df = state["df"]
        row = df.iloc[week_idx]
        
        # We need a form or just a number input. 
        # Using a form prevents accidental submissions and handles "Enter" key nicely.
        q_input = mo.ui.number(value=0, step=1, label="Ihre Bestellmenge für diese Woche:")
        form = mo.ui.form(element=q_input, submit_button_label="Entscheidung abschicken")
        
        # When the form is submitted, its value changes from None to the input value
        # But wait, Marimo forms don't trigger callbacks via on_change easily if we want to update state.
        # It's better to use a standard number input and a button if we update state in the callback.
        # Actually, `mo.ui.button` can't easily read `q_input.value` inside its callback unless `q_input` is defined in an upstream cell.
        # This is a known Marimo pattern: callbacks in the same cell see the INITIAL value of the widget, not the updated one.
        # TO FIX THIS:
        # We must use `form = mo.ui.form(...)` and handle the form submission in a SEPARATE cell, 
        # OR we can just pass the state into the callback? No, the input value is what we need.
        pass

