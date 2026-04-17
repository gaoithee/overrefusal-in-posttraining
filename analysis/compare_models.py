import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def compare_all_results(base_dir="results"):
    models = {
        "OLMo-2": "olmo2",
        "OLMo-3": "olmo3",
        "OLMo-3 Think": "olmo3_think"
    }
    
    STAGE_ORDER = ["base", "sft", "dpo", "final"]
    all_dfs = []

    # 1. Caricamento e Preprocessing
    for name, folder in models.items():
        path = Path(base_dir) / folder / "raw_results.csv"
        if path.exists():
            df = pd.read_csv(path)
            # Estrazione stage e prompt
            df['stage'] = df['checkpoint'].apply(lambda x: str(x).split("__")[0])
            df['prompt_type'] = df['checkpoint'].apply(lambda x: str(x).split("__")[1] if "__" in str(x) else "none")
            
            df = df[df['stage'].isin(STAGE_ORDER)].copy()
            df['model_name'] = name
            all_dfs.append(df)
        else:
            print(f"⚠️ Salto {name}: {path} non trovato.")

    if not all_dfs:
        print("❌ Nessun dato trovato.")
        return

    # UNIONE E FIX INDICE (Risolve il ValueError: duplicate labels)
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['refusal_rate'] = full_df['predicted_refusal'] * 100
    full_df['stage'] = pd.Categorical(full_df['stage'], categories=STAGE_ORDER, ordered=True)

    sns.set_theme(style="whitegrid")
    plots_dir = Path(base_dir) / "plots_comparison"
    plots_dir.mkdir(exist_ok=True)

    # --- PLOT 1: Evoluzione Post-Training (Linee) ---
    subset_none = full_df[full_df['prompt_type'] == 'none'].copy()
    g_evolution = sns.relplot(
        data=subset_none, 
        x="stage", y="refusal_rate", hue="model_name",
        col="source", kind="line", marker="o", linewidth=3, markersize=8,
        facet_kws={'sharey': False}, height=5, aspect=1.1,
        hue_order=["OLMo-2", "OLMo-3", "OLMo-3 Think"]
    )
    
    g_evolution.set_axis_labels("Alignment Stage", "Refusal Rate (%)")
    g_evolution.set_titles("{col_name}", size=14, weight='bold')

    for ax in g_evolution.axes.flat:
        for line in ax.lines:
            x, y = line.get_data()
            for i, val in enumerate(y):
                ax.annotate(f'{val:.1f}%', (x[i], y[i]), textcoords="offset points", 
                            xytext=(0, 10), ha='center', fontsize=8, fontweight='bold')

    g_evolution.savefig(plots_dir / "alignment_evolution_comparison.png", bbox_inches="tight", dpi=200)

    # --- PLOT 2: Confronto Finale (Barre) ---
    final_df = full_df[full_df['stage'] == 'final'].copy()
    # Resettiamo l'indice anche qui per sicurezza estrema
    final_df = final_df.reset_index(drop=True)

    g_final = sns.catplot(
        data=final_df, x="model_name", y="refusal_rate", hue="prompt_type",
        col="source", kind="bar", palette="muted",
        height=5, aspect=1.2, sharey=False,
        order=["OLMo-2", "OLMo-3", "OLMo-3 Think"]
    )
    
    g_final.set_axis_labels("", "Refusal Rate (%)")
    g_final.set_titles("Final Checkpoint: {col_name}", size=14, weight='bold')
    
    for ax in g_final.axes.flat:
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=9, fontweight='bold')

    g_final.savefig(plots_dir / "final_models_comparison.png", bbox_inches="tight", dpi=200)

    print(f"\n✅ Analisi completata!")
    print(f"📈 Grafici salvati in: {plots_dir}")

    # --- TABELLA RIASSUNTIVA ---
    summary_table = subset_none.pivot_table(
        index=["source", "stage"], 
        columns="model_name", 
        values="refusal_rate"
    ).reindex(STAGE_ORDER, level="stage")
    print("\n--- REFUSAL RATE (%) PER STAGE E MODELLO (No Prompt) ---")
    print(summary_table.round(2))

if __name__ == "__main__":
    compare_all_results()
