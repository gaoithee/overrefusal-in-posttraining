import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_full_evolution(base_dir="results"):
    models = {
        "OLMo-2": "olmo2",
        "OLMo-3": "olmo3",
        "OLMo-3 Think": "olmo3_think"
    }
    
    STAGE_ORDER = ["base", "sft", "dpo", "final"]
    plots_dir = Path(base_dir) / "plots_comparison" / "full_evolution"
    plots_dir.mkdir(exist_ok=True, parents=True)

    for model_name, folder in models.items():
        path = Path(base_dir) / folder / "raw_results.csv"
        if not path.exists():
            print(f"⚠️ Salto {model_name}: file non trovato.")
            continue

        df = pd.read_csv(path)
        
        # Preprocessing standard
        df['stage'] = df['checkpoint'].apply(lambda x: str(x).split("__")[0])
        df['prompt_type'] = df['checkpoint'].apply(lambda x: str(x).split("__")[1] if "__" in str(x) else "none")
        
        # Filtro: solo stage corretti e configurazione standard (senza system prompt aggiuntivi)
        df_model = df[(df['stage'].isin(STAGE_ORDER)) & (df['prompt_type'] == 'none')].copy()
        
        if df_model.empty:
            continue

        # Generiamo una heatmap per OGNI dataset (source) trovato per quel modello
        sources = df_model['source'].unique()
        
        for source in sources:
            source_df = df_model[df_model['source'] == source]
            
            # Creazione Pivot Table: Categoria vs Stage
            pivot = source_df.pivot_table(
                index="category", 
                columns="stage", 
                values="predicted_refusal", 
                aggfunc="mean"
            )
            
            # Ordinamento e conversione in %
            pivot = pivot.reindex(columns=STAGE_ORDER) * 100

            # --- PLOT HEATMAP ---
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", vmin=0, vmax=100)
            
            plt.title(f"Evoluzione Categorie: {model_name} | Dataset: {source}", fontsize=15, pad=15)
            plt.xlabel("Post-Training Stage")
            plt.ylabel("Category")
            
            # Nome file pulito: evolution_olmo3_think_toxicchat.png
            file_name = f"evolution_{model_name.lower().replace(' ', '_')}_{source}.png"
            save_path = plots_dir / file_name
            plt.savefig(save_path, bbox_inches="tight", dpi=180)
            plt.close()
            
            print(f"✅ Generata: {file_name}")

if __name__ == "__main__":
    plot_full_evolution()
