import shap
import matplotlib.pyplot as plt
import os

# this generates SHAP explainability visualizations for the best model of each training run
def generate_shap_analysis(best_model, X_train, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    
    print("Generating SHAP explainability plots...")
    try:
        # SHAP requires background data
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_train)

        # Summary plot
        shap.summary_plot(shap_values, X_train, show=False)
        summary_path = os.path.join(run_dir, "shap_summary_plot.png")
        plt.tight_layout()
        plt.savefig(summary_path, bbox_inches="tight", dpi=300)
        plt.close()

        # Bar plot (average absolute SHAP values)
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        bar_path = os.path.join(run_dir, "shap_bar_plot.png")
        plt.tight_layout()
        plt.savefig(bar_path, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"Saved SHAP plots → {summary_path} & {bar_path}")
        return summary_path, bar_path

    except Exception as e:
        print(f"⚠️ SHAP analysis failed: {e}")
        return None, None
