def create_model_card(metadata, model_card_path):
    try:
        with open(model_card_path, "w") as f:
            f.write("📘 MODEL CARD\n")
            f.write("====================\n\n")
            f.write(f"Model Name: {metadata['best_model']}\n")
            f.write(f"Timestamp: {metadata['timestamp']}\n")
            f.write("\n---\n**Metrics:**\n")
            for k, v in metadata["metrics"].items():
                f.write(f"- {k}: {v:.4f}\n")

            f.write("\n---\n**SHAP Explainability Files:**\n")
            if "shap_summary_plot" in metadata:
                f.write(f"- Summary Plot: {metadata['shap_summary_plot']}\n")
            if "shap_bar_plot" in metadata:
                f.write(f"- Bar Plot: {metadata['shap_bar_plot']}\n")

            f.write("\n---\n**Feature Store:**\n")
            f.write("Features pulled from Hopsworks Feature Store.\n")

            # will add later if need be
            # f.write("\n---\n**Evaluation Summary:**\n")
            # passed, message = evaluate_model_performance(metadata["metrics"])
            # f.write(message + "\n")

        print(f"✅ Model card created at {model_card_path}")
    except Exception as e:
        print(f"⚠️ Model card generation failed: {e}")
