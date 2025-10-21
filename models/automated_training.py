from training_models import train_and_evaluate_models
def main():
    data_file_path = "training_data/training_dataset.csv"
    output_path = "models/baseline_metrics.csv"

    print("Starting automated training pipeline...\n")
    best_model, best_model_name, metrics = train_and_evaluate_models(data_file_path = data_file_path)
    print("Training of the models completed!\n\n")

if __name__ == "__main__":
    main()
