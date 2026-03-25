# Titanic Competition — TensorFlow Decision Forests

A solution for the [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition using **TensorFlow Decision Forests (TF-DF)** and Gradient Boosted Trees.

## Approach

The notebook walks through four progressively stronger modeling strategies:

1. **Baseline GBT** — A Gradient Boosted Trees model with default parameters to establish a starting point.
2. **Tuned GBT** — Hand-picked hyperparameters including sparse oblique splits, lower shrinkage (0.05), and 2,000 trees.
3. **Automated Hyperparameter Search** — A `RandomSearch` tuner over 1,000 trials exploring tree depth, shrinkage, split strategies, and more.
4. **Ensemble of 100 GBTs** — Averaging predictions across 100 models trained with different random seeds and `honest=True` regularization to reduce variance.

## Feature Engineering

- **Name tokenization** — Passenger names are split into individual tokens (e.g., titles like *Mr.*, *Mrs.*) and consumed natively by TF-DF as categorical-set features.
- **Ticket parsing** — Ticket strings are split into a prefix (e.g., `STON/O2.`) and a numeric part, giving the model access to ticket group information.
- **Feature selection** — `PassengerId` and raw `Ticket` are excluded from training.

## Requirements

- Python 3
- TensorFlow
- TensorFlow Decision Forests
- NumPy
- Pandas

## Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/kaggle-titanic-tfdf.git
   ```
2. Open the notebook in Jupyter, VS Code, or upload it directly to [Kaggle](https://www.kaggle.com/).
3. Make sure the Titanic dataset is available at the expected path, or update the `pd.read_csv()` calls to point to your local copy.

## Dataset

The dataset is provided by the Kaggle competition and is **not included** in this repo. You can download it from:  
https://www.kaggle.com/competitions/titanic/data

## Resources

- [TensorFlow Decision Forests documentation](https://www.tensorflow.org/decision_forests)
- [TF-DF automated hyperparameter tuning tutorial](https://www.tensorflow.org/decision_forests/tutorials/automatic_tuning_colab)
- [Kaggle Titanic competition page](https://www.kaggle.com/competitions/titanic)

## License

MIT
