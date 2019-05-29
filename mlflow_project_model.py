import mlflow.sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import click


@click.command()
@click.option("--seed", default=0, help="Seed for the experiment")
def main(seed):
  # Set the tensorflow experiment
  mlflow.set_experiment("reproductible_experiment")

  # Start mlflow run
  with mlflow.start_run() as run:
    print("Starting run", run.info.run_uuid)

    # Log params
    mlflow.log_param("seed", seed)

    # Set seed
    np.random.seed(seed)

    # Download iris dataset
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    # Split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Train model
    classifier = LogisticRegression(multi_class="multinomial", solver="lbfgs")

    classifier.fit(X, y)

    # Log model artifact
    mlflow.sklearn.log_model(classifier, "model")

    # Log metric
    mlflow.log_metric("acc", classifier.score(X_test, y_test))


if __name__ == "__main__":
  main()
