import mlflow.sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import click


@click.command()
@click.option("--seed", default=0, help="Seed for the experiment")
@click.option("--max_iter", default=100, help="Number of iterations")
def main(seed, max_iter):
  # Set the tensorflow experiment
  mlflow.set_experiment("reproductible_experiment")

  # Start mlflow run
  with mlflow.start_run() as run:
    print("Starting run", run.info.run_uuid)

    # Log params
    mlflow.log_param("seed", seed)
    mlflow.log_param("max_iter", max_iter)

    # Set seed
    np.random.seed(seed)

    # Download iris dataset
    print("Using Iris dataset")
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    # Split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Train model
    print("Training multinomial logistic regression model")
    classifier = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=max_iter)

    classifier.fit(X, y)

    # Log model artifact
    mlflow.sklearn.log_model(classifier, "model")

    acc = classifier.score(X_test, y_test)

    # Log metric
    mlflow.log_metric("acc", acc)

    print("Score on test set", acc)


if __name__ == "__main__":
  main()
