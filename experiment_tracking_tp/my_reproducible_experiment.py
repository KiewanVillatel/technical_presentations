import click
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


@click.command()
@click.option("--seed", default=0, help="Seed for the experiment")
@click.option("--max_iter", default=200, help="Max training iterations")
@click.option("--n_hidden_layer", default=1, help="Number of hidden layers")
@click.option("--hidden_layers_size", default=300, help="Size of the hidden layers")
def main(seed, max_iter, hidden_layers_size, n_hidden_layer):
  # Set seed
  np.random.seed(seed)

  # Download diabetes dataset
  diabetes = datasets.load_diabetes()

  X = diabetes.data
  y = diabetes.target

  # Split train test
  X_train, X_test, y_train_valid, y_test = train_test_split(X, y, shuffle=False)

  # Train model
  classifier = MLPRegressor(hidden_layer_sizes=[hidden_layers_size] * n_hidden_layer,
                            max_iter=max_iter,
                            solver="lbfgs")

  # Fit classifier
  classifier.fit(X, y)

  # Evaluate performances
  r2 = classifier.score(X_test, y_test)

  print("Test R2", r2)


if __name__ == "__main__":
  main()
