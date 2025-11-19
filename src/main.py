from modules.loaders.housing import fetch_data
from modules.lr.mlr import LR


def main():
    source = "data/housing.csv"
    data = fetch_data(source)
    model = LR()
    model.train(data[0], data[1])
    print("Score:", model.get_score(data[0], data[1]))
    print("Predictions:", model.predict(data[0]))
    print("Intercept:", model.get_intercept())
    print("Coefficients:", model.get_coefficients())
    y_pred = model.predict(data[0])
    print("Predicted values:", y_pred)


if __name__ == "__main__":
    main()
