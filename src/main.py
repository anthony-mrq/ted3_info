from modules.loaders.housing import fetch_data


def main():
    source = "data/housing.csv"
    data = fetch_data(source)
    print(type(data))


if __name__ == "__main__":
    main()
