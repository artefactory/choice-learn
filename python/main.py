from data import SyntheticDataGenerator

data_gen = SyntheticDataGenerator()
baskets = data_gen.generate_synthetic_dataset(5)
print(baskets)
