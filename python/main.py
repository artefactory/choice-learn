from data import SyntheticDataGenerator
from model import BaseModel



data_gen = SyntheticDataGenerator()
n_baskets = 100
baskets = data_gen.generate_synthetic_dataset(n_baskets)
print(baskets)
new_baskets = data_gen.generate_synthetic_dataset(n_baskets, padded=False)
print(new_baskets)
model = BaseModel()
model.fit(baskets, repr = True)