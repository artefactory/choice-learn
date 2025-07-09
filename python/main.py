from data import SyntheticDataGenerator
from model import BaseModel


data_gen = SyntheticDataGenerator()
baskets = data_gen.generate_synthetic_dataset(50)
print(baskets)
model1 = BaseModel()
model1.fit(baskets, repr = True)
