from python.model import BaseModel
from python.data import SyntheticDataGenerator

dataset_genrator = SyntheticDataGenerator()
dataset = dataset_genrator.generate_synthetic_dataset(n_baskets=100)

model1 = BaseModel()
model1.fit(dataset, repr = True)

