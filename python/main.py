import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from DataGen import SyntheticDataGenerator
from AttnModel import BaseModel


data_gen = SyntheticDataGenerator()
n_baskets = 150
baskets = data_gen.generate_synthetic_dataset(n_baskets)

model = BaseModel()
model.fit(baskets, repr=True, loss_type="nce")
