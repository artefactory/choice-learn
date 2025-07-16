import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from data import SyntheticDataGenerator
from model import BaseModel



data_gen = SyntheticDataGenerator()
n_baskets = 3000
baskets = data_gen.generate_synthetic_dataset(n_baskets)

model = BaseModel()
model.fit(baskets, repr = True, loss_type = "nce")    

