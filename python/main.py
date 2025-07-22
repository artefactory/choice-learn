import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from DataGen import SyntheticDataGenerator
from AttnModel import AttnModel

# Parameters

n_baskets = 1000
embedding_dim = 3
K_noise = 7

# Generate synthetic dataset

data_gen = SyntheticDataGenerator()
baskets = data_gen.generate_trip_dataset(n_baskets)

# Instantiate and train the model

model = AttnModel()
model.instantiate(
    n_items=data_gen.assortment_matrix.shape[1],
    embedding_dim=embedding_dim,
    K_noise=K_noise
)
model.fit(baskets, repr=True, loss_type="nce")

#eval_dataset = data_gen.generate_synthetic_dataset(100)
#model.evaluate(eval_dataset)


