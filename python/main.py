import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from DataGen import SyntheticDataGenerator
from AttnModel import AttnModel

# Parameters

n_baskets = 1000
embedding_dim = 3
K_noise = 6

# Generate synthetic dataset

data_gen = SyntheticDataGenerator()
baskets = data_gen.generate_synthetic_dataset(n_baskets)
print("Generated Baskets:", baskets)

# Instantiate and train the model

model = AttnModel()
model.instantiate(
    n_items=len(data_gen.assortment),
    embedding_dim=embedding_dim,
    K_noise=K_noise
)
model.fit(baskets, repr=True, loss_type="nce")


