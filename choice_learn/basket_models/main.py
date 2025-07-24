"""Main script to generate synthetic data and train the model for item recommendation."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from .attn_model import AttnModel
from .synthetic_dataset import SyntheticDataGenerator

# Parameters

n_baskets = 1000
embedding_dim = 3
K_noise = 7

# Generate synthetic dataset

data_gen = SyntheticDataGenerator()
data_gen.instantiate(
    proba_complementary_items=0.7,
    proba_neutral_items=0.3,
    noise_proba=0.15,
)
trip_dataset = data_gen.generate_trip_dataset(n_baskets)



# Instantiate and train the model

model = AttnModel()
model.instantiate(
    n_items=data_gen.assortment_matrix.shape[1],
    embedding_dim=embedding_dim,
    k_noise=K_noise,
)
model.fit(trip_dataset, repr=True, loss_type="nce")


# eval_dataset = data_gen.generate_synthetic_dataset(100)
# model.evaluate(eval_dataset)
