from data import SyntheticDataGenerator
from model import BaseModel
import tensorflow as tf

import matplotlib.pyplot as plt



data_gen = SyntheticDataGenerator()
n_baskets = 1000
baskets = data_gen.generate_synthetic_dataset(n_baskets)

P_ij_true = np.zeros((I, I))
for basket in baskets:
    for i in basket:
        for j in basket:
            if i != j:
                P_ij_true[i, j] += 1

P_ij_true_norm = P_ij_true / (np.sum(P_ij_true, axis=0, keepdims=True) + 1e-10)

print("Empirical P(i|j) from data:\n", np.round(P_ij_true_norm, 3))
plt.imshow(P_ij_true_norm, cmap='Blues')
plt.xlabel('Given j')
plt.ylabel('Empirical i')
plt.title('Empirical P(i|j) from baskets')
plt.colorbar()
plt.show()

print(baskets)


new_baskets = data_gen.generate_synthetic_dataset(n_baskets, padded=False)
print(new_baskets)
model = BaseModel()
model.fit(baskets, repr = True, loss_type = "bad")

