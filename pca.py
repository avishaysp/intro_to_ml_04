import random
from time import sleep

import matplotlib.pyplot as plt
from skeleton_pca import *

sections = ['c']

X, h, w = get_pictures_by_name('Colin Powell')
X_mean = np.mean(X, axis=0)


if 'b' in sections:
    U, S = PCA(np.array(X), 10)

    for i in range(10):
        eigenvector = U[i, :]
        plot_vector_as_image(eigenvector, h, w)

if 'c' in sections:
    rand_pics = random.sample(X, 5)
    l2_sums = []
    k_values = [1, 5, 10, 30, 50, 100]
    for k in k_values:
        U, S = PCA(np.array(X), k)
        l2_sum = 0
        for pic in rand_pics:
            compressed_pic = U.T @ (U @ pic)
            compressed_pic += X_mean
            if k == 100:
                plot_vector_as_image(pic, h, w, title="OG Pic")
                plot_vector_as_image(compressed_pic, h, w, title="COMP Pic")
            l2_sum += np.linalg.norm(pic - compressed_pic)
        l2_sums.append(l2_sum)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, l2_sums, marker='o', linestyle='-')
    plt.title('L2 Norm vs k')
    plt.xlabel('k')
    plt.grid(True)
    plt.show()
