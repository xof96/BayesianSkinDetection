import numpy as np


def p(x, mean, cov, dim=3):
    cov_mat = np.diag(cov)
    inv_cov = np.linalg.inv(cov_mat)
    det_cov = np.linalg.det(cov_mat)
    a = 1 / np.sqrt(np.power(2 * np.pi, dim) * det_cov)
    b = -1 / 2 * np.transpose(x - mean) * inv_cov * (x - mean)
    return a * np.exp(b)


def total_p(x, means, covs, w, dim=3):
    n_weights = w.size
    total_p = 0
    for i in range(n_weights):
        curr_p = w[i] * p(x, means[i], covs[i], dim)
        total_p += curr_p
    return total_p


def is_skin(x, sk_means, sk_covs, sk_w, nsk_means, nsk_covs, nsk_w, th=0.5, dim=3):
    skin_p = total_p(x, sk_means, sk_covs, sk_w, dim)
    no_skin_p = total_p(x, nsk_means, nsk_covs, nsk_w, dim)
    c = skin_p / no_skin_p
    return c >= th



# CONSTANTS
SKIN_MEAN = np.array([[73.53, 29.94, 17.76],
                      [249.71, 233.94, 217.49],
                      [161.68, 116.25, 96.95],
                      [186.07, 136.62, 114.40],
                      [189.26, 98.37, 51.18],
                      [247.00, 152.20, 90.84],
                      [150.10, 72.66, 37.76],
                      [206.85, 171.09, 156.34],
                      [212.78, 152.82, 120.04],
                      [234.87, 175.43, 138.94],
                      [151.19, 97.74, 74.59],
                      [120.52, 77.55, 59.82],
                      [192.20, 119.62, 82.32],
                      [214.29, 136.08, 87.24],
                      [99.57, 54.33, 38.06],
                      [238.88, 203.08, 176.91]])

SKIN_COV = np.array([[765.40, 121.44, 112.80],
                     [39.94, 154.44, 396.05],
                     [291.03, 60.48, 162.85],
                     [274.95, 64.60, 198.27],
                     [633.18, 222.40, 250.69],
                     [65.23, 691.53, 609.92],
                     [408.63, 200.77, 257.57],
                     [530.08, 155.08, 572.79],
                     [160.57, 84.52, 243.90],
                     [163.80, 121.57, 279.22],
                     [425.40, 73.56, 175.11],
                     [330.45, 70.34, 151.82],
                     [152.76, 92.14, 259.15],
                     [204.90, 140.17, 270.19],
                     [448.13, 90.18, 151.29],
                     [178.38, 156.27, 404.99]])

SKIN_WEIGHT = np.array([0.0294,
                        0.0331,
                        0.0654,
                        0.0756,
                        0.0554,
                        0.0314,
                        0.0454,
                        0.0469,
                        0.0956,
                        0.0763,
                        0.1100,
                        0.0676,
                        0.0755,
                        0.0500,
                        0.0667,
                        0.0749])

NO_SKIN_MEAN = np.array([[254.37, 254.41, 253.82],
                         [9.39, 8.09, 8.52],
                         [96.57, 96.95, 91.53],
                         [160.44, 162.49, 159.06],
                         [74.98, 63.23, 46.33],
                         [121.83, 60.88, 18.31],
                         [202.18, 154.88, 91.04],
                         [193.06, 201.93, 206.55],
                         [51.88, 57.14, 61.55],
                         [30.88, 26.84, 25.32],
                         [44.97, 85.96, 131.95],
                         [236.02, 236.27, 230.70],
                         [207.86, 191.20, 164.12],
                         [99.83, 148.11, 188.17],
                         [135.06, 131.92, 123.10],
                         [135.96, 103.89, 66.88]])

NO_SKIN_COV = np.array([[2.77, 2.81, 5.46],
                        [46.84, 33.59, 32.48],
                        [280.69, 156.79, 436.58],
                        [355.98, 115.89, 591.24],
                        [414.84, 245.95, 361.27],
                        [2502.24, 1383.53, 237.18],
                        [957.42, 1766.94, 1582.52],
                        [562.88, 190.23, 447.28],
                        [344.11, 191.77, 433.40],
                        [222.07, 118.65, 182.41],
                        [651.32, 840.52, 963.67],
                        [225.03, 117.29, 331.95],
                        [494.04, 237.69, 533.52],
                        [955.88, 654.95, 916.70],
                        [350.35, 130.30, 388.43],
                        [806.44, 642.20, 350.36]])

NO_SKIN_WEIGHT = np.array([0.0637,
                           0.0516,
                           0.0864,
                           0.0636,
                           0.0747,
                           0.0365,
                           0.0349,
                           0.0649,
                           0.0656,
                           0.1189,
                           0.0362,
                           0.0849,
                           0.0368,
                           0.0389,
                           0.0943,
                           0.0477])
