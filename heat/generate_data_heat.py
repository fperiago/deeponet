import numpy as np

# from pathos.pools import ProcessPool
from sklearn import gaussian_process as gp
from utilities import compute_eig
from numpy.random import default_rng
from heat_1d import heat


def generate_heat_X_y(
    l,
    n_sensors_x,
    sigma,
    length_KL_expansion,
    n_samples_functions,
    seed,
):
    rng = default_rng(seed)
    kernel = sigma ** 2 * gp.kernels.RBF(
        length_scale=l
    )  # squared exponential covariance
    eigval, eigfun = compute_eig(
        kernel, length_KL_expansion, n_sensors_x, eigenfunction=True
    )
    n_sensors_t = n_sensors_x 
    initial_temperature = np.zeros((n_samples_functions, n_sensors_x))
    repetition_initial_temperature = np.zeros(
        (n_samples_functions, n_sensors_t, n_sensors_x)
    )
    control = np.zeros((n_samples_functions, n_sensors_t))

    x = np.linspace(0, 1, n_sensors_x)
    for k in range(n_samples_functions):
        normal_gaussian = rng.standard_normal(length_KL_expansion)
        KL_matrix_pos = np.sqrt(eigval) * eigfun * normal_gaussian
        initial_temperature[k, :] = np.sum(KL_matrix_pos, axis=1)
        repetition_initial_temperature[k, :, :] = np.tile(
            initial_temperature[k, :], (n_sensors_t, 1)
        )    
        control[k, :] = heat(x, initial_temperature[k, :]) 
        print(f"n_function = {k}")

    y = control.reshape(n_samples_functions * n_sensors_t, 1)
    repetition_initial_temperature = np.reshape(
        repetition_initial_temperature, (n_samples_functions * n_sensors_t, n_sensors_x)
    )  # replicating y^0 in all discrete times
    
    t = np.linspace(0, 0.5, n_sensors_t)
    t = np.vstack([t] * n_samples_functions)
    t = np.reshape(t, (n_sensors_t * n_samples_functions, 1))
    X = np.concatenate((repetition_initial_temperature, t), axis=1)  # features
    return X, y
