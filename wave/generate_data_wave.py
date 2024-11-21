import numpy as np

# from pathos.pools import ProcessPool
from sklearn import gaussian_process as gp
from utilities import compute_eig
from numpy.random import default_rng


def generate_wave_X_y(
    l_pos,
    l_vel,
    n_sensors_x,
    sigma_pos,
    sigma_vel,
    length_KL_expansion_pos,
    length_KL_expansion_vel,
    n_samples_functions,
    seed,
):
    rng = default_rng(seed)
    kernel_pos = sigma_pos ** 2 * gp.kernels.RBF(
        length_scale=l_pos
    )  # squared exponential covariance
    kernel_vel = sigma_vel ** 2 * gp.kernels.RBF(
        length_scale=l_vel
    )  # squared exponential covariance
    eigval_pos, eigfun_pos = compute_eig(
        kernel_pos, length_KL_expansion_pos, n_sensors_x, eigenfunction=True
    )
    eigval_vel, eigfun_vel = compute_eig(
        kernel_vel, length_KL_expansion_vel, n_sensors_x, eigenfunction=True
    )
    n_sensors_t = 2 * n_sensors_x - 1
    initial_position = np.zeros((n_samples_functions, n_sensors_x))
    initial_velocity = np.zeros((n_samples_functions, n_sensors_x))
    repetition_initial_position = np.zeros(
        (n_samples_functions, n_sensors_t, n_sensors_x)
    )
    repetition_initial_velocity = np.zeros(
        (n_samples_functions, n_sensors_t, n_sensors_x)
    )
    control = np.zeros((n_samples_functions, n_sensors_t))

    x = np.linspace(0, 1, n_sensors_x)
    for k in range(n_samples_functions):
        normal_gaussian_pos = rng.standard_normal(length_KL_expansion_pos)
        KL_matrix_pos = np.sqrt(eigval_pos) * eigfun_pos * normal_gaussian_pos
        initial_position[k, :] = np.sum(KL_matrix_pos, axis=1)
        repetition_initial_position[k, :, :] = np.tile(
            initial_position[k, :], (n_sensors_t, 1)
        )
        normal_gaussian_vel = rng.standard_normal(length_KL_expansion_vel)
        KL_matrix_vel = np.sqrt(eigval_vel) * eigfun_vel * normal_gaussian_vel
        initial_velocity[k, :] = np.sum(KL_matrix_vel, axis=1)
        repetition_initial_velocity[k, :, :] = np.tile(
            initial_velocity[k, :], (n_sensors_t, 1)
        )
        integral_d = np.zeros(n_sensors_x)
        for j in range(n_sensors_x):
            z = initial_velocity[k, :]
            integral_d[j] = np.trapz(z[j:], x[j:])
        y1 = 0.5 * initial_position[k, ::-1] + 0.5 * integral_d[::-1]
        y2 = -0.5 * initial_position[k, 1:] + 0.5 * integral_d[1:]
        control[k, :] = np.hstack((y1, y2))

    y = control.reshape(n_samples_functions * n_sensors_t, 1)
    repetition_initial_position = np.reshape(
        repetition_initial_position, (n_samples_functions * n_sensors_t, n_sensors_x)
    )  # replicating datum y^0 in all discrete times
    repetition_initial_velocity = np.reshape(
        repetition_initial_velocity, (n_samples_functions * n_sensors_t, n_sensors_x)
    )  # replicating datum y^1 in all discrete times
    initial_data = np.concatenate(
        (repetition_initial_position, repetition_initial_velocity), axis=1
    )  # joining y^0 and y^1
    t = np.linspace(0, 2, n_sensors_t)
    t = np.vstack([t] * n_samples_functions)
    t = np.reshape(t, (n_sensors_t * n_samples_functions, 1))
    X = np.concatenate((initial_data, t), axis=1)  # features
    return X, y
