# This is the specific code for the case of discontinuous control
## ITERATIONS = 100000 in fit_dde

import numpy as np
from generate_data_wave_discontinuous import generate_wave_X_y
from fit_deeoponet_wave import fit_dde
from pathlib import Path
import shutil

ROOT_DIRECTORY = Path(".") 
RESULTS_DIRECTORY = ROOT_DIRECTORY / "results" / "wave_discontinuous_n_sensors_11"
print(RESULTS_DIRECTORY.absolute())
generar = True
n_sensors_x_list = [11]
SIGMA_POS = 1
SIGMA_VEL = 1
n_funciones = [10 ** 2, 5 * 10 ** 2, 10 ** 3, 5 * 10 ** 3, 10 ** 4]
n_funciones = [5 * 10 ** 4]
l_pos = np.array([0.03125])
l_vel = l_pos
N = {0.5: 3, 0.25: 5, 0.125: 8, 0.0625: 14, 0.03125: 28}
l_posvel = [(lp, l_vel[i]) for i, lp in enumerate(l_pos)]
p = 100  # dimension of the trunk net

for l_p, l_v in l_posvel:
    results_path = RESULTS_DIRECTORY / f"l_pos_{l_p}_l_vel_{l_v}"
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    for n_sensors_x in n_sensors_x_list:
        # generating data
        X = dict()
        y = dict()
        seeds = [31415, 314159]
        for i, set in enumerate(["train", "test"]):
            X[set], y[set] = generate_wave_X_y(
                l_p,
                l_v,
                n_sensors_x=n_sensors_x,
                sigma_pos=SIGMA_POS,
                sigma_vel=SIGMA_VEL,
                length_KL_expansion_pos=N[l_p],
                length_KL_expansion_vel=N[l_v],
                n_samples_functions=n_funciones[-1],
                seed=seeds[i],
            )
            print(
                f"""data generation completed l_pos {l_p}, l_vel {l_v},
                        n_sensors_x: {n_sensors_x}, n: {n_funciones[-1]}"""
            )
        y_train_long = y["train"]
        y_test_long = y["test"]
        X_train_long = X["train"]
        X_test_long = X["test"]

        previous_save_path = None
        previous_best_step = None
        previous_model = None
        for i_n, n in enumerate(n_funciones):
            save_path = results_path / f"n_sensors_x_{n_sensors_x}" / f"n_{n}"
            print(save_path.absolute())
            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)
            predict_plot_path = save_path #/ "predict_check"
            if not predict_plot_path.exists():
                predict_plot_path.mkdir(parents=True, exist_ok=True)
            y_train = y_train_long[: (n_sensors_x * n)]
            y_test = y_test_long[: (n_sensors_x * n)]
            X_train = X_train_long[: (n_sensors_x * n), :]
            X_test = X_test_long[: (n_sensors_x * n), :]
            _, best_step, model = fit_dde(
                X_train,
                y_train,
                X_test,
                y_test,
                p,
                n_sensors_x,
                save_path,
                predict_plot_path,
                previous_save_path,
                previous_best_step,
                previous_model,
            )
            previous_save_path = save_path
            previous_best_step = best_step
            previous_model = model
            print(
                f"""fit and predict completed l_pos {l_p}, l_vel {l_v},
                                    n_sensors_x: {n_sensors_x}, n: {n}"""
            )
