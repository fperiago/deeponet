# This is the main code for Expriment 2, concerning the heat equation.
# Data generation takes time. Instead, you may run fit_heat.py where the dataset 
# is available

import numpy as np
from generate_data_heat import generate_heat_X_y
from fit_deeoponet_heat import fit_dde
from pathlib import Path
import shutil

ROOT_DIRECTORY = Path("..")  # Path(".")
RESULTS_DIRECTORY = ROOT_DIRECTORY / "results_heat_p_40"

generar = True
n_sensors_x_list = [101]
SIGMA = 1
n_funciones = [10 ** 2, 5 * 10 ** 2, 10 ** 3, 5 * 10 ** 3, 10 ** 4]
#n_funciones = [10 ** 2, 10 ** 3, 10 ** 4]
#n_funciones = [2]

l = np.array([0.5, 0.25, 0.125, 0.0625])
N = {0.5: 3, 0.25: 5, 0.125: 8, 0.0625: 14}
l_cor = [0.5, 0.25, 0.125, 0.0625]
l_cor = [0.5, 0.0625]
#l_posvel = [(lp, l_vel[i]) for i, lp in enumerate(l_pos) if i == 1]
p = 40  # dimension of the trunk net

for l_c in l_cor:
    results_path = RESULTS_DIRECTORY / f"l_{l_c}"
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    for n_sensors_x in n_sensors_x_list:
        # generating data
        X = dict()
        y = dict()
        seeds = [31415, 314159]
        for i, set in enumerate(["train", "test"]):
            X[set], y[set] = generate_heat_X_y(
                l_c,
                n_sensors_x=n_sensors_x,
                sigma=SIGMA,
                length_KL_expansion=N[l_c],
                n_samples_functions=n_funciones[-1],
                seed=seeds[i],
            )
            print(
                f"""data generation completed l {l}
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
            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)
            predict_plot_path = save_path / "predict_check"
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
                f"""fit and predict completed l {l},
                                    n_sensors_x: {n_sensors_x}, n: {n}"""
            )
