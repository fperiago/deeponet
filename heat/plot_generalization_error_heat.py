import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

ROOT_DIRECTORY = Path("..")  # Path(".")
RESULTS_DIRECTORY = ROOT_DIRECTORY / "results_heat"
FIGURES_DIRECTORY = ROOT_DIRECTORY / "figures_heat"
if not FIGURES_DIRECTORY.exists():
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)

n_funciones = [10 ** 2, 5 * 10 ** 2, 10 ** 3, 5 * 10 ** 3, 10 ** 4]
#n_funciones = [5 * 10 ** 2, 10 ** 3, 5 * 10 ** 3, 10 ** 4]
l_pos = np.array([0.5, 0.25, 0.125, 0.0625])

l_posvel = [(lp) for i, lp in enumerate(l_pos)]

generalization_error = []
for n_sensors_x in [101]:
    for l_p in l_posvel:
        results_path = RESULTS_DIRECTORY / f"n_sensors_x_{n_sensors_x}" / f"l_pos_{l_p}"
        for n in n_funciones:
            save_path = results_path / f"n_{n}"
            if not (save_path / "loss_history_heat.npy").exists():
                break
            loss_history = np.load(
                save_path / "loss_history_heat.npy", allow_pickle=True
            ).item()
            print(f"n: {n}")
            print(f"train loss: {loss_history.loss_train[-1]}")
            print(f"test loss: {loss_history.loss_test[-1]}")
            print(
                f"generalization error {(loss_history.loss_test[-1] - loss_history.loss_train[-1]).item()}"
            )
            error = np.abs(
                (loss_history.loss_test[-1] - loss_history.loss_train[-1]).item()
            )
            generalization_error.append( 
                {"n_sensors_x": n_sensors_x, "l_p": l_p, "n_funciones": n, "error": error}
            )
generalization_error = pd.DataFrame(generalization_error)
print(generalization_error)

i = 1
for (n_sensors_x, l_p), g in generalization_error.groupby(["n_sensors_x", "l_p"]):
    if n_sensors_x == 101:
        fig, ax = plt.subplots()
        ax.plot(
            g["n_funciones"], g["error"], label=f"l_pos:{l_p}"
        )
        ax.legend()
        plt.savefig(FIGURES_DIRECTORY / "n_sensors_x_101" / f"generalization_error_{i}.pdf")
        #i += 1

for (l_p), g in generalization_error.groupby(["l_p", "l_v"]):
    if (l_p) in [(0.125), (0.0625)]:
        fig, ax = plt.subplots()
        for n_sensors_x, gg in g.groupby("n_sensors_x"):
            ax.plot(
                gg["n_funciones"], gg["error"], label=f"n_sensores: {n_sensors_x}, l_pos:{l_p}"
            )
        ax.legend()
        plt.savefig(FIGURES_DIRECTORY / f"generalization_error_n_sensors_x_lp_{l_p}.pdf")
        plt.close()
