import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

ROOT_DIRECTORY =  Path(".")
RESULTS_DIRECTORY = ROOT_DIRECTORY / "results/wave"
FIGURES_DIRECTORY = ROOT_DIRECTORY / "figures/wave"
if not FIGURES_DIRECTORY.exists():
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)

n_funciones = [10 ** 2, 5 * 10 ** 2, 10 ** 3, 5 * 10 ** 3, 10 ** 4]
n_funciones = [5 * 10 ** 2, 10 ** 3, 5 * 10 ** 3, 10 ** 4]
l_pos = np.array([0.5, 0.25, 0.125, 0.0625])
l_vel = l_pos / 2
l_posvel = [(lp, l_vel[i]) for i, lp in enumerate(l_pos)]
#print(f"l_pos_vel= {l_posvel}")

generalization_error = []
for n_sensors_x in [101, 202]:
    for l_p, l_v in l_posvel:
        #print(f"l_p= {l_p}")
        #print(f"l_v= {l_v}")
        #results_path = RESULTS_DIRECTORY / f"n_sensors_x_{n_sensors_x}" / f"l_pos_{l_p}_l_vel_{l_v}"
        results_path = RESULTS_DIRECTORY / f"l_pos_{l_p}_l_vel_{l_v}" / f"n_sensors_x_{n_sensors_x}" 
        for n in n_funciones:
            save_path = results_path / f"n_{n}"
            #if not (save_path / "loss_history_wave.npy").exists():
            #    break           
            #loss_history = np.load(
            #'../results/wave/f"l_pos_{lp}_l_vel_{l_v}"/f"n_sensors_x_{n_senssors_x}"/f"n_{100}"/loss_history_wave.npy', allow_pickle=True
            #).item()
            
            print((save_path / "loss_history_wave.npy").absolute())
            loss_history = np.load(
                save_path / "loss_history_wave.npy", allow_pickle=True
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
                {"n_sensors_x": n_sensors_x, "l_p": l_p, "l_v": l_v, "n_funciones": n, "error": error}
            )
generalization_error = pd.DataFrame(generalization_error)
print(f"generalization error = {generalization_error}")

i = 1
for (n_sensors_x, l_p, l_v), g in generalization_error.groupby(["n_sensors_x", "l_p", "l_v"]):
    if n_sensors_x == 101:
        fig, ax = plt.subplots()
        ax.plot(
            g["n_funciones"], g["error"], label=f"l_pos:{l_p}, l_vel:{l_v}"
        )
        ax.legend()
        figures_sensors = FIGURES_DIRECTORY / "n_sensors_x_101"
        if not figures_sensors.exists():
            figures_sensors.mkdir(parents=True, exist_ok=True)
        plt.savefig( figures_sensors / f"generalization_error_{i}.pdf")
        i += 1

for (l_p, l_v), g in generalization_error.groupby(["l_p", "l_v"]):
    if (l_p, l_v) in [(0.125, 0.0625), (0.0625, 0.03125)]:
        fig, ax = plt.subplots()
        for n_sensors_x, gg in g.groupby("n_sensors_x"):
            ax.plot(
                gg["n_funciones"], gg["error"], label=f"n_sensores: {n_sensors_x}, l_pos:{l_p}, l_vel:{l_v}"
            )
        ax.legend()
        plt.savefig(FIGURES_DIRECTORY / f"generalization_error_n_sensors_x_lp_{l_p}.pdf")
        plt.close()
