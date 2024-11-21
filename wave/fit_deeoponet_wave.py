import numpy as np
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import deepxde as dde
import matplotlib.pyplot as plt

ITERATIONS = 20000


def plot_check_prediction(
    x, t, y0, y1, exact_control, predict_control, predict_plot_file
):
    y_initial = np.concatenate((y0, y1))
    fig, ax = plt.subplots()
    ax.plot(t, exact_control, "-b", linewidth=2, label="Exact control")
    ax.plot(t, predict_control, "-r", linewidth=2, label="Predicted control")
    ax.set_xlabel("$t$")
    ax.legend()
    plt.savefig(predict_plot_file)
    plt.close(fig)
    error = dde.metrics.l2_relative_error(exact_control, predict_control)
    return error


def check_prediction(model, predict_plot_path, n_sensores_x):
    x = np.linspace(0, 1, n_sensores_x)
    t = np.linspace(0, 2, 2 * n_sensores_x - 1)
    y0 = np.sin(np.pi * x)
    y1 = np.sin(np.pi * x)
    y_initial = np.concatenate((y0, y1))
    exact_control = np.piecewise(
        t,
        [t <= 1, t > 1],
        [
            lambda t: 0.5 * np.sin(np.pi * (1 - t))
            - (0.5 / np.pi) * (np.cos(np.pi) - np.cos(np.pi * (1 - t))),
            lambda t: -0.5 * np.sin(np.pi * (t - 1))
            - (0.5 / np.pi) * (np.cos(np.pi) - np.cos(np.pi * (t - 1))),
        ],
    )
    predict_control = np.ravel(
        model.predict((np.tile(y_initial, (len(t), 1)), t[:, None]))
    )
    smooth_error = plot_check_prediction(
        x,
        t,
        y0,
        y1,
        exact_control,
        predict_control,
        predict_plot_path / "predict_smooth.pdf",
    )
    y0 = np.piecewise(
        x,
        [x <= 0.5, x > 0.5],
        [
            lambda x: 4 * x,
            lambda x: 0.0 * x,
        ],
    )
    y1 = np.zeros(len(x))
    y_initial = np.concatenate((y0, y1))
    exact_control = np.piecewise(
        t,
        [t < 0.5, (0.5 <= t) & (t <= 1.5)],
        [
            lambda t: 0.0 * t,
            lambda t: 2.0 * (1 - t),
            lambda t: 0.0 * t,
        ],
    )
    predict_control = np.ravel(
        model.predict((np.tile(y_initial, (len(t), 1)), t[:, None]))
    )
    unsmooth_error = plot_check_prediction(
        x,
        t,
        y0,
        y1,
        exact_control,
        predict_control,
        predict_plot_path / "predict_unsmooth.pdf",
    )
    l2_error_file = predict_plot_path / "l2_error.txt"
    with open(l2_error_file, "w") as l2_f:
        l2_f.write(f"smooth, {smooth_error}\r")
        l2_f.write(f"unsmooth, {unsmooth_error}")


def fit_dde(
    X_train_wave,
    y_train_wave,
    X_test_wave,
    y_test_wave,
    p,
    n_sensores_x,
    save_path,
    predict_plot_path,
    previous_save_path=None,
    previous_best_step=None,
    previous_model=None,
):
    print("save path in fitdde")
    print(save_path.absolute())
    X_train = (X_train_wave[:, :-1], X_train_wave[:, -1:])
    X_test = (X_test_wave[:, :-1], X_test_wave[:, -1:])
    y_train = y_train_wave
    y_test = y_test_wave
    m = (
        X_train_wave.shape[1] - 1
    )  # input dimension of the branch net = 2 * n_sensors_x (postion and velocity)
    dim_t = 1  # input dimension of the trunk
    net = dde.nn.DeepONet(
        [m, 40, p],  # dimensions of the fully connected branch net
        [dim_t, 40, p],  # dimensions of the fully connected trunk net
        "relu",
        "Glorot normal",  # initialization of parameters
    )
    if previous_save_path is None:
        data = dde.data.Triple(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
        model = dde.Model(data, net)
    else:
        model = previous_model
        model.data.train_x = X_train
        model.data.train_y = y_train
        model.data.test_x = X_test
        model.data.test_y = y_test
    checkpointer = dde.callbacks.ModelCheckpoint(
        filepath=save_path / "model" / "model.ckpt",
        verbose=1,
        save_better_only=True,
        period=1000,
    )
    model.compile("adam", lr=0.001)
    if previous_save_path is not None:
        model.restore(
            (
                previous_save_path / "model" / f"model.ckpt-{previous_best_step}.ckpt"
            ).as_posix(),
            verbose=1,
        )
    losshistory, train_state = model.train(
        iterations=ITERATIONS, callbacks=[checkpointer]
    )
    np.save(save_path / "loss_history_wave.npy", losshistory)
    fig, ax = plt.subplots()
    dde.utils.plot_loss_history(losshistory)
    plt.savefig(save_path / "loss_history_wave.pdf")
    plt.close(fig)
    if predict_plot_path is not None:
        check_prediction(model, predict_plot_path, n_sensores_x)
    return losshistory, train_state.best_step, model
