# experiment1.py solves the boundary controllability problem
# for the linear heat equation
# y_t = kappa*y_xx in (0,1) x (0,T)
# y(x,0) = sin(pi*x)
# y(0,t) = 0, y(1,t) = u(t)
# y(x, T) = 0, 0< x <1


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import deepxde as dde
from deepxde.backend import tf



def heat(xx, initial_temp):
    kappa = 0.25 # thermal conductivity
    T= 0.5 # control time
    def pde(x, y): # heat equation
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - kappa * dy_xx
    
    def temp_0(x):
        return np.interp(x[:, 0:1], xx, initial_temp)
    
    def boundary_left(x, on_boundary): # boundary x = 0
        return on_boundary and np.isclose(x[0], 0)    

    def boundary_bottom(x, on_boundary): # boundary t = 0
        return on_boundary and np.isclose(x[1], 0)
    
    def boundary_upper(x, on_boundary): # boundary t = T
        return on_boundary and np.isclose(x[1], T)

    geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, T])
    
    bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_left)
    bc2 = dde.DirichletBC(geom, temp_0, boundary_bottom)
    bc3 = dde.DirichletBC(geom, lambda x: 0, boundary_upper) 
    
    data = dde.data.PDE(
        geom, 
        pde, 
        [bc1, bc2, bc3], 
        num_domain=400, 
        num_boundary=60
        )
    net = dde.maps.FNN([2] + [100] * 5 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    model.compile("adam", lr=0.001)
    model.train(epochs=10000)
    model.compile("L-BFGS-B")
    losshistory, train_state = model.train()
    #dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    # Post-processing: error analysis and figures
    tt = np.linspace(0, T, 101) # time discretization
    xx = np.ones_like(tt)
    X = np.vstack((xx, tt)).T 
    control_predict = np.ravel(model.predict(X)) # predicted control
    #control_predict[0]=0 # at t=0 this value is known
    #print(f"control_predict_shape = {control_predict.shape}")

    return control_predict

   # fig = plt.figure()    
   # plt.plot(tt,control_predict, "b", linewidth=2, label='Predicted control')
   # plt.xlabel('$t$')
   # plt.grid()
   # plt.show()

   # X = np.linspace(0, 1, 100)
   # t = np.linspace(0, T, 100)
   # X_repeated = np.repeat(X,  t.shape[0])  # repeat each value of X the number of elements of t 
   # t_tiled = np.tile(t, X.shape[0]) # repeat the whole vector t, the number of elements of X
   # XX = np.vstack((X_repeated, t_tiled)).T # stack them vertically and take the transpose
    
   # state_predict = model.predict(XX).T # predicted state
   # state_predict_M = state_predict.reshape((100,100)).T # predicted state
   # Xx, Tt = np.meshgrid(
   # np.linspace(0, 1, 100),
   # np.linspace(0, T, 100)
   # )
   # fig = plt.figure() # plot of predicted state
   # ax = plt.axes(projection="3d")
   # surf = ax.plot_surface(Xx,Tt, state_predict_M)
   # ax.set_title('PINN state ')
   # ax.set_xlabel('$x$')
   # ax.set_ylabel('$t$')
   # plt.show()
    
