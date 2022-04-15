from scipy.integrate import RK45
import numpy as np
from .constants import *
import matplotlib.pyplot as plt

class Ephemeris:

    def __init__(self, pv, t):
        self.pv0 = pv
        self.t0 = t

    def get_traj(self, *, t0: float, t1: float):
        """
        tf is a unix time stamp
        """
        return propagate_earth(pv=self.get_pv(t=t0), t0=t0, t1=t1)


    def get_pv(self, *, t: float):
        """
        tf is a unix time stamp
        """
        times, states = propagate_earth(pv=self.pv0, t0=self.t0, t1=t)
        return states[-1,:]

    def plot(self, *, ax, t0, t1, label, pathColor="blue", pointColor="grey"):
        times, states = self.get_traj(t0=t0, t1=t1)
        ax.plot3D(states[:,0], states[:,1], states[:,2], label=f"{label} Trajectory", color=pathColor)
        
        ax.scatter3D(states[-1,0], states[-1,1], states[-1,2], label=label,color=pointColor);

def plot_earth(*, ax, label):
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    
    x = r_earth * np.outer(np.sin(u), np.sin(v))
    y = r_earth * np.outer(np.sin(u), np.cos(v))
    z = r_earth * np.outer(np.cos(u), np.ones_like(v))

    ax.plot_surface(x, y, z, color="blue", label=label)

def earth_dynamics(time, state):
    """
    The time is in seconds.
    
    The state is [p | v]
    """
    r_vec = state[:3]
    vel = state[3:]
    
    r = np.linalg.norm(r_vec)
    
    a = -1 * mu_earth / r**3 * r_vec

    return np.hstack((vel, a))

def moon_dynamics(time, state):
    """
    The time is in seconds.
    
    The state is [p | v]
    """
    pos = state[:3]
    vel = state[3:]
    
    r_mag = np.linalg.norm(pos, ord=2)
    
    a = -1 * G_const * mass_moon / r_mag**2 * (pos/r_mag)

    return np.hstack((vel, a))

def propagate_earth(*, pv: np.ndarray, t0: float, t1: float):
    rk45 = RK45(earth_dynamics,
            t0,
            pv,
            t1,
            atol=1e-6,
            rtol=1e-12)

    times = [t0]
    states = [pv]
    while(rk45.status == 'running'):
        rk45.step()
        times.append(rk45.t)
        states.append(rk45.y)
    
    times = np.array(times, dtype=np.double)
    states = np.array(states, dtype=np.double)

    return times, states

def propagate_moon(*, pv: np.ndarray, t0: float, dt: float):
    rk45 = RK45(moon_dynamics,
            t0,
            pv,
            t0 + dt,
            atol=1e-6,
            rtol=1e-12)

    times = []
    states = []
    times.append(rk45.t)
    states.append(rk45.y)
    while(rk45.status == 'running'):
        rk45.step()
        times.append(rk45.t)
        states.append(rk45.y)
    
    times = np.array(times, dtype=np.double)
    states = np.array(states, dtype=np.double)

    return times, states
