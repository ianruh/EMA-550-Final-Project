from astropy.time import Time
import numpy as np
from .propagator import propagate_earth, Ephemeris

moon_ephemeris = Ephemeris(
        np.array([
            -3.770392178192614e+05,
            -9.996027470442034e+03,
             2.673353465391585e+04,
             8.157864969157962e-02,
            -1.037907433689475e+00,
            -6.368865693556963e-02
        ], dtype=np.double)*1000.0,
        Time("2022-04-15").unix)
