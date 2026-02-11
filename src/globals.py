import time
from amuse.ext.galactic_potentials import MWpotentialBovy2015
from amuse.lab import constants, units

##### PHYSICAL PARAMETERS
ASTEROID_RADIUS = 10. | units.km
MIN_EVOL_MASS = 0.08 | units.MSun
MWG = MWpotentialBovy2015()

##### SIMULATION PARAMETERS
CONNECTED_COEFF = 2.
EPS = 1.e-8
PARENT_NWORKER = 1
PARENT_RADIUS_COEFF = 100. | units.au
PARENT_RADIUS_MAX = 300. | units.au

##### SI UNITS
ACC_UNITS = units.m / units.s**2
GRAV_CONST = constants.G
SI_UNITS = (1. | units.kg * units.m**-2.) * constants.G

##### OTHER GLOBALS
START_TIME = time.time()