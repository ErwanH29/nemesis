from amuse.lab import read_set_from_file, units, Particles, constants
from amuse.ext.orbital_elements import orbital_elements

import natsort
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
from scipy import stats


### Set some globals
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"

SAVE_DIR = "tests/cluster_test"
COLOURS = ["black", "dodgerblue"]
LABELS = ["Ph4", "Nemesis"]
LINESTYLE = ["-", "--"]
MARKER_SIZE = [50, 20, 40, 10]
LW = [8,3]
AXLABEL_SIZE = TICK_SIZE = 14

DATA_DIR = "tests/cluster_test/data"



def tickers(ax):
    """
    Function to setup axis
    Args:
        ax (axis):  Axis needing cleaning up
    """
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
    ax.tick_params(
        axis="y", 
        which='both', 
        direction="in", 
        labelsize=TICK_SIZE
    )
    ax.tick_params(
        axis="x", 
        which='both', 
        direction="in", 
        labelsize=TICK_SIZE
    )


def compare_visually(dir_data, nem_data):
    """
    Compare the direct and Nemesis data visually.
    Args:
        dir_data (list):  List of data files assosciated to direct integration runs.
        nem_data (list):  List of data files assosciated to Nemesis runs.
    """
    out_dir = f"{SAVE_DIR}/visual_comparison/"
    os.makedirs(out_dir, exist_ok=True)
    colours_pastel = ["black", "dodgerblue", "red", "blue"]
    for step, (direct, nem_) in enumerate(zip(dir_data, nem_data)):
        dir_particles = read_set_from_file(direct)
        nem_particles = read_set_from_file(nem_)
        for particles in [dir_particles, nem_particles]:
            particles.move_to_center()

        ### Extract stars
        star_nem = nem_particles[nem_particles.mass >= 0.08 | units.MSun]
        star_dir = dir_particles[dir_particles.mass >= 0.08 | units.MSun]

        dr = 0 | units.au
        for s in star_nem:
            target = star_dir[star_dir.original_key == s.original_key]
            dr = max(dr, (s.position - target.position).lengths())
        print(f"    Biggest displacement: drij = {dr.max().in_(units.au)}")

        ### Extract asteroids
        dir_ast = dir_particles[dir_particles.mass == (0. | units.kg)]
        nem_ast = nem_particles[nem_particles.mass == (0. | units.kg)]
        dir_particles -= dir_ast
        nem_particles -= nem_ast

        fig, ax = plt.subplots(figsize=(7, 6))
        for i, major in enumerate([dir_particles, nem_particles]):
            ax.scatter(
                major.x.value_in(units.au), 
                major.y.value_in(units.au), 
                color=COLOURS[i], 
                label=LABELS[i], 
                s=MARKER_SIZE[i],
                zorder=2
            )

        for i, minor in enumerate([dir_ast, nem_ast]):
            ax.scatter(
                minor.x.value_in(units.au), 
                minor.y.value_in(units.au), 
                color=colours_pastel[i+2], 
                s=5, alpha=0.25,
                zorder=3
            )
        
        tickers(ax)
        ax.set_xlabel(r"$x$ [au]", fontsize=AXLABEL_SIZE)
        ax.set_ylabel(r"$y$ [au]", fontsize=AXLABEL_SIZE)
        plt.savefig(f"{out_dir}/{step}.pdf", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()


def process_data(dir_data, nem_data):
    """
    Save direct N-body and Nemesis integration eccentricity and 
    semi-major axis data of asteroids.
    """
    def _save_kepler(binary, sma_df, ecc_df):
        kepler_elements = orbital_elements(binary, G=constants.G)
        sma, ecc = kepler_elements[2], kepler_elements[3]
        if kepler_elements[3] <= 1.:
            sma_df.append(sma.value_in(units.au))
            ecc_df.append(ecc)

    initial_nem = read_set_from_file(nem_data[0])
    host_dic = {}
    for system_id in np.unique(initial_nem.syst_id):
        system = initial_nem[initial_nem.syst_id == system_id]
        if system_id > 0:
            host = system[system.mass.argmax()]
            asteroids = system[system.mass == 0. | units.kg]
            host_dic[host.original_key] = asteroids.original_key
    
    for i, (p, q) in enumerate(zip(dir_data, nem_data)):
        px = read_set_from_file(p)
        qx = read_set_from_file(q)
        if len(px) != len(qx):
            raise AssertionError(
                "!!! Particle count mismatch !!!\n"
                f"    Pair #{i}\n"
                f"    Direct snapshot: {p} ----> {len(px)}"
                f"    Nemesis snapshot: {q} ---> {len(qx)}"
                )
        assert len(px) == len(qx)
    
    dt_dir = read_set_from_file(dir_data[-1])
    dt_nem = read_set_from_file(nem_data[-1])
    ast_dir = dt_dir[dt_dir.mass == 0. | units.kg]

    temp_dir_df = [[ ] for _ in range(2)]  # SMA, Ecc
    temp_nem_df = [[ ] for _ in range(2)]
    for host_key, system_key in host_dic.items():
        host_dir = dt_dir[dt_dir.original_key == host_key]
        host_nem = dt_nem[dt_nem.original_key == host_key]

        for a_key in system_key:
            ### First deal with Direct data
            target = ast_dir[ast_dir.original_key == a_key]
            binary = Particles(particles=[host_dir, target])
            _save_kepler(binary, temp_dir_df[0], temp_dir_df[1])
            
            ### Then deal with Nemesis data
            target = dt_nem[dt_nem.original_key == a_key]
            binary = Particles(particles=[host_nem, target])
            _save_kepler(binary, temp_nem_df[0], temp_nem_df[1])

    return temp_dir_df, temp_nem_df


def plot_cluster_cdf(dir_data, nem_data):
    fin_dir = read_set_from_file(dir_data[-1], "hdf5")
    fin_nem = read_set_from_file(nem_data[-1], "hdf5")
    for particles in [fin_dir, fin_nem]:
        particles.move_to_center()
        
    dir_stars = fin_dir[fin_dir.mass > 0.08 | units.MSun]
    nem_stars = fin_nem[fin_nem.mass > 0.08 | units.MSun]
    dir_ast = fin_dir[fin_dir.mass == 0. | units.kg]
    nem_ast = fin_nem[fin_nem.mass == 0. | units.kg]
    
    velocity_df = [[ ], [ ]]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, data in enumerate([dir_stars, nem_stars, dir_ast, nem_ast]):
        velocities = data.velocity.lengths().value_in(units.kms)
        sorted_vel = np.sort(velocities)
        sample = np.arange(1, len(sorted_vel)+1) / len(sorted_vel)
        ax.plot(
            sorted_vel, sample, 
            color=COLOURS[i%2], 
            ls=LINESTYLE[i//2], 
            lw=LW[i%2]
            )
        if i < 2:  # Store velocity data of massive bodies
            velocity_df[i] = sorted_vel
            ax.scatter(None, None, color=COLOURS[i], label=LABELS[i], s=50)
    tickers(ax)
    ax.set_xscale("log")
    ax.set_xlabel(r"$v$ [km/s]", fontsize=AXLABEL_SIZE)
    ax.set_ylabel(r"$f_{<}$", fontsize=AXLABEL_SIZE)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=AXLABEL_SIZE+5, frameon=False)
    plt.savefig(f"{SAVE_DIR}/cdf_final_vel.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    cramer = stats.cramervonmises_2samp(velocity_df[0], velocity_df[1])
    print(f"    Cramer-von Miss test for stellar velocity distributions: {cramer}")


    dr = [[ ], [ ]]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, data in enumerate([dir_stars, nem_stars, dir_ast, nem_ast]):
        positions = data.position.lengths().value_in(units.au)
        sorted_pos = np.sort(positions)
        sample = np.arange(1, len(sorted_pos)+1) / len(sorted_pos)
        ax.plot(
            sorted_pos, sample, 
            color=COLOURS[i%2], 
            ls=LINESTYLE[i//2], 
            lw=LW[i%2]
            )
        if i < 2:
            ax.scatter(None, None, color=COLOURS[i], label=LABELS[i], s=50)
            dr[i] = sorted_pos
    tickers(ax)
    ax.set_xscale("log")
    ax.set_xlabel(r"$r$ [pc]", fontsize=AXLABEL_SIZE)
    ax.set_ylabel(r"$f_{<}$", fontsize=AXLABEL_SIZE)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=AXLABEL_SIZE+5, frameon=False)
    plt.savefig(f"{SAVE_DIR}/cdf_final_pos.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    cramer = stats.cramervonmises_2samp(dr[0], dr[1])
    print(f"   Cramer-von Miss test for stellar position distributions: {cramer}")


def plot_ast_cdf(dir_sma_ast, dir_ecc_ast, nem_sma_ast, nem_ecc_ast):
    data_dic = {
        "sma": [[dir_sma_ast, nem_sma_ast], r"$a$ [au]", [2, 1e5]],
        "ecc": [[dir_ecc_ast, nem_ecc_ast], r"$e$", [1e-3, 1]]
    }
    for i, (fname, (df, xlabel, xlims)) in enumerate(data_dic.items()):
        fig, ax = plt.subplots(figsize=(7, 6))
        for j, data in enumerate(df):
            sorted_data = np.sort(data)
            sample_data = np.arange(1, len(sorted_data)+1) / len(sorted_data)
            ax.plot(sorted_data, sample_data, lw=LW[j], color=COLOURS[j])
            ax.scatter([], [], color=COLOURS[j], label=LABELS[j], s=50)
        ax.set_xlabel(xlabel, fontsize=AXLABEL_SIZE)
        ax.set_ylabel(r"$f_{<}$", fontsize=AXLABEL_SIZE)
        ax.legend(fontsize=AXLABEL_SIZE+5, frameon=False)
        ax.set_ylim(0, 1.)
        ax.set_xlim(xlims)
        tickers(ax)
        if i==0:
            ax.set_xscale("log")
        plt.savefig(f"{SAVE_DIR}/cdf_{fname}.pdf", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

        dir_sorted = np.asarray(np.sort(df[0]))
        nem_sorted = np.asarray(np.sort(df[1]))

        anderson = stats.anderson_ksamp(
            [dir_sorted, nem_sorted], 
            method=stats.PermutationMethod()
            )
        cramer = stats.cramervonmises_2samp(dir_sorted, nem_sorted)
        ks = stats.ks_2samp(dir_sorted, nem_sorted)
        print(f"Anderson-Darling test for {fname}: {anderson}")
        print(f"Cramer-von Mises test for {fname}: {cramer}")
        print(f"KS test for {fname}: {ks}")

        D_obs = stats.ks_2samp(
            dir_sorted, nem_sorted,
            alternative='two-sided',
            mode='asymp'
            ).statistic
        n1, n2 = len(dir_sorted), len(nem_sorted)

        alpha = 0.05
        c_alpha = np.sqrt(-0.5 * np.log(alpha / 2))
        D_alpha = c_alpha * np.sqrt((n1 + n2) / (n1 * n2))
        print(f"D_obs = {D_obs}, D_alpha (alpha={alpha})= {D_alpha}")


def plot_energy(dir_df, nem_df):
    """Plot the energy evolution of the system"""
    dE_array = [[ ] for _ in range(2)]  # Direct, Nemesis
    
    for i, data_files in enumerate([dir_df, nem_df]):
        for j, snapshot in enumerate(data_files):
            print(f"\rProcessing file {j+1}/{len(data_files)}", end=" ", flush=True)
            p = read_set_from_file(snapshot)
            massives = p[p.mass > 0. | units.kg]
            if i == 0:
                E0 = massives.kinetic_energy() + massives.potential_energy()
            else:
                Et = massives.kinetic_energy() + massives.potential_energy()
                dE = abs((Et - E0)/E0)
                dE_array[i].append(dE)

    smoothing = 5
    value = np.cumsum(dE_array[1], dtype=np.float64)
    value[smoothing:] = value[smoothing:] - value[:-smoothing]
    dE_array[1] = value[smoothing-1:] / smoothing

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, df in enumerate(dE_array):
        time = np.linspace(0, 0.1, len(df))
        ax.plot(time, df, color=COLOURS[i], lw=3)
    ax.scatter(None, None, color=COLOURS[0], label=LABELS[0], s=50)
    ax.scatter(None, None, color=COLOURS[1], label=LABELS[1], s=50)
    ax.set_xlabel(r"$t$ [Myr]", fontsize=AXLABEL_SIZE)
    ax.set_ylabel(r"$\Delta E / E_0$", fontsize=AXLABEL_SIZE)
    tickers(ax)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=AXLABEL_SIZE, frameon=False)
    plt.savefig(f"{SAVE_DIR}/energy.pdf", dpi=300, bbox_inches='tight')
    plt.clf()


dir_data = natsort.natsorted(glob.glob(f'{DATA_DIR}/cluster_run_direct/simulation_snapshot/*.hdf5'))
nem_data = natsort.natsorted(glob.glob(f'{DATA_DIR}/cluster_run_nemesis/simulation_snapshot/*.hdf5'))

Nsnaps = min(len(dir_data), len(nem_data))
dir_data = dir_data[:Nsnaps]
nem_data = nem_data[:Nsnaps]

direct_initial  = read_set_from_file(dir_data[0])
Nemesis_initial = read_set_from_file(nem_data[0])
dr = (direct_initial.position - Nemesis_initial.position).lengths()
dv = (direct_initial.velocity - Nemesis_initial.velocity).lengths()
dm = direct_initial.mass - Nemesis_initial.mass
if dr.max() > (0. | units.m):
    raise AssertionError(
        f"Error: Particle not initialised the same."
        f" Maximum dr = {dr.max().in_(units.m)}"
        )
if dv.max() > (0. | units.kms):
    raise AssertionError(
        f"Error: Particle not initialised the same. "
        f" Maximum dv = {dv.max().in_(units.kms)}"
        )
if dm.max() > (0. | units.MSun):
    raise AssertionError(
        f"Error: Particle not initialised the same. "
        f" Maximum dm = {dm.max().in_(units.MSun)}"
    )
print("...Simulation has same IC confirmed.")

#print(f"...Plotting dE...")
#plot_energy(dir_data, nem_data)

print(f"...Comparing visually...")
compare_visually(dir_data, nem_data)

print("...Cluster Overall CDF Plots...")
plot_cluster_cdf(dir_data, nem_data)

print("...Plotting Asteroid CDFs")
direct_df, nemesis_df = process_data(dir_data, nem_data)
print(f"...Asteroids CDF Plots...")
plot_ast_cdf(
    direct_df[0], 
    direct_df[1], 
    nemesis_df[0], 
    nemesis_df[1]
)