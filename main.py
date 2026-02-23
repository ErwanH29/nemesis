"""
This is the main script to run Nemesis simulations.
Possible Room for Improvements:
    1. Implement cleaner way to initialise child systems.
    2. Upon resumption, code assumes single star population. That is,
       stellar ages correspond to the time of the simulation at last stop.
       To account for multi-stellar population, may require several stellar
       workers.

For any help or questions, please contact:
    - Erwan Hochart...........(hochart@strw.leidenuniv.nl)
    - Simon Portegies Zwart...(spz@strw.leidenuniv.nl)
"""
from __future__ import annotations

import glob
from natsort import natsorted
import numpy as np
import os
import time

from amuse.io import read_set_from_file, write_set_to_file
from amuse.lab import Particles
from amuse.units import units, nbody_system
from amuse.units.optparse import OptionParser

from src.environment_functions import galactic_frame, set_parent_radius
from src.globals import EPS, MIN_EVOL_MASS, START_TIME
from src.hierarchical_particles import HierarchicalParticles
from src.nemesis import Nemesis


def create_output_directories(dir_path: str) -> None:
    """
    Creates directories for output.
    Args:
        dir_path (str):  Directory path.
    """
    subdirs = [
        "collision_snapshot",
        "sim_stats",
        "simulation_snapshot"
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(dir_path, subdir), exist_ok=True)


def load_particle_set(ic_file: str) -> Particles:
    """
    Load particle set from file.
    Args:
        ic_file (str):  Path to initial conditions.
    Returns:
        Particles:  Initial particle set.
    """
    particle_set = read_set_from_file(ic_file)
    if len(particle_set) == 0:
        raise ValueError(f"Error: Particle set {ic_file} is empty.")

    particle_set.coll_events = 0
    particle_set.move_to_center()
    particle_set.original_key = particle_set.key

    syst_ids = particle_set.syst_id

    # Ensure no orphaned syst_id > 0
    ids, counts = np.unique(syst_ids[syst_ids > 0], return_counts=True)
    orphan_ids = ids[counts == 1]
    if orphan_ids.size:
        mask = np.isin(syst_ids, orphan_ids)
        assert mask.sum() == orphan_ids.size, "Error: Not all systems isolated."
        particle_set[mask].syst_id = -1

    return particle_set


def configure_galactic_frame(particle_set: Particles) -> Particles:
    """
    Shift particles to galactic reference frame.
    Args:
        particle_set (Particles):  The particle set.
    Returns:
        Particles: Particle set with galactocentric coordinates.
    """
    return galactic_frame(
        particle_set,
        dpos=[-8.4, 0.0, 0.017] | units.kpc,
        dvel=[11.352, 12.25, 7.41] | units.kms
        )


def identify_parents(particle_set: Particles) -> Particles:
    """
    Identify parents in particle set. These are either:
        - Isolated particles (syst_id < 0).
        - Hosts of subsystem (max mass in system).

    Isolated particles are added directly into Nemesis. The
    most massive particle part of a child system acts as a
    proxy to set the length and mass scale of simulation.
    Args:
        particle_set (Particles):  The particle set.
    Returns:
        Particles:  Parents (Lonely + Host) in particle set.
    """
    system_id_arr = particle_set.syst_id
    parents = particle_set[system_id_arr <= 0]
    system_ids = np.unique(system_id_arr[system_id_arr > 0])
    for system_id in system_ids:
        system = particle_set[np.flatnonzero(system_id_arr == system_id)]
        parents += system[np.argmax(system.mass)]

    return parents


def setup_simulation(dir_path: str, particle_set: Particles) -> tuple:
    """
    Setup simulation directories and load particle set.
    Args:
        dir_path (str):  Directory path for outputs.
        particle_set (Particles):  The particle set.
    Returns:
        tuple: (dir_path, snapshot_path, particles)
    """
    snapshot_path = os.path.join(dir_path, "simulation_snapshot")
    particle_set = load_particle_set(particle_set)
    return snapshot_path, particle_set


def run_simulation(
    IC_file: str,
    run_idx: int,
    tend: units.time,
    dtbridge: units.time,
    dt_diag: units.time,
    test_particle: bool,
    n_worker_parent: int,
    code_dt: float,
    dE_track: bool,
    gal_field: bool,
    star_evol: bool,
    verbose: bool
) -> None:
    """
    Run simulation and output data.
    Args:
        IC_file (str):          Path to initial conditions.
        run_idx (int):          Index of specific run.
        tend (units.time):      Simulation end time.
        dtbridge (units.time):  Bridge timestep.
        dt_diag (units.time):   Diagnostic time step.
        test_particle (bool):   Flag to run test particle simulation.
        n_worker_parent (int):  Number of workers for parent code.
        code_dt (float):        Gravitational integrator internal timestep.
        dE_track (boolean):     Flag turning on energy error tracker.
        gal_field (boolean):    Flag turning on galactic field or not.
        star_evol (boolean):    Flag turning on stellar evolution or not.
        verbose (boolean):      Flag turning on print statements or not.
    """
    sim_dir = IC_file.split("ICs/")[0]
    if "tests" in IC_file:
        if "ZKL" in IC_file:
            from src.globals import PARENT_RADIUS_COEFF
            rpar_in_au = int(PARENT_RADIUS_COEFF.value_in(units.au))
            dir_path = os.path.join(sim_dir, f"ZKL_Rpar{rpar_in_au}au")
        else:
            dir_path = os.path.join(sim_dir, "cluster_run_nemesis")

    else:
        dir_path = os.path.join(sim_dir, f"Nrun{run_idx}")

    init_params = os.path.join(
        dir_path, 'sim_stats', f'sim_params_{run_idx}.txt'
        )
    coll_dir = os.path.join(dir_path, "collision_snapshot")

    print(f"...Starting Nemesis simulation Run {run_idx}...")
    print(f"Job ID: {os.getpid()}")
    if os.path.exists(init_params):
        if verbose:
            print("...Loading from previous simulation...")

        with open(init_params, 'r') as f:
            iparams = f.readlines()
            diag_dt = iparams[3].split(":")[1].split("yr")[0]
            brdg_dt = iparams[4].split(":")[1].split("yr")[0]
            end_time = iparams[5].split(":")[1].split("Myr")[0]

            diag_dt = float(diag_dt) | units.yr
            brdg_dt = float(brdg_dt) | units.yr
            end_time = float(end_time) | units.Myr

        if end_time != tend:
            print("WARNING: End time is not the same as before.")
        if brdg_dt != dtbridge:
            print("WARNING: Bridge timestep is not the same as before.")
        if diag_dt != dt_diag:
            print("WARNING: Diagnostic timestep is not the same as before.")

        snapshot_path = os.path.join(dir_path, "simulation_snapshot")
        previous_snaps = natsorted(glob.glob(os.path.join(snapshot_path, "*")))
        snapshot_no = len(previous_snaps)
        particle_set = read_set_from_file(previous_snaps[-1])

        time_offset = (snapshot_no - 1) * diag_dt
        tend = tend - time_offset
        current_mergers = particle_set.coll_events.sum()
        if verbose:
            print(f"{tend.in_(units.Myr)} remaining in simulation")
            print(f"# Mergers: {current_mergers}")
            print(f"# Snaps: {snapshot_no}")

        # These parameters ensure SeBa doesn't reset stellar age.
        stars = particle_set[particle_set.mass > MIN_EVOL_MASS]
        stars.relative_mass = stars.mass
        stars.age = time_offset
        stars.relative_age = stars.age

    else:
        if verbose:
            print("...Starting new simulation...")

        time_offset = 0. | units.yr
        current_mergers = 0
        snapshot_no = 0
        create_output_directories(dir_path)
        snapshot_path, particle_set = setup_simulation(dir_path, IC_file)
        if (gal_field):
            particle_set = configure_galactic_frame(particle_set)

    snap_path = os.path.join(snapshot_path, "snap_{}.hdf5")
    major_bodies = identify_parents(particle_set)

    Nmajor = len(major_bodies)
    if Nmajor > 2:
        Rvir = major_bodies.virial_radius()
    elif "ZKL" in IC_file:
        dr = (particle_set[1].position - particle_set[0].position).lengths()
        Rvir = dr.max()
    else:
        raise ValueError(
            f"{Nmajor} parents detected."
            "Please define a scale length."
            )

    # Setting up system
    conv_par = nbody_system.nbody_to_si(np.sum(major_bodies.mass), Rvir)
    isolated_systems = major_bodies[major_bodies.syst_id <= 0]
    bounded_systems = major_bodies - isolated_systems

    parents = HierarchicalParticles(isolated_systems)
    nemesis = Nemesis(
        dtbridge=dtbridge,
        code_dt=code_dt,
        test_particle=test_particle,
        n_worker_parent=n_worker_parent,
        par_conv=conv_par,
        coll_dir=coll_dir,
        nmerge=current_mergers,
        resume_time=time_offset,
        dE_track=dE_track,
        star_evol=star_evol,
        gal_field=gal_field,
        verbose=verbose
        )
    for nsyst, id_ in enumerate(np.unique(bounded_systems.syst_id)):
        print(f"\rAdding subsystem with syst_id = {id_}", end="", flush=True)
        subsystem = particle_set[particle_set.syst_id == id_]
        newparent = nemesis.particles.add_children(subsystem)
        newparent.radius = set_parent_radius(newparent.mass)

    nemesis.particles.add_particles(parents)
    nemesis.commit_particles()
    if (nemesis.dE_track):
        energy_arr = []
        E0 = nemesis.calculate_total_energy()

    if snapshot_no == 0:
        allparts = nemesis.particles.all()
        write_set_to_file(
            allparts.savepoint(0 | units.Myr),
            snap_path.format(0), 'amuse',
            close_file=True,
            overwrite_file=True
        )
        allparts.remove_particles(allparts)  # Clean memory

    with open(init_params, 'w') as f:
        f.write("Simulation Parameters:\n")
        f.write(f"  Number of particles: {len(particle_set)}\n")
        f.write(f"  Number of initial children: {nsyst+1}\n")
        f.write(f"  Diagnostic timestep: {dt_diag.in_(units.yr)}\n")
        f.write(f"  Bridge timestep: {dtbridge.in_(units.yr)}\n")
        f.write(f"  End time: {tend.in_(units.Myr)}\n")
        f.write(f"  Galactic field: {gal_field}")

    t = 0. | units.yr
    t_diag = dt_diag
    prev_step = nemesis.dt_step
    snap_time = time.time()
    while t < tend:
        if verbose:
            t0 = time.time()

        t += dtbridge
        while nemesis.model_time < t*(1. - EPS):
            nemesis.evolve_model(t)

        if (nemesis.model_time >= t_diag) and (nemesis.dt_step != prev_step):
            if verbose:
                print(f"Saving snapshot {snapshot_no}: t={t.in_(units.yr)}")
                print(f"Time since last snapshot: {time.time() - snap_time}")
                snap_time = time.time()

            snapshot_no += 1
            fname = snap_path.format(snapshot_no)
            allparts = nemesis.particles.all()
            write_set_to_file(
                allparts.savepoint(nemesis.model_time),
                fname, 'amuse',
                close_file=True,
                overwrite_file=True
            )
            allparts.remove_particles(allparts)  # Clean memory
            t_diag += dt_diag

        if (dE_track) and (prev_step != nemesis.dt_step):
            E1 = nemesis.calculate_total_energy() + nemesis.corr_energy
            energy_arr.append(abs((E1-E0)/E0))

            prev_step = nemesis.dt_step
            if verbose:
                print(f"t = {t.in_(units.Myr)}, dE = {abs((E1-E0)/E0)}")

        prev_step = nemesis.dt_step
        if verbose:
            t1 = time.time()
            print(f"Step took {t1-t0} seconds")

    allparts = nemesis.particles.all()
    write_set_to_file(
        allparts.savepoint(nemesis.model_time),
        snap_path.format(snapshot_no+1), 'amuse',
        close_file=True,
        overwrite_file=True
    )

    # Store simulation statistics
    print("...Simulation Ended...")
    sim_time = (time.time() - START_TIME) / 60.
    fname = os.path.join(dir_path, 'sim_stats', f'sim_stats_{run_idx}.txt')
    with open(fname, 'w') as f:
        f.write(f"Total CPU Time: {sim_time} minutes")
        f.write(f"\nEnd Time: {t.in_(units.Myr)}")
        f.write(f"\nTime step: {dtbridge.in_(units.Myr)}")

    nemesis.cleanup_code()
    if (dE_track):
        with open(os.path.join(dir_path, "energy_error.csv"), 'w') as f:
            f.write(f"Energy error: {energy_arr}")


def new_option_parser():
    result = OptionParser()
    result.add_option("--par_nworker",
                      dest="par_nworker",
                      type="int",
                      default=1,
                      help="Number of workers for parent code")
    result.add_option("--tend",
                      dest="tend",
                      type="float",
                      unit=units.Myr,
                      default=10 | units.Myr,
                      help="End time of simulation")
    result.add_option("--tbridge",
                      dest="tbridge",
                      type="float",
                      unit=units.yr,
                      default=500. | units.yr,
                      help="Bridge timestep")
    result.add_option("--test_particles",
                      dest="test_idx",
                      type="int",
                      default=0,
                      help="Index of specific run")
    result.add_option("--code_dt",
                      dest="code_dt",
                      type="float",
                      default=0.03,
                      help="Gravitational integrator internal timestep")
    result.add_option("--dt_diag",
                      dest="dt_diag",
                      type="int",
                      unit=units.kyr,
                      default=10 | units.kyr,
                      help="Diagnostic time step")
    result.add_option("--gal_field",
                      dest="gal_field",
                      type="int",
                      default=1,
                      help="Flag to turn on galactic field")
    result.add_option("--dE_track",
                      dest="dE_track",
                      type="int",
                      default=0,
                      help="Flag to turn on energy error tracker")
    result.add_option("--star_evol",
                      dest="star_evol",
                      type="int",
                      default=1,
                      help="Flag to turn on stellar evolution")
    result.add_option("--verbose",
                      dest="verbose",
                      type="int",
                      default=1,
                      help="Flag to turn on print statements")
    result.add_option("--run_idx",
                      dest="run_idx",
                      type="int",
                      default=0,
                      help="Index of specific run")

    return result


if __name__ == "__main__":
    o, args = new_option_parser().parse_args()

    path = os.getcwd()
    IC_files = natsorted(glob.glob(f"{path}/examples/basic_cluster/ICs/*"))
    try:
        IC_file = IC_files[o.run_idx]
    except IndexError:
        raise IndexError(
            f"Error: Run index {o.run_idx} out of range. \n"
            f"Available particle sets: {IC_files}."
            )

    run_simulation(
        IC_file=IC_file,
        run_idx=o.run_idx,
        tend=o.tend,
        dtbridge=o.tbridge,
        test_particle=o.test_idx,
        n_worker_parent=o.par_nworker,
        code_dt=o.code_dt,
        dt_diag=o.dt_diag,
        gal_field=o.gal_field,
        dE_track=o.dE_track,
        star_evol=o.star_evol,
        verbose=o.verbose
    )
