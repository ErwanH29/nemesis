""""
Possible Room for Improvements:
    1. Parallelise algorithm. Dictionary look ups and
       current recycling of old code complicates this.
    2. Remove hand-wavey dependence (Rpar and SPLIT_PARAMS)
       from parent radius. Ideally, this would be some fraction
       of Hill radius with a similar calculation to _get_dr_threshold,
       but preliminary tests show a pure-physics approach is too
       agressive and some hybrid model was needed. For future work.
            - Machine learning adaptable parent radius?

NOTE: Since number of asteroids per system is typically
small, one can use the grid-based method to check for splits.
If the user wishes to simulate a large number of asteroids,
the logic will need to be modified.
"""
from __future__ import annotations


from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.spatial import cKDTree

from amuse.datamodel import Particles
from amuse.units import units

from src.environment_functions import (
    connected_components_kdtree, hill_radius,
    set_parent_radius
)
from src.globals import (
    SPLIT_PARAM,
    GRAV_CONST,
    MIN_EVOL_MASS,
    PARENT_RADIUS_MAX,
)


def _get_orb_properties(
    drij: units.length,
    dvij: units.velocity,
    mu,
    dr: units.length,
    dv2: units.velocity
) -> tuple:
    """
    Extract various orbital properties, assuming Keplerian orbits.
    Args:
        drij (units.length):    Relative position vector between bodies.
        dvij (units.velocity):  Relative velocity vector between bodies.
        mu (array):             Standard gravitational parameter of the system.
        dr (units.length):      Relative distance between bodies.
        dv2 (units.velocity):   Relative velocity squared between bodies.
    Returns:
        ecc (array):   Eccentricity of the orbit.
        rp (array):    Periapsis distance of the orbit.
    """
    eps = 0.5 * dv2 - mu / dr

    h = drij.cross(dvij)
    h_sq = h.lengths_squared()
    ecc_sq = (1 + (2 * eps * h_sq) / (mu**2))
    ecc_sq = np.maximum(ecc_sq, 0)  # Handle numerical issues
    ecc = np.sqrt(ecc_sq)

    sma = -mu / (2 * eps)
    rapo = sma * (1 + ecc)
    rperi = h_sq / (mu * (1 + ecc))

    unbound = eps >= 0.0 | eps.unit
    rapo[unbound] = np.inf | dr.unit

    return rapo, eps, rperi


def _get_dr_threshold(
    new_parents: Particles,
    ext_parents: Particles,
    number_of_neighbours: int
) -> list:
    """
    Compute distance criterias for each new parent. Two criterias are computed:
       1) Hill radius based on current positions and velocities.
       2) Current distance to nearest neighbor in the massive external set.

    Args:
        new_parents (Particles):     New parent particle set.
        ext_parents (Particles):     Massive external particle set.
        number_of_neighbours (int):  Number of neighbors for Hill radius.
    Returns:
        dr_criteria (list):  Minimum between Hill radius and distance
                             to nearest neighbor for each new parent.
    """
    nparents = len(new_parents)
    rows = np.arange(nparents)

    ext_pos = ext_parents.position
    dr_grid = new_parents.position[:, np.newaxis] - ext_pos  # (N_par N_ext, 3)
    dr2_grid = dr_grid.lengths()

    same_key = new_parents.key[:, np.newaxis] == ext_parents.key
    if np.any(same_key):  # Exclude self
        dr2_grid[same_key] = np.inf | dr2_grid.unit

    K = nparents + number_of_neighbours
    neigh_idx = np.argsort(dr2_grid, axis=1)[:, :K]  # (N_par K)
    neighb_dist = dr2_grid[rows[:, np.newaxis], neigh_idx]
    neighb_mass = ext_parents.mass[neigh_idx]

    np_mass = new_parents.mass[:, np.newaxis]
    Rhill_grid = hill_radius(np_mass, neighb_mass, neighb_dist)
    dr_criteria = np.minimum(
        Rhill_grid.min(axis=1),
        neighb_dist.min(axis=1)
        )

    return dr_criteria


def _check_asteroid_splits(
    asteroids: Particles,
    new_parent_set: Particles,
    ext_parents: Particles,
    cluster_mass: units.mass,
    cluster_com: units.length,
    number_of_neighbours: int,
) -> list:
    """
    Flag asteroid splits. This procedure is purely for asteroids
    and mitigates the issue of comets, where they can be flagged
    as ejected in the original method due to wide orbits but remain
    bound to parent. Such bodies add splitting + parent merger times.
    Args:
        asteroids (Particles):       Asteroid particle set.
        new_parent_set (Particles):  New parent particle set.
        ext_parents (Particles):     Parent particle set.
        cluster_mass (units.mass):   Total mass of the cluster.
        cluster_com (units.length):  Center of mass of the cluster.
        number_of_neighbours (int):  Number of neighbors for Hill radius.
    Returns:
        new_parent_map (dict):   Mapping of new parent keys to particles.
        new_parent_key (array):  Array of new parent keys for each asteroid.
        asteroids (Particles):   Original asteroid set for reference.
    """
    if len(new_parent_set) == 0 or len(asteroids) == 0:
        return asteroids

    new_keys = np.array(new_parent_set.key, dtype=np.uint64)
    ext_keys = np.array(ext_parents.key, dtype=np.uint64)
    keep_nn = ~np.isin(ext_keys, new_keys)
    ext_parents_nn = ext_parents[keep_nn]  # Exclude new parents
    if len(ext_parents_nn) == 0:  # New parents make up all massive externals
        return asteroids

    # Setup arrays to get nearest neighbours for asteroids
    length_unit = asteroids.position.unit

    ast_pos = asteroids.position
    ast_vel = asteroids.velocity
    ast_pos_unitless = ast_pos.value_in(length_unit)
    ast_tree = cKDTree(ast_pos_unitless)

    # Compute external gravitational field
    ext_pos = ext_parents_nn.position
    drij_ast_to_ext = (ast_pos[:, np.newaxis] - ext_pos)  # (N_ast, N_ext, 3)
    dr_ast_to_ext = drij_ast_to_ext.lengths()  # (N_ast, N_ext)
    rhat = drij_ast_to_ext / dr_ast_to_ext[:, :, np.newaxis]  # (N_ast, N_ext, 3)

    fg_scalar = GRAV_CONST * ext_parents_nn.mass / dr_ast_to_ext**2
    fg_ext_vec = -rhat * fg_scalar[:, :, np.newaxis]   # (N_ast, N_ext, 3)
    fg_ext_mag = fg_ext_vec.sum(axis=1).lengths()      # (N_ast, )

    # Extract nearest neighbours for each asteroid
    tree_ext = cKDTree(ext_pos.value_in(length_unit))
    dr_ast_to_nn, ast_nn_idx = tree_ext.query(ast_pos_unitless, k=1)
    dr_ast_to_nn = dr_ast_to_nn | length_unit

    # min(Hill radius, NN distance) for new parents
    mu_np = GRAV_CONST * new_parent_set.mass
    newp_pos = new_parent_set.position
    newp_vel = new_parent_set.velocity
    dr_criteria = _get_dr_threshold(
        new_parent_set,
        ext_parents,
        number_of_neighbours
        )

    dr_cluster = (newp_pos - cluster_com).lengths()
    cluster_tide = hill_radius(
        new_parent_set.mass,
        cluster_mass,
        dr_cluster
        )

    n_asts = len(asteroids)
    ast_orb_energy = np.inf * np.ones(n_asts) | (units.ms)**2
    new_parent_key = np.full(n_asts, 0, dtype=np.uint64)
    for ip, new_par in enumerate(new_parent_set):
        dr_crit = min(dr_criteria[ip], cluster_tide[ip])

        # Criteria 1: Asteroid within dr_crit of new parent
        cand1 = ast_tree.query_ball_point(
            newp_pos[ip].value_in(length_unit),
            r=dr_crit.value_in(length_unit)
            )
        if len(cand1) == 0:
            continue

        cand1 = np.array(cand1)
        ast_nn_idx_c1 = ast_nn_idx[cand1]
        drij1_c1 = ast_pos[cand1] - newp_pos[ip]
        dvij1_c1 = ast_vel[cand1] - newp_vel[ip]
        dr1_c1 = drij1_c1.lengths()
        dv1_sq_c1 = dvij1_c1.lengths_squared()

        nn_pos_c1 = ext_parents_nn.position[ast_nn_idx_c1]
        dr_np_to_nn = (nn_pos_c1 - newp_pos[ip]).lengths()

        # Hill radii of host wrt NN and NN wrt host
        m_nn_c1 = ext_parents_nn.mass[ast_nn_idx_c1]
        Rhill_np = hill_radius(
            m1=new_par.mass,
            m2=m_nn_c1,
            dr=dr_np_to_nn
            )
        Rhill_nn = hill_radius(
            m1=m_nn_c1,
            m2=new_par.mass,
            dr=dr_np_to_nn
            )

        rapo, eps_np, _ = _get_orb_properties(
            drij1_c1,
            dvij1_c1,
            mu=mu_np[ip],
            dr=dr1_c1,
            dv2=dv1_sq_c1,
            )

        # Criteria 2: Orbit does not impinge NN Hill radius
        mask_2 = (rapo < Rhill_np) & (rapo < (dr_np_to_nn - Rhill_nn))

        # Criteria 3: New parent dominates local gravitational field
        fg_np = mu_np[ip] / rapo**2
        mask_3 = fg_np > fg_ext_mag[cand1]

        mask = mask_2 & mask_3
        if not np.any(mask):
            continue

        candidates = cand1[mask]
        eps = eps_np[mask]

        # Criteria 4: Most bound energetically to new parent
        bound_mask = eps < ast_orb_energy[candidates]
        idx = candidates[bound_mask]
        ast_orb_energy[idx] = eps[bound_mask]
        new_parent_key[idx] = new_par.key

    del ext_parents
    new_parent_map = {p.key: p for p in new_parent_set}

    return (new_parent_map, new_parent_key, asteroids)


def _process_asteroid_splits(
    nem_class: object,
    results: list,
    new_rogue: Particles
) -> None:
    """
    Process asteroid splits and add them to the appropriate child systems.
    Args:
        nem_class (object):      Nemesis instance.
        results (list):          List of tuples containing new parent maps,
                                 keys, and asteroid sets.
        new_rogue (Particles):   Set to accumulate newly isolated asteroids.
    """
    for new_parent_map, new_parent_key, asteroids in results:
        unique_keys = np.unique(new_parent_key)
        for new_key in unique_keys:
            mask = (new_parent_key == new_key)
            children = asteroids[mask]
            if new_key == 0:  # Unbound asteroids
                new_rogue.add_particles(children)
                continue

            new_parent = new_parent_map[new_key]
            children.position -= new_parent.position
            children.velocity -= new_parent.velocity

            nem_class.resume_workers(nem_class._pid_workers[new_key])
            subsystem = nem_class.children[new_key][1]
            subcode = nem_class.subcodes[new_key]

            child_as_set = children.as_set()
            subsystem.add_particles(child_as_set)
            subcode.particles.add_particles(child_as_set)
            nem_class.hibernate_workers(nem_class._pid_workers[new_key])


def split_subcodes(nem_class, number_of_neighbours) -> None:
    """
    Check for any isolated children
    Args:
        nem_class (object):         Nemesis instance.
        number_of_neighbours (int):  Number of neighbors for Hill radius.
    """
    if nem_class._verbose:
        print("...Checking Splits...")

    Nsplits = 0
    new_rogue = Particles()
    if nem_class._test_particle:
        split_ast_dic = {}
        cluster_mass = nem_class.particles.mass.sum()
        cluster_com = nem_class.particles.center_of_mass()

    for parent_key, (parent, subsys) in list(nem_class.children.items()):
        components = connected_components_kdtree(
            child_set=subsys,
            threshold=SPLIT_PARAM * parent.radius
            )
        if len(components) <= 1:
            continue

        if nem_class._verbose:
            Nsplits += 1

        new_parent_set = Particles()
        rework_code = False
        par_vel = parent.velocity
        par_pos = parent.position

        pid = nem_class._pid_workers.pop(parent_key)
        nem_class.resume_workers(pid)
        nem_class.particles.remove_particle(parent)

        code = nem_class.subcodes.pop(parent_key)
        offset = nem_class._time_offsets.pop(code)
        nem_class._child_channels.pop(parent_key)
        cpu_time = nem_class._cpu_time.pop(parent_key)

        asteroids = Particles()
        for c in components:
            sys = c.as_set()
            sys.position += par_pos
            sys.velocity += par_vel

            ast_mask = sys.mass == 0.0 | units.kg
            has_massive = (len(sys) > 1) and (np.sum(~ast_mask) > 0)
            if has_massive:
                newparent = nem_class.particles.add_children(sys)
                newparent_key = newparent.key
                if nem_class._test_particle:
                    new_parent_set.add_particle(newparent)
                    new_parent_set[-1].original_key = newparent_key

                scale_mass = newparent.mass
                scale_radius = set_parent_radius(scale_mass)
                newparent.radius = scale_radius
                if not rework_code:  # Recycle old code
                    rework_code = True
                    newcode = code
                    newcode.particles.remove_particles(code.particles)
                    newcode.particles.add_particles(sys)
                    nem_class._time_offsets[newcode] = offset
                    worker_pid = pid

                else:
                    newcode, worker_pid = nem_class._sub_worker(
                        children=sys,
                        scale_mass=scale_mass,
                        scale_radius=scale_radius
                        )
                    nem_class._set_worker_affinity(worker_pid)
                    nem_class._time_offsets[newcode] = nem_class.model_time

                nem_class._cpu_time[newparent_key] = cpu_time
                nem_class.subcodes[newparent_key] = newcode

                channel = nem_class._child_channel_maker(
                    parent_key=newparent_key,
                    code_set=newcode.particles,
                    children=sys
                )
                channel["from_children_to_gravity"].copy()  # More precise

                nem_class._pid_workers[newparent_key] = worker_pid
                nem_class.hibernate_workers(worker_pid)

            else:
                if nem_class._test_particle:
                    asteroids.add_particles(sys[ast_mask])
                    new_rogue.add_particles(sys[~ast_mask])
                else:
                    new_rogue.add_particles(sys)

        if nem_class._test_particle:
            if len(new_parent_set) != 0 and len(asteroids) != 0:
                split_ast_dic[new_parent_set] = asteroids

        if not rework_code:  # Only triggered if pure ionisation
            code.cleanup_code()
            code.stop()

    if nem_class._test_particle:
        star_mask = nem_class.particles.mass > MIN_EVOL_MASS
        ext_parents = nem_class.particles[star_mask]

        results = []
        with ThreadPoolExecutor(max_workers=nem_class.num_workers) as executor:
            futures = {
                executor.submit(
                    _check_asteroid_splits,
                    asteroids=asts,
                    new_parent_set=new_par,
                    ext_parents=ext_parents.copy(),  # Otherwise crashes due to concurrent access
                    cluster_mass=cluster_mass,
                    cluster_com=cluster_com,
                    number_of_neighbours=number_of_neighbours
                ): new_par for new_par, asts in split_ast_dic.items()
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    nem_class.cleanup_code()
                    raise RuntimeError(f"Error in asteroid split check: {exc}")

        _process_asteroid_splits(nem_class, results, new_rogue)

    if len(new_rogue) > 0:
        if nem_class._verbose:
            print(f"{len(new_rogue)} new rogue bodies...")

        new_rogue.radius = set_parent_radius(new_rogue.mass)
        mask = new_rogue.radius > PARENT_RADIUS_MAX
        new_rogue[mask].radius = PARENT_RADIUS_MAX
        nem_class.particles.add_particles(new_rogue)

    if nem_class._verbose:
        print(f"{Nsplits} splits processed...")
