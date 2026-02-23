""""
STILL TO DO:
Parallelise algorithm. Namely, when checking
friends-of-friends for each parent, then sequentially
processing each parent followed by another parallel
scheme to check for asteroid splits.

NOTE: Since number of asteroids per system is typically
small, one can use the grid-based method to check for splits.
If the user wishes to simulate a large number of asteroids,
the logic will need to be modified.
"""


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
    SPLIT_PARAM, GRAV_CONST, 
    MIN_EVOL_MASS, PARENT_RADIUS_MAX
)


def _get_orb_properties(drij, dvij, mu, dr=None, dv2=None):
    """
    Extract eccentricity and periapsis distance between bodies.
    Args: 
        drij (array):    Relative position vector between bodies.
        dvij (array):    Relative velocity vector between bodies.
        mu (array):      Standard gravitational parameter of the system.
        dr (array):      Relative distance between bodies.
        dv2 (array):     Relative velocity squared between bodies.
        get_rapo (bool): Whether to return apoapsis distance and specific energy.
    Returns:
        ecc (array):   Eccentricity of the orbit.
        rp (array):    Periapsis distance of the orbit.
    """
    eps = 0.5 * dv2 - mu / dr

    h = drij.cross(dvij)
    h_sq = h.lengths_squared()
    ecc_sq = (1 + (2 * eps * h_sq) / (mu**2))
    ecc_sq = np.maximum(ecc_sq, 0)
    ecc = np.sqrt(ecc_sq)
    
    unbound = eps >= 0.0 | eps.unit

    sma = -mu / (2 * eps)
    rapo = sma * (1 + ecc)
    rapo[unbound] = np.inf | dr.unit
    rperi = h_sq / (mu * (1 + ecc))

    return rapo, eps, rperi


def _get_dr_threshold(
    np_set,
    np_pos,
    massive_ext,
    number_of_neighbours
    ) -> list:
    """
    Compute distance criterias between new parents and external parents.
    Two critieras are computed:
       1) First-order estimate of distance at end of next bridge step
       2) Hill radius based on current positions and velocities
    The minimum is used as splitting asteroid cutoff.
    Args:
        np_set (Particles):          New parent particle set.
        np_pos (array):              New parent positions in SI units.
        massive_ext (Particles):     Massive external particle set.
        number_of_neighbours (int):  Number of neighbors for Hill radius.
    Returns:
        dr_criteria (list):  Minimum between Hill radius and distance
                             to nearest neighbor for each new parent.
    """
    np_mass = np_set.mass[:, np.newaxis]
    nparents = len(np_set)
    K = nparents + number_of_neighbours
    rows = np.arange(nparents)

    ext_pos = massive_ext.position
    dr_grid = (np_pos[:, np.newaxis] - ext_pos)
    dr2_grid = dr_grid.lengths()

    same_key = (np_set.key[:, np.newaxis] == massive_ext.key)
    if np.any(same_key):  # Exclude self
        dr2_grid[same_key] = np.inf | dr2_grid.unit

    neigh_idx = np.argsort(dr2_grid, axis=1)[:, :K]
    neighb_mass = massive_ext.mass[neigh_idx]
    neighb_dist = dr2_grid[rows[:, np.newaxis], neigh_idx]

    Rhill_grid = hill_radius(np_mass, neighb_mass, neighb_dist)

    dr_criteria = np.minimum(
        Rhill_grid.min(axis=1), 
        neighb_dist.min(axis=1)
        )

    return dr_criteria


def _check_asteroid_splits(
    nem_class,
    asteroids,
    new_parent_set,
    massive_ext,
    number_of_neighbours
    ) -> Particles:
    """
    Function to check for asteroid splits from parents. This is
    added to mitigate the issue of comets, where they can be flagged
    as ejected due to high eccentricity yet remain bound to parent
    system. Such bodies add splitting + parent merger times.
    Args:
        nem_class (Nemesis):         Nemesis instance.
        asteroids (Particles):       Asteroid particle set.
        new_parent_set (Particles):  New parent particle set.
        massive_ext (Particles):     Massive external particle set.
        number_of_neighbours (int):  Number of neighbors for Hill radius.
        bridge_time (units.time):    Nemesis bridge time step.
    """
    if len(new_parent_set) == 0 or len(asteroids) == 0:
        return asteroids

    length_unit = asteroids.position.unit

    n_asts = len(asteroids)
    new_keys = np.array(new_parent_set.key, dtype=np.uint64)
    ext_keys = np.array(massive_ext.key, dtype=np.uint64)

    keep_nn = ~np.isin(ext_keys, new_keys)
    massive_ext_nn = massive_ext[keep_nn]  # Excludes new parents
    if len(massive_ext_nn) == 0:  # New parents make up all massive externals
        return asteroids

    # Setup arrays to get nearest neighbours for asteroids
    ast_r = asteroids.position
    ast_v = asteroids.velocity
    ast_xyz = ast_r.value_in(length_unit)
    tree_ast = cKDTree(ast_xyz)

    ext_r = massive_ext_nn.position
    ext_xyz = ext_r.value_in(length_unit)
    mu_ext = GRAV_CONST * massive_ext_nn.mass

    # Compute external gravitational field
    drij_ast_to_ext = (ast_r[:, np.newaxis] - ext_r)         # (N_ast, N_ext, 3)
    r_ast_to_ext = drij_ast_to_ext.lengths()                 # (N_ast, N_ext)
    rhat = drij_ast_to_ext / r_ast_to_ext[:, :, np.newaxis]  # (N_ast, N_ext, 3)

    fg_scalar = mu_ext / r_ast_to_ext**2
    fg_ext_vec = -rhat * fg_scalar[:, :, np.newaxis]         # (N_ast, N_ext, 3)
    fg_ext_tot = fg_ext_vec.sum(axis=1)                      # (N_ast, 3)
    fg_ext_mag = fg_ext_tot.lengths()                        # (N_ast, )
    
    # Extract nearest neighbours for each asteroid
    tree_ext = cKDTree(ext_xyz)
    dr_ast_to_nn, ast_nn_idx = tree_ext.query(ast_xyz, k=1)
    dr_ast_to_nn = dr_ast_to_nn | length_unit
    eff_pot = -GRAV_CONST * massive_ext_nn.mass[ast_nn_idx] / dr_ast_to_nn

    # min(Hill radius, NN distance) for new parents
    newp_pos = new_parent_set.position
    newp_vel = new_parent_set.velocity

    dr_criteria = _get_dr_threshold(
        new_parent_set,
        newp_pos,
        massive_ext,
        number_of_neighbours
        )
    
    cluster_mass = massive_ext_nn.mass.sum()
    cluster_com = massive_ext_nn.center_of_mass()
    cluster_tide = hill_radius(
        new_parent_set.mass,
        cluster_mass,
        (newp_pos - cluster_com).lengths()
    )

    ast_orb_energy = np.inf * np.ones(n_asts) | (units.ms)**2
    new_parent_key = np.full(n_asts, 0, dtype=np.uint64)
    for ip, new_par in enumerate(new_parent_set):
        dr_crit = dr_criteria[ip]
        dr_crit = min(dr_crit, cluster_tide[ip])

        # (1) Within dr_crit of new parent
        cand1 = tree_ast.query_ball_point(
            newp_pos[ip].value_in(length_unit), 
            r=dr_crit.value_in(length_unit)
            )
        if len(cand1) == 0:
            continue
        
        cand1 = np.array(cand1)
        drij1_c1 = ast_r[cand1] - newp_pos[ip]
        dvij1_c1 = ast_v[cand1] - newp_vel[ip]
        dr1_c1 = drij1_c1.lengths()
        dv1_sq_c1 = dvij1_c1.lengths_squared()
        
        nn_pos_c1 = massive_ext_nn.position[ast_nn_idx[cand1]]
        dr_np_to_nn = (nn_pos_c1 - newp_pos[ip]).lengths()
        
        # Hill radii of host wrt NN and NN wrt host
        m_nn_c1 = massive_ext_nn.mass[ast_nn_idx[cand1]]
        Rhill_np = hill_radius(
            new_par.mass, 
            m_nn_c1, 
            dr_np_to_nn
            )
        Rhill_nn = hill_radius(
            m_nn_c1, 
            new_par.mass, 
            dr_np_to_nn
            )

        mu_np = GRAV_CONST * new_par.mass
        rapo, eps_np, _ = _get_orb_properties(
            drij1_c1,
            dvij1_c1,
            mu_np,
            dr=dr1_c1,
            dv2=dv1_sq_c1,
        )
        
        rapo *= SPLIT_PARAM**2
        # (2) Orbit does not impinge any of the three NN Hill radius
        mask_2 = (rapo < Rhill_np) & (rapo < (dr_np_to_nn - Rhill_nn))

        # (3) Bound to new parent
        
        fg_np = mu_np / rapo**2
        mask_4 = fg_np > fg_ext_mag[cand1]

        mask = mask_2 & mask_4
        
        n4 = np.sum(mask_2) # KEEP
        n6 = np.sum(mask_4) # KEEP
        print(f"{n4} {n6}")
        if not np.any(mask):
            continue

        candidates = cand1[mask]
        eps = eps_np[mask]

        # (6) Most bound to new parent
        bound_mask = eps < ast_orb_energy[candidates]
        idx = candidates[bound_mask]
        ast_orb_energy[idx] = eps[bound_mask]
        new_parent_key[idx] = new_par.key

    unique_keys = np.unique(new_parent_key)
    new_parent_map = {p.key: p for p in new_parent_set}

    if nem_class._verbose:
        Nbound = 0

    isolated = Particles()
    for new_key in unique_keys:
        mask = new_parent_key == new_key
        children = asteroids[mask]
        if new_key == 0:  # Unbound asteroids
            isolated = children
            continue

        if nem_class._verbose:
            Nbound += len(children)

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

    if nem_class._verbose:
        print(f"...{Nbound} bound to new parents, {n_asts-Nbound} unbound...")
    
    del massive_ext

    return isolated


def split_subcodes(nem_class, number_of_neighbours) -> None:
    """
    Check for any isolated children
    Args:
        nem_class (Nemesis):         Nemesis instance.
        number_of_neighbours (int):  Number of neighbors for Hill radius.
    """
    if nem_class._verbose:
        print("...Checking Splits...")

    Nsplits = 0
    new_isolated = Particles()
    if nem_class._test_particle:
        split_ast_dic = {}

    for parent_key, (parent, subsys) in list(nem_class.children.items()):
        par_rad = parent.radius
        
        components = connected_components_kdtree(
            child_set=subsys,
            threshold=SPLIT_PARAM * par_rad
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

            has_massive = (len(sys) > 1) and np.any(sys.mass.value_in(units.kg) > 0.0)
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
                    ast_mask = sys.mass == 0.0 | units.kg
                    asteroids.add_particles(sys[ast_mask])
                    new_isolated.add_particles(sys[~ast_mask])

                else:
                    new_isolated.add_particles(sys)

        if nem_class._test_particle:  # Could be parallelised
            if len(new_parent_set) == 0 or len(asteroids) == 0:
                continue

            split_ast_dic[new_parent_set] = asteroids

        if not rework_code:  # Only triggered if pure ionisation
            code.cleanup_code()
            code.stop()

    if nem_class._test_particle:
        star_mask = nem_class.particles.mass > MIN_EVOL_MASS
        massive_ext = nem_class.particles[star_mask].copy()
        
        cpu_nem = nem_class.avail_cpus
        no_syst = len(split_ast_dic)
        nworkers = max(1, min(cpu_nem//20, no_syst))
        with ThreadPoolExecutor(max_workers=nworkers) as executor:
            futures = {
                executor.submit(
                    _check_asteroid_splits,
                    nem_class=nem_class,
                    asteroids=asts,
                    new_parent_set=new_par,
                    massive_ext=massive_ext,
                    number_of_neighbours=number_of_neighbours
                ): new_par for new_par, asts in split_ast_dic.items()
            }
            for future in as_completed(futures):
                isolated = future.result()
                new_isolated.add_particles(isolated)

    if len(new_isolated) > 0:
        if nem_class._verbose:
            print(f"{len(new_isolated)} new rogue bodies...")

        new_isolated.radius = set_parent_radius(new_isolated.mass)
        mask = new_isolated.radius > PARENT_RADIUS_MAX
        new_isolated[mask].radius = PARENT_RADIUS_MAX
        nem_class.particles.add_particles(new_isolated)

    if nem_class._verbose:
        print(f"{Nsplits} splits processed...")
