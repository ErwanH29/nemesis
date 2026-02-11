""""
STILL TO DO:
The current implementation of the asteroid split check is not 
parallelised  and can be time-consuming for large numbers of 
asteroids and/or external parents and/or children system.
"""



import numpy as np

from amuse.datamodel import Particles
from amuse.units import units, constants

from src.environment_functions import (
    connected_components_kdtree, hill_radius, 
    set_parent_radius, specific_orbital_energy
)
from src.globals import CONNECTED_COEFF, GRAV_CONST, PARENT_RADIUS_MAX
    

def _get_rhill_list_and_neighbours(
    new_parent_set, 
    massive_ext, 
    number_of_neighbours,
    bridge_time
    ) -> (list, list):
    """
    Compute minimum Hill radius and approximate distance between new parents and external particles.
    Args:
        new_parent_set (Particles):  New parent particle set
        massive_ext (Particles):     Massive external particle set (for Hill radius calculation)
        number_of_neighbours (int):  Number of nearest neighbors to consider for Hill radius calculation
        bridge_time (units.time):    Nemesis bridge time step
    Returns:
        neigh_idx (list): List of nearest neighbor indices for each parent
        RH_list (list):   List of Hill radii for each parent
    """
    parent_vel = new_parent_set.velocity
    parent_pos = new_parent_set.position
    external_pos = massive_ext.position
    external_vel = massive_ext.velocity
    dr_grid = (parent_pos[:, np.newaxis] - external_pos).lengths()
    dv_grid = (parent_vel[:, np.newaxis] - external_vel).lengths()
    dr_bridge = dr_grid - (dv_grid * bridge_time)
    
    ### Exclude self
    external_map = {q.key: i for i, q in enumerate(massive_ext)}
    self_idx = np.array([external_map[p.key] for p in new_parent_set])
    dr_grid[np.arange(len(new_parent_set)), self_idx] = np.inf | units.m
    dr_bridge[np.arange(len(new_parent_set)), self_idx] = np.inf | units.m
    
    ### Get nearest neighbors and compute Hill radius
    K = len(new_parent_set) + number_of_neighbours
    neigh_idx = np.argsort(dr_grid, axis=1)[:, :K]
    
    ### Compute Hill radius
    parent_mass = new_parent_set.mass[:, np.newaxis]
    neighb_mass = massive_ext.mass[neigh_idx]
    neighb_dist = dr_grid[np.arange(len(new_parent_set))[:, np.newaxis], neigh_idx]

    Rhill_grid = hill_radius(parent_mass, neighb_mass, neighb_dist)
    RH_list = 0.5 * np.minimum(Rhill_grid.min(axis=1), dr_bridge.min(axis=1))
    
    return neigh_idx, RH_list


def _check_asteroid_splits(
    nem_class, 
    asteroids, 
    new_parent_set, 
    new_isolated, 
    massive_ext, 
    number_of_neighbours,
    bridge_time,
    fHill=0.1
    ) -> Particles:
    """
    Function to check for asteroid splits from parents. This is added to mitigate
    the issue of comets, where they can be flagged as ejected due to high eccentricity
    yet remain bound to parent system. Such bodies add splitting + parent merger times.
    Args:
        nem_class (Nemesis):         Nemesis instance
        asteroids (Particles):       Asteroid particle set
        new_parent_set (Particles):  New parent particle set
        new_isolated (Particles):    New isolated particle set
        massive_ext (Particles):     Massive external particle set (for Hill radius calculation)
        number_of_neighbours (int):  Number of nearest neighbors to consider for Hill radius calculation
        bridge_time (units.time):    Nemesis bridge time step
        fHill (float):               Fraction of Hill radius for orbital energy cutoff
    """
    ### Compute Hill Radius between parents, and relative to cluster
    neigh_idx, RH_list = _get_rhill_list_and_neighbours(
        new_parent_set, 
        massive_ext, 
        number_of_neighbours, 
        bridge_time
        )

    ### Compute orbital energy of asteroids relative to parents
    n_asts = len(asteroids)
    ast_orb_energy = np.ones(n_asts) | (units.ms)**2
    new_parent_key = np.full(n_asts, 0, dtype=np.uint64)

    ast_r = asteroids.position
    ast_v = asteroids.velocity
    for ip, new_par in enumerate(new_parent_set):
        rhill = RH_list[ip]
        dr_to_np = (ast_r - new_par.position)
        drij_to_np = dr_to_np.lengths()
        dr_mask = drij_to_np < rhill
        if not np.any(dr_mask):
            continue

        ### Two-body orbital energy calculation
        dv_to_np = (ast_v - new_par.velocity)
        orbital_energy = specific_orbital_energy(
            asteroids, new_par, dr=drij_to_np, dv=dv_to_np
            )
        
        ### Find neighbour and self for this parent
        neighbours = massive_ext[neigh_idx[ip]]
        rij_np = (new_par.position - neighbours.position).lengths()
        self_idx = neighbours.key == new_par.key
        rij_np[self_idx] = np.inf | units.m
        
        ### Compute effective potential of new parent from other parent particles
        peff_np = (-constants.G * (neighbours.mass / rij_np)).sum()
        
        ### Compute effective potential of asteroids from other parent particles
        cand_idx = np.where(dr_mask)[0]
        dr_ast_grid = (ast_r[cand_idx][:, np.newaxis] - neighbours.position).lengths()
        dr_ast_grid[:, self_idx] = np.inf | units.m
        peff_ast = (-constants.G * (neighbours.mass / dr_ast_grid)).sum(axis=1)

        ### Total effective potential
        peff = np.zeros(n_asts) | (units.ms)**2
        peff[cand_idx] = peff_ast - peff_np
        orbital_energy = orbital_energy + peff

        bounded_mask = orbital_energy < ast_orb_energy
        mask = bounded_mask & dr_mask
        if np.any(mask):
            dr_travelled = drij_to_np + (dv_to_np.lengths() * bridge_time)
            cutoff_rhill = -fHill * GRAV_CONST * new_par.mass / rhill
            mask = mask & (orbital_energy < cutoff_rhill) & (dr_travelled < rhill)

            ast_orb_energy[mask] = orbital_energy[mask]
            new_parent_key[mask] = new_par.key

    unique_keys = np.unique(new_parent_key)
    new_parent_map = {p.key: p for p in new_parent_set}
    for new_key in unique_keys:
        mask = new_parent_key == new_key
        children = asteroids[mask]
        if new_key == 0:  # Unbound asteroids
            if nem_class._verbose:
                print(f"{len(children)} unbound asteroids detected...")

            new_isolated.add_particles(children)
            continue

        if nem_class._verbose:
            print(f"{len(children)} asteroids energetically bound...")

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

    return new_isolated


def split_subcodes(nem_class, number_of_neighbours, bridge_time) -> None:
    """
    Check for any isolated children
    Args:
        nem_class (Nemesis):        Nemesis instance
        number_of_neighbours (int): Number of nearest neighbors for Hill radius calculation
        bridge_time (units.time):   Nemesis bridge time step
    """
    if nem_class._verbose:
        print("...Checking Splits...")

    new_isolated = Particles()
    for parent_key, (parent, subsys) in list(nem_class.children.items()):
        par_rad = parent.radius
        components = connected_components_kdtree(
            system=subsys,
            threshold=CONNECTED_COEFF * par_rad
            )
        if len(components) <= 1:
            continue

        if nem_class._verbose:
            print("...Split Detected...")

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
                    if nem_class._verbose:
                        print("Recycling old code")
                    rework_code = True
                    newcode = code
                    newcode.particles.remove_particles(code.particles)
                    newcode.particles.add_particles(sys)
                    nem_class._time_offsets[newcode] = offset
                    worker_pid = pid

                else:
                    if nem_class._verbose:
                        print("Making new code")
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
                    code_particles=newcode.particles,
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

            massive_ext = nem_class.particles[nem_class.particles.mass > 0.0 | units.kg]
            import time
            
            t0 = time.time()
            new_isolated = _check_asteroid_splits(
                nem_class, asteroids, new_parent_set, 
                new_isolated, massive_ext,
                number_of_neighbours=number_of_neighbours,
                bridge_time=bridge_time
                )
            print("Asteroid split check time: ", time.time() - t0)

        if not rework_code:  # Only triggered if pure ionisation
            code.cleanup_code()
            code.stop()

    if len(new_isolated) > 0:
        if nem_class._verbose:
            print(f"Adding {len(new_isolated)} isolated")

        new_isolated.radius = set_parent_radius(new_isolated.mass)
        mask = new_isolated.radius > PARENT_RADIUS_MAX
        new_isolated[mask].radius = PARENT_RADIUS_MAX
        nem_class.particles.add_particles(new_isolated)