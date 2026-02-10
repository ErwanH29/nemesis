import numpy as np

from amuse.datamodel import Particles
from amuse.units import units, constants

from src.environment_functions import (
    hill_radius, set_parent_radius, 
    connected_components_kdtree
)
from src.globals import CONNECTED_COEFF, PARENT_RADIUS_MAX


def _check_asteroid_splits(
    nem_class, asteroids, new_parent_set, new_isolated,
    massive_ext, number_of_neighbours
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
    """
    ### Compute Hill Radius between all new parents with other massive external particles     
    n_parents = len(new_parent_set)
    dr_NN = (np.inf | units.m) * np.ones(n_parents)
    for ip in range(n_parents):
        p = new_parent_set[ip]
        dr_ij = (p.position - massive_ext.position).lengths()
        if ip == 0:
            neighbour_idx = np.argsort(dr_ij)[:n_parents + number_of_neighbours]
            nearest_mass = massive_ext[neighbour_idx].mass
            external_map = {m.key: i for i, m in enumerate(massive_ext)}

        idx = external_map[p.key]
        dr_ij[idx] = np.inf | units.m  # Ignore self

        ext_Rhill = hill_radius(
            p.mass, nearest_mass, dr_ij[neighbour_idx]
            )
        dr_NN[ip] = 0.5 * min(ext_Rhill)
        print(f"Parent {ip} Hill radius: {dr_NN[ip].in_(units.au)}")

    ### Compute orbital energy of asteroids relative to parents
    n_asts = len(asteroids)
    ast_orb_energy = np.ones(n_asts) | (units.ms)**2
    new_parent_key = np.full(n_asts, 0, dtype=np.uint64)

    ast_r = asteroids.position
    ast_v = asteroids.velocity
    for ip in range(len(new_parent_set)):
        p = new_parent_set[ip]
        dr  = (ast_r - p.position).lengths()
        dr_mask = dr < dr_NN[ip]
        if not np.any(dr_mask):
            if nem_class._verbose:
                print(f"No asteroids within Rhill...")

            continue

        dv2 = (ast_v - p.velocity).lengths_squared()
        orbital_energy = 0.5 * dv2 - constants.G * p.mass / dr
        bounded_mask = orbital_energy < ast_orb_energy
        mask_energy = bounded_mask & dr_mask
        
        ast_orb_energy[mask_energy] = orbital_energy[mask_energy]
        new_parent_key[mask_energy] = p.key

    unique_keys = np.unique(new_parent_key)
    new_parent_map = {p.key: p for p in new_parent_set}
    for new_key in unique_keys:
        mask = new_parent_key == new_key
        children = asteroids[mask]

        if new_key == 0:  # Unbound asteroids
            if nem_class._verbose:
                print(f"{len(children)} unbound asteroids detected...")

            new_isolated.add_particles(children)

        else:
            if nem_class._verbose:
                print(f"{len(children)} asteroids within Rhill and energetically bound...")

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


def split_subcodes(nem_class, number_of_neighbours=5) -> None:
    """
    Check for any isolated children
    Args:
        nem_class (Nemesis):         Nemesis instance
        number_of_neighbours (int):  Number of nearest neighbors to consider for Hill radius calculation
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
            new_isolated = _check_asteroid_splits(
                nem_class, asteroids, new_parent_set, 
                new_isolated, massive_ext, 
                number_of_neighbours
                )

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