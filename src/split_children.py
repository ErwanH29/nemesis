import numpy as np

from amuse.datamodel import Particles
from amuse.units import units

from src.environment_functions import (
    set_parent_radius, connected_components_kdtree
)
from src.globals import CONNECTED_COEFF, PARENT_RADIUS_MAX


def split_subcodes(nem_class) -> None:
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