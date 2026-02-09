import numpy as np

from amuse.datamodel import Particles
from amuse.units import units

from src.environment_functions import (
    set_parent_radius, connected_components_kdtree
)
from src.globals import CONNECTED_COEFF, PARENT_RADIUS_MAX


def split_subcodes(self) -> None:
    """Check for any isolated children"""
    if self._verbose:
        print("...Checking Splits...")

    new_isolated = Particles()
    for parent_key, (parent, subsys) in list(self.children.items()):
        par_rad = parent.radius
        components = connected_components_kdtree(
            system=subsys,
            threshold=CONNECTED_COEFF * par_rad
            )
        if len(components) <= 1:
            continue

        if self._verbose:
            print("...Split Detected...")

        rework_code = False
        par_vel = parent.velocity
        par_pos = parent.position

        pid = self._pid_workers.pop(parent_key)
        self.resume_workers(pid)
        self.particles.remove_particle(parent)

        code = self.subcodes.pop(parent_key)
        offset = self._time_offsets.pop(code)
        self._child_channels.pop(parent_key)
        cpu_time = self._cpu_time.pop(parent_key)
        
        asteroids = Particles()
        for c in components:
            sys = c.as_set()
            sys.position += par_pos
            sys.velocity += par_vel

            has_massive = (len(sys) > 1) and np.any(sys.mass.value_in(units.kg) > 0.0)
            if has_massive:
                newparent = self.particles.add_children(sys)
                newparent_key = newparent.key

                scale_mass = newparent.mass
                scale_radius = set_parent_radius(scale_mass)
                newparent.radius = scale_radius
                if not rework_code:  # Recycle old code
                    if self._verbose:
                        print("Recycling old code")
                    rework_code = True
                    newcode = code
                    newcode.particles.remove_particles(code.particles)
                    newcode.particles.add_particles(sys)
                    self._time_offsets[newcode] = offset
                    worker_pid = pid

                else:
                    if self._verbose:
                        print("Making new code")
                    newcode, worker_pid = self._sub_worker(
                        children=sys,
                        scale_mass=scale_mass,
                        scale_radius=scale_radius
                        )
                    self._set_worker_affinity(worker_pid)
                    self._time_offsets[newcode] = self.model_time

                self._cpu_time[newparent_key] = cpu_time
                self.subcodes[newparent_key] = newcode

                self._child_channel_maker(
                    parent_key=newparent_key,
                    code_particles=newcode.particles,
                    children=sys
                )
                self._child_channels[newparent_key]["from_children_to_gravity"].copy()  # More precise

                self._pid_workers[newparent_key] = worker_pid
                self.hibernate_workers(worker_pid)
            else:
                new_isolated.add_particles(sys)

        if not rework_code:  # Only triggered if pure ionisation
            code.cleanup_code()
            code.stop()

    if len(new_isolated) > 0:
        if self._verbose:
            print(f"Adding {len(new_isolated)} isolated")
        new_isolated.radius = set_parent_radius(new_isolated.mass)
        mask = new_isolated.radius > PARENT_RADIUS_MAX
        new_isolated[mask].radius = PARENT_RADIUS_MAX
        self.particles.add_particles(new_isolated)