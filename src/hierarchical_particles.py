"""
Possible Room for Improvements:
- Recentering child systems in parallel can be cumbersome if the
  number of child systems is small. Also makes it less readable.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from amuse.datamodel import Particle, Particles, ParticlesOverlay
from amuse.units import units


class HierarchicalParticles(ParticlesOverlay):
    """Class to make particle set"""
    def __init__(self, *args, **kwargs):
        ParticlesOverlay.__init__(self, *args, **kwargs)
        self.collection_attributes.subsystems = dict()

    def add_particles(self, parts: Particles) -> Particles:
        """
        Add particles to particle set.
        Args:
            parts (Particles):  The particle to add.
        Returns:
            ParticlesOverlay:   The particle set.
        """
        _parts = ParticlesOverlay.add_particles(self, parts)
        if hasattr(parts.collection_attributes, "subsystems"):
            for parent, child in parts.collection_attributes.subsystems.values():
                parent = parent.as_particle_in_set(self)
                self.collection_attributes.subsystems[parent.key] = (parent, child)

        return _parts

    def assign_children(self, parent: Particle, child: Particles) -> None:
        """
        Assign children to parent particle. No reshifting needed.
        Args:
            parent (Particle):  The parent particle.
            child (Particles):  The child system particle set.
        """
        if not isinstance(child, Particles):
            raise TypeError("Child must be Particles instance.")

        if len(child) == 1:
            return self.add_particles(child)[0]

        self.collection_attributes.subsystems[parent.key] = (parent, child)
        return parent

    def add_children(self, child: Particles, recenter=True) -> Particle:
        """
        Create a parent from children.
        Args:
            child (Particles):  The child particle set.
            recenter (bool):    Flag to recenter the parent.
        Returns:
            Particle:  The parent particle.
        """
        if len(child) == 1:
            return self.add_particles(child)[0]

        parent = Particle()
        self.assign_parent_attributes(
            child, parent,
            relative=False,
            recenter=recenter
            )
        parent = self.add_particle(parent)
        self.collection_attributes.subsystems[parent.key] = (parent, child)

        return parent

    def assign_parent_attributes(
        self,
        child: Particles,
        parent: Particle,
        relative=True,
        recenter=True
    ) -> None:
        """
        Create parent from children attributes.
        Args:
            child (Particles):  The child system particle set.
            parent (Particle):  The parent particle.
            relative (bool):    Flag to assign relative attributes.
            recenter (bool):    Flag to recenter the parent.
        """
        if not relative:
            parent.position = 0.*child[0].position
            parent.velocity = 0.*child[0].velocity

        massives = child[child.mass > (0. | units.kg)]
        parent.mass = np.sum(massives.mass)
        try:
            if recenter:
                com = massives.center_of_mass()
                com_vel = massives.center_of_mass_velocity()

                parent.position = com
                parent.velocity = com_vel
                child.position -= com
                child.velocity -= com_vel

        except Exception as e:
            raise ValueError(f"Error calculating parent attributes: {e}")

    def recenter_children(self, max_workers: int) -> None:
        """
        Recenter child systems.
        Args:
            max_workers (int):  Number of cores to use.
        """
        def calculate_com(
            par_pos: units.length,
            par_vel: units.velocity,
            chd_set: Particles
        ) -> tuple:
            """
            Calculate and shift system relative to center of mass.
            Args:
                par_pos (units.length):    Parent particle position.
                par_vel (units.velocity):  Parent particle velocity.
                chd_set (Particles):       The child particle set.
            Returns:
                tuple:  The shifted position and velocity.
            """
            massives = chd_set[chd_set.mass > (0. | units.kg)]
            masses = massives.mass.value_in(units.kg)
            system_pos = massives.position.value_in(units.m)
            system_vel = massives.velocity.value_in(units.ms)

            com = np.average(system_pos, weights=masses, axis=0) | units.m
            com_vel = np.average(system_vel, weights=masses, axis=0) | units.ms

            chd_set.position -= com
            chd_set.velocity -= com_vel
            par_pos += com
            par_vel += com_vel

            return par_pos, par_vel

        children_systems = self.collection_attributes.subsystems
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    calculate_com,
                    parent.position,
                    parent.velocity,
                    child
                    ): parent
                for parent, child in children_systems.values()
            }

            for future in as_completed(futures):
                parent = futures.pop(future)
                com_pos, com_vel = future.result()
                parent.position = com_pos
                parent.velocity = com_vel

    def remove_particles(self, parts: Particles) -> None:
        """
        Remove particles from particle set.
        Args:
            parts (Particles):  Particle to remove.
        """
        for p in parts:
            self.collection_attributes.subsystems.pop(p.key, None)

        ParticlesOverlay.remove_particles(self, parts)

    def all(self) -> Particles:
        """
        Get copy of complete particle set. Children are shifted
        to their parent position and velocity.
        Returns:
            Particles:  Complete data on simulated particle set.
        """
        parts = self.copy()
        parts.syst_id = -1

        children_systems = self.collection_attributes.subsystems
        for system_id, (parent, child) in enumerate(children_systems.values()):
            parts.remove_particle(parent)

            chd_set = parts.add_particles(child)
            chd_set.position += parent.position
            chd_set.velocity += parent.velocity
            chd_set.syst_id = system_id + 1

        return parts
