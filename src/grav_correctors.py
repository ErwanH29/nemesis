"""
STILL IN DEVELOPMENT. 
1. AMUSIFY C++ LIBRARY WITH INTERFACE  --> ASK ORIGINAL AUTHOR FOR SKELETON
2. RELEASE GIL IN C++ LIBRARY
3. GET_POTENTIAL_AT_POINT FUNCTION NOT USED --> TO VALIDATE
"""


from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from amuse.lab import units, Particles

from src.globals import ACC_UNITS, SI_UNITS



def _as_float64_si(q, target_unit) -> np.ndarray:
    """
    Convert an AMUSE Quantity (scalar or vector) to a 
    float64 numpy array in target_unit.
    """
    arr = np.asarray(q.value_in(target_unit), dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.ravel()


def compute_gravity(
    grav_lib,
    pert_m,
    pert_x,
    pert_y,
    pert_z,
    infl_x,
    infl_y,
    infl_z,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gravitational force felt by perturber particles due to externals
    Args:
        grav_lib (library):         Library to compute gravity
        pert_m (units.mass):        Mass of perturber particles
        pert_x/y/z (units.length):  x/y/z coordinate of perturber particles
        infl_x/y/z (units.length):  x/y/z coordinate of influenced particles
    Returns:
        tuple:  Acceleration array of particles (ax, ay, az)
    """
    pm = _as_float64_si(pert_m, units.kg)
    px = _as_float64_si(pert_x, units.m)
    py = _as_float64_si(pert_y, units.m)
    pz = _as_float64_si(pert_z, units.m)
    
    ix = _as_float64_si(infl_x, units.m)
    iy = _as_float64_si(infl_y, units.m)
    iz = _as_float64_si(infl_z, units.m)
    
    n_pert = len(pm)
    n_part = len(ix)

    ax = np.zeros(n_part, dtype=np.float64)
    ay = np.zeros(n_part, dtype=np.float64)
    az = np.zeros(n_part, dtype=np.float64)

    grav_lib.find_gravity_at_point(
        pm, px, py, pz,
        ix, iy, iz,
        ax, ay, az,
        n_part, n_pert
    )
    
    return ax, ay, az


def correct_parents_threaded(
        lib, acc_units,
        particles_pos,
        parent_mass, 
        parent_pos,
        child_mass,
        child_pos,
        removed_idx
        ):
    """
    Correct the gravitational influence of a parent particle on its child system.
    Args:
        lib (library):                   Library to compute gravity
        acc_units (units):               Units of acceleration
        particles_pos (units.length):    x/y/z coordinate of all particles
        parent_mass (units.mass):        Mass of the parent particle
        parent_pos (units.length):       x/y/z coordinate of the parent particle
        child_mass (units.mass):         Mass of the child particles
        child_pos (units.length):        x/y/z coordinate of the child particles
        removed_idx (int):               Index of the parent particle in the original array
    Returns:
        tuple:  Acceleration array of parent particles (ax, ay, az)
    """
    external_x = particles_pos[0]
    external_y = particles_pos[1]
    external_z = particles_pos[2]
    
    parent_x = parent_pos[0]
    parent_y = parent_pos[1]
    parent_z = parent_pos[2]
    
    child_x = child_pos[0]
    child_y = child_pos[1]
    child_z = child_pos[2]
    
    mask = np.ones(len(external_x), dtype=bool)
    mask[removed_idx] = False
    external_x = external_x[mask]
    external_y = external_y[mask]
    external_z = external_z[mask]

    ax_chd, ay_chd, az_chd = compute_gravity(
        grav_lib=lib,
        pert_m=child_mass,
        pert_x=child_x + parent_x,
        pert_y=child_y + parent_y,
        pert_z=child_z + parent_z,
        infl_x=external_x,
        infl_y=external_y,
        infl_z=external_z
    )

    ax_par, ay_par, az_par = compute_gravity(
        grav_lib=lib,
        pert_m=parent_mass,
        pert_x=parent_x,
        pert_y=parent_y,
        pert_z=parent_z,
        infl_x=external_x,
        infl_y=external_y,
        infl_z=external_z,
    )

    corr_ax = ((ax_chd - ax_par) * SI_UNITS).value_in(acc_units).astype(np.float64)
    corr_ay = ((ay_chd - ay_par) * SI_UNITS).value_in(acc_units).astype(np.float64)
    corr_az = ((az_chd - az_par) * SI_UNITS).value_in(acc_units).astype(np.float64)

    corr_ax = np.insert(corr_ax, removed_idx, 0.0)
    corr_ay = np.insert(corr_ay, removed_idx, 0.0)
    corr_az = np.insert(corr_az, removed_idx, 0.0)

    return (
        corr_ax | acc_units,
        corr_ay | acc_units,
        corr_az | acc_units
        )


class CorrectionFromCompoundParticle(object):
    def __init__(
        self, grav_lib, particles, 
        particles_x, particles_y, particles_z, 
        children: Particles, num_of_workers: int
        ):
        """
        Correct force exerted by some parent system on other particles by that of its system.
        Args:
            grav_lib (Library):              The gravity library (e.g., a wrapped C++ library).
            particles (units.length):        Original parent particle set
            particles_x/y/z (units.length):  x/y/z coordinate of particles
            children (Particles):            Collection of children present
            num_of_workers (int):            Number of cores to use
        """
        self.particles = particles
        self.particles_x = particles_x
        self.particles_y = particles_y
        self.particles_z = particles_z
        self.children = children

        self.lib = grav_lib
        self.max_workers = num_of_workers

    def get_gravity_at_point(self, radius, x, y, z) -> tuple:
        """
        Compute difference in gravitational acceleration felt by parents
        due to force exerted by parents which host system, and force
        exerted by their system.

        :math:`dF = \sum_{j} \left( \sum_{i} F_{i} - F_{j} \right)`

        where j is parent and i is constituent childrens of parent j.
        Args:
            radius (units.length):  Radius of parent particles
            x/y/z (units.length):   x/y/z coordinate of parent particles
        Returns:
            tuple:  Acceleration array of parent particles (ax, ay, az)
        """
        Nparticles = len(self.particles_x)

        ax_corr = np.zeros(Nparticles) | ACC_UNITS
        ay_corr = np.zeros(Nparticles) | ACC_UNITS
        az_corr = np.zeros(Nparticles) | ACC_UNITS

        parent_idx = {p.key: i for i, p in enumerate(self.particles)}
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for parent, system in list(self.children.values()):
                removed_idx = parent_idx.pop(parent.key)

                future = executor.submit(
                    correct_parents_threaded,
                    lib=self.lib,
                    acc_units=ACC_UNITS,
                    particles_pos=[
                        self.particles_x,
                        self.particles_y,
                        self.particles_z
                    ],
                    parent_mass=parent.mass,
                    parent_pos=[
                        parent.x,
                        parent.y,
                        parent.z
                    ],
                    child_mass=system.mass,
                    child_pos=[
                        system.x,
                        system.y,
                        system.z
                    ],
                    removed_idx=removed_idx
                    )
                futures.append(future)

            for future in as_completed(futures):
                ax, ay, az = future.result()
                ax_corr += ax
                ay_corr += ay
                az_corr += az

        return ax_corr, ay_corr, az_corr

    def get_potential_at_point(self, radius, x, y, z) -> np.ndarray:
        """
        Get the potential at a specific location
        Args:
            radius (units.length):  Radius of the particle at that location
            x/y/z (units.length):   x/y/z coordinate of the location
        Returns:
            Array:  The potential field at the location
        """
        raise NotImplementedError("Potential correction is not yet implemented.")


class CorrectionForCompoundParticle(object):  
    def __init__(
        self, grav_lib, parent_x, parent_y, parent_z,
        child_x, child_y, child_z, child: Particles,
        perturber_mass, perturber_x, perturber_y, perturber_z
        ):
        """
        Correct force vector exerted by global particles on systems
        Args:
            grav_lib (Library):             The gravity library (e.g., a wrapped C++ library).
            parent_x/y/z (units.length):    x/y/z coordinate of the parent particle.
            child_x/y/z (units.length):     x/y/z coordinate of the system particle.
            child (Particles):              The child particles.
            perturber_mass (units.mass):    Mass of the perturber particle.
            perturber_x/y/z (units.length): x/y/z coordinate of the perturber particle.
        """
        self.lib = grav_lib

        self.parent_x = parent_x
        self.parent_y = parent_y
        self.parent_z = parent_z

        self.child = child
        self.child_x = child_x
        self.child_y = child_y
        self.child_z = child_z

        self.pert_mass = perturber_mass
        self.pert_x = perturber_x
        self.pert_y = perturber_y
        self.pert_z = perturber_z

    def get_gravity_at_point(self, radius, x, y, z) -> tuple:
        """
        Compute gravitational acceleration felt by system due to parents present.
        Args:
            radius (units.length):  Radius of the system particle
            x/y/z (units.length):   x/y/z coordinate of the system particle
        Returns: 
            tuple:  Acceleration array of system particles (ax, ay, az)
        """
        Nsystem = len(self.child)
        dax = np.zeros(Nsystem) | ACC_UNITS
        day = np.zeros(Nsystem) | ACC_UNITS
        daz = np.zeros(Nsystem) | ACC_UNITS

        ax_chd, ay_chd, az_chd = compute_gravity(
            grav_lib=self.lib, 
            pert_m=self.pert_mass, 
            pert_x=self.pert_x,
            pert_y=self.pert_y,
            pert_z=self.pert_z,
            infl_x=self.child_x,
            infl_y=self.child_y,
            infl_z=self.child_z
            )

        ax_par, ay_par, az_par = compute_gravity(
            grav_lib=self.lib, 
            pert_m=self.pert_mass, 
            pert_x=self.pert_x,
            pert_y=self.pert_y,
            pert_z=self.pert_z,
            infl_x=self.parent_x,
            infl_y=self.parent_y,
            infl_z=self.parent_z,
            )

        dax += (ax_chd - ax_par) * SI_UNITS
        day += (ay_chd - ay_par) * SI_UNITS
        daz += (az_chd - az_par) * SI_UNITS

        return dax, day, daz

    def get_potential_at_point(self, radius, x, y, z) -> np.ndarray:
        """
        Get the potential at a specific location.
        Args:
            radius (units.length):  Radius of the system particle
            x/y/z (units.length):   x/y/z Location of the system particle
        Returns:
            Array:  The potential field at the system particle's location
        """
        raise NotImplementedError("Potential correction is not yet implemented.")


class CorrectionKicks(object):
    def __init__(self, grav_lib, avail_cpus: int):
        """
        Apply correction kicks onto particles.
        Args:
            grav_lib (Library):  The gravity library (e.g., a wrapped C++ library).
            avail_cpus (int):    Number of available CPU cores
        """
        self.lib = grav_lib
        self.avail_cpus = avail_cpus
        
    def _kick_particles(self, particles: Particles, corr_code, dt) -> None:
        """
        Apply correction kicks onto target particles.
        Args:
            particles (Particles):  Particles whose accelerations are corrected
            corr_code (Code):  Object providing the difference in gravity
            dt (units.time):   Time-step of correction kick
        """
        parts = particles.copy()
        ax, ay, az = corr_code.get_gravity_at_point(
            particles.radius,
            particles.x, 
            particles.y, 
            particles.z
            )

        parts.vx = particles.vx + dt * ax
        parts.vy = particles.vy + dt * ay
        parts.vz = particles.vz + dt * az

        channel = parts.new_channel_to(particles)
        channel.copy_attributes(["vx","vy","vz"])

    def _correct_children(
            self, 
            pert_mass, 
            pert_x, 
            pert_y, 
            pert_z,
            parent_x, 
            parent_y, 
            parent_z, 
            child: Particles, 
            dt
            ) -> None:
        """
        Apply correcting kicks onto children particles.
        Args:
            pert_mass (units.mass):      Mass of perturber
            pert_x/y/z (units.length):   x/y/z position of perturber
            parent_x/y/z (units.length): x/y/z position of parent
            child (Particles):           Children particle set
            dt (units.time):             Time interval for applying kicks
        """
        child_x = child.x + parent_x
        child_y = child.y + parent_y
        child_z = child.z + parent_z
        
        corr_par = CorrectionForCompoundParticle(
            grav_lib=self.lib,
            parent_x=parent_x,
            parent_y=parent_y,
            parent_z=parent_z, 
            child=child,
            child_x=child_x,
            child_y=child_y,
            child_z=child_z, 
            perturber_mass=pert_mass,
            perturber_x=pert_x,
            perturber_y=pert_y,
            perturber_z=pert_z,
            )
        self._kick_particles(child, corr_par, dt)
       
    def _correction_kicks(
            self, 
            particles: Particles, 
            children: dict, dt
            ) -> None:
        """
        Apply correcting kicks onto children and parent particles.
        Args:
            particles (Particles):  Parent particle set
            children (dict):        Dictionary of children system
            dt (units.time):        Time interval for applying kicks
            kick_par (boolean):     Whether to apply correction to parents
        """
        def process_children_jobs(parent, children):
            removed_idx = abs(particles_key - parent.key).argmin()
            mask = np.ones(len(particles_mass), dtype=bool)
            mask[removed_idx] = False
            
            pert_mass = particles_mass[mask]
            pert_xpos = particles_x[mask]
            pert_ypos = particles_y[mask]
            pert_zpos = particles_z[mask]

            future = executor.submit(
                self._correct_children,
                pert_mass=pert_mass,
                pert_x=pert_xpos,
                pert_y=pert_ypos,
                pert_z=pert_zpos,
                parent_x=parent.x,
                parent_y=parent.y,
                parent_z=parent.z,
                child=children,
                dt=dt
                )

            return future

        if len(children) > 0 and len(particles) > 1:
            # Setup array for CorrectionFor
            particles_key  = particles.key
            particles_mass = particles.mass
            particles_x = particles.x
            particles_y = particles.y
            particles_z = particles.z
            
            corr_chd = CorrectionFromCompoundParticle(
                grav_lib=self.lib,
                particles=particles,
                particles_x=particles_x,
                particles_y=particles_y,
                particles_z=particles_z,
                children=children,
                num_of_workers=self.avail_cpus
                )
            self._kick_particles(particles, corr_chd, dt)
            del corr_chd

            futures = []
            with ThreadPoolExecutor(max_workers=self.avail_cpus) as executor:
                try:
                    for parent, child in children.values():
                        future = process_children_jobs(parent, child)
                        futures.append(future)
                    for future in as_completed(futures):
                        future.result()
                except Exception as e:
                    raise RuntimeError(f"Error during correction kicks: {e}")