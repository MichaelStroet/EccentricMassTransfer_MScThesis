import numpy as np
from amuse.units import units, constants

################################################################################
# Orbital equations
################################################################################

def orbital_separation(e, a, v):
    """
    Calculates the distance between two stars on an eccentric orbit
    :e:  Eccentricity
    :a:  Semi-major axis
    :v:  True anomaly (periastron at v=0, apastron at v=pi)
    """
    D = a * (1 - e**2) / (1 + (e * np.cos(v)))
    return D.in_(units.RSun)

def smaxis_from_periapsis(e, D):
    """
    Calculates the semi-major axis of an orbit from periapsis and eccentricity
    :e:  Eccentricity
    :D:  Periapsis distance
    """
    a = D / (1 - e)
    return a.in_(units.RSun)

def apoapsis_from_periapsis(e, D):
    """
    Calculates the apoapsis
    :e:  Eccentricity
    :D:  Periapsis distance
    """
    apoapsis = D * (1 + e) / (1 - e)
    return apoapsis.in_(units.RSun)

def relative_velocity_apastron(Mp, Ms, e, a):
    """
    Calculates the orbital velocity of the secondary star at apastron
    in the rest frame of the primary star.
    V_apa = sqrt(G(M+m)/a * (1-e)/(1+e))
    :Mp:  Mass of the primary
    :Ms:  Mass of the secondary
    :e:   Eccentricity
    :a:   Semi-major axis
    """
    vel = np.sqrt((constants.G * (Mp + Ms) * (1 - e)) / (a * (1 + e)))
    return vel.in_(units.km / units.s)

def period_to_smaxis(Mp, Ms, P):
    """
    Calculates the period of an orbit given the masses and the semi major axis.
    :Mp:  Mass of the primary star
    :Ms:  Mass of the secondary star
    :P:   Orbital period
    """
    constant = (constants.G * (Mp + Ms)) / (2 * constants.pi)**2
    sm_axis = (constant * P**2)**(1/3)
    return sm_axis.in_(units.RSun)

def smaxis_to_period(Mp, Ms, a):
    """
    Calculates the semi major axis of an orbit given the masses and the period.
    :Mp:  Mass of the primary star
    :Ms:  Mass of the secondary star
    :a:   Semi-major axis
    """
    constant = (constants.G * (Mp + Ms)) / (2 * constants.pi)**2
    period = np.sqrt(a**3 / constant)
    return period.in_(units.day)

################################################################################
# Orbital elements from particles
################################################################################

def orbit_from_particles(sph_particles, dm_particles):
    """
    Calculate the semi-major axis and eccentricity of the orbit from the particles
    """
    sec = dm_particles[0]
    core = dm_particles[1]

    # Positions
    r_rel = core.position - sec.position # relative distance vector
    r = np.sqrt(np.sum(r_rel**2)) # relative separation

    # Velocities
    v_rel = core.velocity - sec.velocity # relative velocity vector    
    v2 = np.sum(v_rel**2) # relative speed squared

    # Mass
    M = core.mass + np.sum(sph_particles.mass) + sec.mass # Total mass
    mu = constants.G * M

    # Calculate the semi-major axis
    E = 0.5 * v2 - mu/r # specific mechanical energy
    sm_axis = -mu / (2 * E) # Rewritten vis-visa equation

    # Calculate the eccentricity
    eccentricity_vector = ((v2 - mu/r) * r_rel - (r_rel.dot(v_rel)) * v_rel) / mu
    eccentricity = np.sqrt(np.sum(eccentricity_vector**2))
    
    return sm_axis.in_(units.RSun), eccentricity

################################################################################
# Roche lobe estimate
################################################################################

def estimate_periapsis_from_roche_radius(m_giant, m_sec, roche_radius):
    """
    Calculates the Eggleton (1983) estimate for the roche radius (assuming co-rotation).
    Here the roche lobe is defined at closest approach (periapsis) which is given as
    D = a (1-e^2) / (1 + e cos(v)), where true anomaly v = 0.
    :m_giant:   Mass of the primary star
    :m_sec:     Mass of the secondary star
    :ecc:       Eccentricity
    :sm_axis:   Semi-major axis
    Returns roche lobe radius in RSun
    """

    # Calculating mass ratio q and cube roots
    q = m_giant / m_sec
    q_13root = q**(1/3)
    q_23root = q_13root**2

    # Eggleton (1983): R(L1) = a * 0.49 q^2/3 / (0.69 q^2/3 = ln(1 + q^1/3))
    periapsis = roche_radius / (0.49 * q_23root / (0.69 * q_23root + np.log(1 + q_13root)))

    return periapsis.as_quantity_in(units.RSun)

def estimate_roche_radius(m_giant, m_sec, ecc, sm_axis):
    """
    Calculates the Eggleton (1983) estimate for the roche radius (assuming co-rotation).
    Here the roche lobe is defined at closest approach (periapsis) which is given as
    D = a (1-e^2) / (1 + e cos(v)), where true anomaly v = 0.
    :m_giant:   Mass of the primary star
    :m_sec:     Mass of the secondary star
    :ecc:       Eccentricity
    :sm_axis:   Semi-major axis
    Returns roche lobe radius in RSun
    """

    # Calculating mass ratio q and cube roots
    q = m_giant / m_sec
    q_13root = q**(1/3)
    q_23root = q_13root**2

    # Eggleton (1983): R(L1) = a * 0.49 q^2/3 / (0.69 q^2/3 = ln(1 + q^1/3))
    # Replacing a with D(a,e,v) following Davis, Siess and Deschamps (2013)
    D = orbital_separation(ecc, sm_axis, 0) # v = 0 at periapsis
    roche_lobe = D * (0.49 * q_23root / (0.69 * q_23root + np.log(1 + q_13root)))

    return roche_lobe.as_quantity_in(units.RSun)

def estimate_roche_radius_parameters(parameters):
    """ Estimates the roche radius from the parameters directly """
    m_giant = parameters["m_primary_MSun"] | units.MSun
    m_sec = parameters["m_secondary_MSun"] | units.MSun
    ecc = parameters["eccentricity"]
    sm_axis = parameters["semi_major_axis_RSun"] | units.RSun

    return estimate_roche_radius(m_giant, m_sec, ecc, sm_axis)

def estimate_roche_radius_periapsis(m_giant, m_sec, periapsis):
    """
    Calculates the Eggleton (1983) estimate for the roche radius (assuming co-rotation).
    Here the roche lobe is defined at closest approach (periapsis).
    :m_giant:    Mass of the primary star
    :m_sec:      Mass of the secondary star
    :periapsis:  Periapsis distance
    """

    # Calculating mass ratio q and cube roots
    q = m_giant / m_sec
    q_13root = q**(1/3)
    q_23root = q_13root**2

    # Eggleton (1983): R(L1) = a * 0.49 q^2/3 / (0.69 q^2/3 = ln(1 + q^1/3))
    roche_radius = periapsis * (0.49 * q_23root / (0.69 * q_23root + np.log(1 + q_13root)))

    return roche_radius.as_quantity_in(units.RSun)

def estimate_roche_radius_periapsis_parameters(parameters):
    """ Estimates the roche radius from the parameters directly """
    m_giant = parameters["m_primary_MSun"] | units.MSun
    m_sec = parameters["m_secondary_MSun"] | units.MSun
    periapsis = parameters["periapsis_RSun"] | units.RSun

    return estimate_roche_radius_periapsis(m_giant, m_sec, periapsis)