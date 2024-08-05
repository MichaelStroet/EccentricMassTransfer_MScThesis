import os

from amuse.units import units
from amuse.datamodel import Particle

from amuse.ext.star_to_sph import pickle_stellar_model

from equations import *
from visualisations import animate_HR

################################################################################
# In/Output
################################################################################

def save_giant_model(giant_in_code, save_path):
    """
    Save the evolved giant model to file
    """
    print(f"\nSaving evolved giant model to\n{save_path}")
    pickle_stellar_model(giant_in_code, save_path)


################################################################################
# Run MESA evolution
################################################################################

def evolve_giant(giant_in_code, stop_radius):
    """
    Evolves the primary star until roche lobe overflow.
    :giant_in_code:   Giant particle in MESA
    :stop_radius:     Radius at which to stop evolution
    """

    # Arrays for data to animate an HR diagram
    HR_data = [[] | units.K, [] | units.LSun, [] | units.yr, [] | units.RSun]
    HR_data[0].append(giant_in_code.temperature)
    HR_data[1].append(giant_in_code.luminosity)
    HR_data[2].append(giant_in_code.age)
    HR_data[3].append(giant_in_code.radius)

    i = 0
    while (giant_in_code.radius < stop_radius):

        # Evolve the giant one step
        giant_in_code.evolve_one_step()

        # Save HR data
        HR_data[0].append(giant_in_code.temperature)
        HR_data[1].append(giant_in_code.luminosity)
        HR_data[2].append(giant_in_code.age)
        HR_data[3].append(giant_in_code.radius)

        # Print the status every 50 steps
        if i % 50 == 0:
            r = giant_in_code.radius.value_in(units.RSun)
            t = giant_in_code.age.value_in(10**6 * units.yr)
            L = giant_in_code.luminosity.value_in(units.LSun)
            T = giant_in_code.temperature.value_in(units.K)
            print(f"{i:04}: {r:10.3f} RSun | {t:10.3f} Myr | {L:10.3f} LSun | {T:10.3f} K")
        
        i += 1

    print(f"\nPrimary star post-evolution:\n{giant_in_code}\n")

    return giant_in_code, HR_data


################################################################################
# Main
################################################################################

def A_EvolveGiant(evolution_code, dirs, parameters):
    """
    Evolve the primary star until it overflows it roche lobe (* multiplier)
    :evolution_code:  Code to be used for stellar evolution (MESA r2208)
    :dirs:            Dictionary containing the paths to directories
    :parameters:      Dictionary containing the simulation parameters
    """

    m_giant = parameters["m_primary_MSun"] | units.MSun
    m_sec = parameters["m_secondary_MSun"] | units.MSun
    periapsis = parameters["periapsis_RSun"] | units.RSun

    # Setup the star in MESA
    giant = Particle(mass = m_giant)
    giant_in_code = evolution_code.particles.add_particle(giant)
    print(f"Primary star pre-evolution:\n{giant_in_code}\n")

    # Determine the radius at which to stop evolving
    roche_radius = estimate_roche_radius_periapsis(m_giant, m_sec, periapsis) # Independent of eccentricity
    stop_radius = parameters["roche_multiplier"] * roche_radius
    print(f"Roche radius = {roche_radius.value_in(units.RSun):.3f} RSun")
    print(f"Evolving giant to {parameters['roche_multiplier']} * RL = {stop_radius.value_in(units.RSun):.3f} RSun")


    # Evolve the primary star until Roche lobe overflow
    print(f"\nEvolving primary with MESA r{evolution_code.mesa_version}\n")
    evolved_giant, HR_data = evolve_giant(giant_in_code, stop_radius)

    # Save the evolved giant to file
    model_name = f"evolved_giant_M{parameters['m_primary_MSun']}_D{parameters['periapsis_RSun']}"
    model_path = os.path.join(dirs["models"], model_name)
    save_giant_model(evolved_giant, model_path)

    # Create an animation of the evolution
    filepath = os.path.join(dirs["plots"], "HR_diagram_evolution.gif")
    animate_HR(HR_data, parameters, filepath=filepath)

    # Clean up after evolution code
    evolution_code.stop()