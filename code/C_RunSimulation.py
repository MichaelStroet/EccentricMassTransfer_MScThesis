import os
import numpy as np
from tqdm import tqdm

from amuse.datamodel import Particles
from amuse.io import write_set_to_file, read_set_from_file
from amuse.ext.star_to_sph import StellarModelInSPH
from amuse.ext.sink import new_sink_particles
from amuse.couple.bridge import Bridge, CalculateFieldForParticles

from amuse.units import units, constants, nbody_system
from amuse.units.generic_unit_converter import ConvertBetweenGenericAndSiUnits

from equations import *

################################################################################
# In/Output
################################################################################

def load_relaxed_giant(gas_model, core_model):
    """
    Loads the relaxed giant model
    :gas_model:    Path to the gas particles file
    :core_model:   Path to the core particle file
    """
    # Load the relaxed SPH particles
    sph_particles = read_set_from_file(gas_model, format="amuse")

    # Load the corresponding core particle
    core_particle = read_set_from_file(core_model, format="amuse")[0]

    # Convert to named tuple StellarModelInSPH
    giant_model = StellarModelInSPH(
        gas_particles = sph_particles,
        core_particle = core_particle,
        core_radius = core_particle.radius)

    return giant_model

def load_simulation_snapshot(snapshot_gas, snapshot_dm, dirs):
    """
    Loads particles from the given simulation snapshots
    :snapshot_gas:  Name of the snapshot file containing the sph particles
    :snapshot_dm:   Name of the snapshot file containing the core and secondary particles
    :dirs:          Dictionary containing the paths to directories
    """
    # Get the snapshot step and time
    split = snapshot_gas[:-7].split("_")
    step = int(split[-2])
    time = float(split[-1])
    
    # Load the snapshot particles
    print(f"Restarting simulation from snapshots:\n{snapshot_gas}\n{snapshot_dm}\n\nStep: {step}\nTime: {time} days\n")
    sph_particles = read_set_from_file(os.path.join(dirs["snapshots"], snapshot_gas), format="amuse")
    dm_particles = read_set_from_file(os.path.join(dirs["snapshots"], snapshot_dm), format="amuse")
    secondary_particle_set = dm_particles[0].as_set()
    core_particle = dm_particles[1]

    # Convert to named tuple StellarModelInSPH
    giant_model = StellarModelInSPH(
        gas_particles = sph_particles,
        core_particle = core_particle,
        core_radius = core_particle.radius)

    return giant_model, secondary_particle_set, time, step+1
################################################################################
# Setup
################################################################################

def set_up_binary_system(giant_model, parameters):
    """
    Set up the binary system orbit with two particles
    :giant_model:   Named-tuple containing the core and sph particles   
    :parameters:    Dictionary containing the simulation parameters
    """
    # Parameters of the binaries & orbit
    m_giant = parameters["m_primary_MSun"] | units.MSun
    m_sec = parameters["m_secondary_MSun"] | units.MSun
    ecc = parameters["eccentricity"]
    sm_axis = parameters["periapsis_RSun"] / (1 - ecc) | units.RSun

    # Create amuse particles for the binary
    binary = Particles(2)

    # Add mass, position and velocity to the particles
    # in reference frame of the primary giant star
    binary.mass = [m_giant, m_sec]
    binary.position = [0.0, 0.0, 0.0] | units.RSun
    binary.velocity = [0.0, 0.0, 0.0] | units.km / units.s

    # Set the position and velocity of the secondary star at apastron
    # in the reference frame of the primary star
    binary[1].x = orbital_separation(ecc, sm_axis, np.pi) # apastron v = pi
    binary[1].vy = relative_velocity_apastron(m_giant, m_sec, ecc, sm_axis)

    # Move the reference frame to the center of mass of the system
    binary.move_to_center()
    
    # Match the position and velocities of the particles to the primary
    giant_model.gas_particles.position += binary[0].position
    giant_model.gas_particles.velocity += binary[0].velocity

    # Match the position and velocities of the core to the primary
    giant_model.core_particle.position += binary[0].position
    giant_model.core_particle.velocity += binary[0].velocity

    return giant_model, binary[1].as_set()


def set_up_hydrodynamics(sph_code, giant_model, dirs, parameters):
    """
    Sets up the SPH code for the giant sph and core particles.
    :sph_code:      SPH code to be used
    :giant_model:   Named-tuple containing the core and sph particles
    :dirs:          Dictionary containing the paths to directories
    :parameters:    Dictionary containing the simulation parameters
    """

    # Set up hydrodynamics code
    code_mass_unit = parameters["code_mass_unit"] | units.MSun
    code_length_unit = parameters["code_length_unit"] | units.RSun
    code_time_unit = parameters["code_time_unit"] | units.day
    unit_converter = ConvertBetweenGenericAndSiUnits(code_mass_unit, code_length_unit, code_time_unit)
    hydro_options = dict(number_of_workers=parameters["n_workers"],
                         redirection="file", # Save log to file <redirect_file>
                         redirect_file=os.path.join(dirs["data"], "sph_code_sim_out.log"))
    hydro = sph_code(unit_converter, **hydro_options)

    # Set additional hydro parameters
    hydro.parameters.epsilon_squared = (giant_model.gas_particles.h_smooth)**2 # Use SPH smoothing length as epsilon

    # Time settings such that code never ends prematurely
    hydro.parameters.time_max = 2*parameters["max_simulation_time_day"] | units.day   # Maximum in-code time.
    hydro.parameters.time_limit_cpu = 1 | units.yr                                    # Maximum real time

    # Add the particles to hydro code (AFTER SETTING OPTIONS!)
    hydro.gas_particles.add_particles(giant_model.gas_particles)
    hydro.dm_particles.add_particle(giant_model.core_particle)

    # Create channels between the local (giant_model) and code (hydro) variables
    hydro_channels = {}
    hydro_channels["local_to_code"] = giant_model.gas_particles.new_channel_to(hydro.gas_particles)
    hydro_channels["code_to_local_gas"] = hydro.gas_particles.new_channel_to(giant_model.gas_particles)
    hydro_channels["code_to_local_core"] = hydro.dm_particles.new_channel_to(giant_model.core_particle.as_set())

    return hydro, hydro_channels

def set_up_dynamics(nbody_code, second_particle, dirs, parameters):
    """
    Sets up the N-body code for the secondary star.
    Creates channels between the local and code variables.
    :nbody_code:       N-body code to be used
    :second_particle:  The secondary star as an amuse particle set
    :dirs:             Dictionary containing the paths to directories
    :parameters:       Dictionary containing the simulation parameters
    """

    # Set up nbody dynamics code
    code_mass_unit = parameters["code_mass_unit"] | units.MSun
    code_length_unit = parameters["code_length_unit"] | units.RSun
    unit_converter = nbody_system.nbody_to_si(code_mass_unit, code_length_unit)
    nbody_options = dict(redirection="file",  # Save log to file <redirect_file>
                         redirect_file=os.path.join(dirs["data"], "nbody_code_sim_out.log"))
    nbody = nbody_code(unit_converter, **nbody_options)

    # Add the particles to nbody code (AFTER SETTING OPTIONS!)
    nbody.particles.add_particles(second_particle)

    # Create channels between the local (second_particle) and code (nbody) variables
    nbody_channels = {}
    nbody_channels["local_to_code"] = second_particle.new_channel_to(nbody.particles)
    nbody_channels["code_to_local"] = nbody.particles.new_channel_to(second_particle)

    return nbody, nbody_channels

def set_up_bridge(hydro, nbody, parameters):
    """
    Set up the bridge between the hydro and nbody codes
    More info at ~/amuse/couple/bridge.py
    :hydro:        The SPH code with particles set up
    :nbody:        The N-body code with binary set up
    :parameters:   Dictionary containing the simulation parameters
    """
    # "not the best way of doing it, but it works" ~Steven
    kick_from_hydro = CalculateFieldForParticles(particles=hydro.particles, gravity_constant=constants.G)
    kick_from_hydro.smoothing_length_squared = hydro.parameters.gas_epsilon**2

    # The timestep of the Bridge is a critical parameter and needs to be explored # Adam
    bridge_timestep = parameters["simulation_timestep_day"] | units.day
    coupled_system = Bridge(timestep=bridge_timestep, verbose=False, use_threading=True) # Adam
    # coupled_system = Bridge(timestep=bridge_timestep / 2, verbose=False, use_threading=True) # Nathan

    # Add the systems to the bridge
    coupled_system.add_system(nbody, (kick_from_hydro,), False) # Kick from hydro on nbody
    coupled_system.add_system(hydro, (nbody,), False)           # Kick from nbody on hydro

    return coupled_system

################################################################################
# Simulation
################################################################################

def run_simulation(coupled_system, hydro_channels, nbody_channels, dirs, parameters, t_start=0, i_next=0):
    """
    Simulate mass transfer in the binary orbit
    :coupled_system:   Bridge between the hydro and nbody codes
    :hydro_channels:   Channels of the hydro code (needed?)
    :nbody_channels:   Channels of the nbody code (needed?)
    :dirs:             Dictionary containing the paths to directories
    :parameters:       Dictionary containing the simulation parameters
    :t_start:          Time at the start of the simulation
    :i_next:           Value of the next step
    """
    # Create a sink particle for the companion with sink radius a multiple of the radius
    secondary_radius = parameters["m_secondary_MSun"]**0.8 | units.RSun # MS mass-radius relationship
    sink_radius = parameters["sink_multiplier"] * secondary_radius
    sink_particle = new_sink_particles(coupled_system.particles[0].as_set(), sink_radius=sink_radius)

    # Define the escape radius as a multiple of the initial semi-major axis
    sm_axis = (parameters["periapsis_RSun"] / (1 - parameters["eccentricity"])) | units.RSun
    escape_radius = parameters["escape_multiplier"] * sm_axis

    # Define the simulation times
    t_end = parameters["max_simulation_time_day"]
    timestep = parameters["simulation_timestep_day"]
    times = np.arange(t_start, t_end + timestep, timestep)[1:] | units.day
    t_offset = t_start | units.day

    # Run the simulation up to each defined time value
    for i, time in tqdm(enumerate(times, start=i_next), total=len(times)):

        # Evolve to the next timestep
        coupled_system.evolve_model(time - t_offset)
        coupled_system.particles.move_to_center()

        # Accrete the particles that came close to the sink particle
        sink_particle.accrete(coupled_system.gas_particles)
        
        # Every n steps, locate and remove far-away particles
        if i > 0 and i % parameters["n_steps_escape"] == 0:
            distances = coupled_system.gas_particles.position.lengths()
            escaped_particles = coupled_system.gas_particles[distances > escape_radius]
            if len(escaped_particles) > 0:
                print(f"{i:03}, time={time}: Deleting {len(escaped_particles)} particles")
                coupled_system.gas_particles.remove_particles(escaped_particles)
            else:
                print(f"{i:03}, time={time}: No particles deleted")

        # update the particles FROM the code
        ### ===> SHOULD USE/SAVE THE COPIED PARTICLES INSTEAD OF coupled_system.particles
        ### AS IMPLEMENTED NOW, THIS DOES NOTHING
        hydro_channels["code_to_local_gas"].copy()
        hydro_channels["code_to_local_core"].copy()
        nbody_channels["code_to_local"].copy()

        # Save the current configuration to file
        snapshotfile = os.path.join(dirs["snapshots"], f"simulation_gas_{i:05}_{time.value_in(units.day):.2f}d.amuse")
        write_set_to_file(coupled_system.gas_particles, snapshotfile, format="amuse", overwrite_file = True)
        snapshotfile = os.path.join(dirs["snapshots"], f"simulation_dm_{i:05}_{time.value_in(units.day):.2f}d.amuse")
        write_set_to_file(coupled_system.dm_particles, snapshotfile, format="amuse", overwrite_file = True)

################################################################################
# Main
################################################################################

def C_RunSimulation(sph_code, nbody_code, dirs, parameters):
    """
    Add the companion star to the giant and simulate mass transfer
    :sph_code:     SPH code to be used
    :nbody_code:   N-Body code to be used
    :dirs:         Dictionary containing the paths to directories
    :parameters:   Dictionary containing the simulation parameters
    """
    # Get the relaxed SPH particles file path
    gas_model = [file for file in os.listdir(dirs["models"]) if file.startswith("relaxed_gas")][0]
    gas_file = os.path.join(dirs["models"], gas_model)

    # Get the relaxed core particle file path
    core_model = [file for file in os.listdir(dirs["models"]) if file.startswith("relaxed_core")][0]
    core_file = os.path.join(dirs["models"], core_model)

    # Load the relaxed giant model
    giant_model = load_relaxed_giant(gas_file, core_file)

    # Set up the binary system (positions and velocities)
    giant_model, secondary_particle = set_up_binary_system(giant_model, parameters)

    # Set up hydrodynamics for the SPH particles
    hydro, hydro_channels = set_up_hydrodynamics(sph_code, giant_model, dirs, parameters)

    # Set up N-body dynamics for the secondary star
    nbody, nbody_channels = set_up_dynamics(nbody_code, secondary_particle, dirs, parameters)

    # Set up a Bridge between hydrodynamics and N-body dynamics
    coupled_system = set_up_bridge(hydro, nbody, parameters)
    coupled_system.particles.move_to_center()

    # Run the simulation
    particles = run_simulation(coupled_system, hydro_channels, nbody_channels, dirs, parameters)

    # Clean up afterwards
    hydro.stop()
    nbody.stop()
    coupled_system.stop()

def restart_simulation(sph_code, nbody_code, snapshot_gas, snapshot_dm, dirs, parameters):
    """
    Restart the simulation
    :sph_code:       SPH code to be used
    :nbody_code:     N-Body code to be used
    :snapshot_gas:   Snapshot containing the sph particles
    :snapshot_dm:    Snapshot containing the core and secondary particles
    :dirs:           Dictionary containing the paths to directories
    :parameters:     Dictionary containing the simulation parameters
    """
    # Load the particles from the snapshot files
    giant_model, secondary_particle, time, next_step = load_simulation_snapshot(snapshot_gas, snapshot_dm, dirs)
    if time >= parameters['max_simulation_time_day']:
        print(f"Error: time ({time}) >= max_time ({parameters['max_simulation_time_day']})")
        return

    # Set up hydrodynamics for the SPH particles
    hydro, hydro_channels = set_up_hydrodynamics(sph_code, giant_model, dirs, parameters)

    # Set up N-body dynamics for the secondary star
    nbody, nbody_channels = set_up_dynamics(nbody_code, secondary_particle, dirs, parameters)

    # Set up a Bridge between hydrodynamics and N-body dynamics
    coupled_system = set_up_bridge(hydro, nbody, parameters)
    coupled_system.particles.move_to_center()

    # Run the simulation
    print(f"Restarting simulation from {time} days to {parameters['max_simulation_time_day']} days")
    particles = run_simulation(coupled_system, hydro_channels, nbody_channels, dirs, parameters, t_start=time, i_next=next_step)

    # Clean up afterwards
    hydro.stop()
    nbody.stop()
    coupled_system.stop()