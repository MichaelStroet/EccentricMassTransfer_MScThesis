import os
import numpy as np
from tqdm import tqdm

from amuse.io import read_set_from_file, write_set_to_file
from amuse.ext.star_to_sph import convert_stellar_model_to_sph

from amuse.units import units
from amuse.units.generic_unit_converter import ConvertBetweenGenericAndSiUnits

from equations import *
from visualisations import *

################################################################################
# In/Output
################################################################################

def convert_to_sph(evolved_file, parameters):
    """
    Load the evolved model and convert to SPH particles + massive core particle
    More info at ~/amuse/ext/star_to_sph.py
    :evolved_file:  Path to the evolved stellar structure model file
    :parameters:    Dictionary containing the simulation parameters
    """
    print(f"Converting {evolved_file} to {parameters['n_particles']} SPH particles")
    core_mass = parameters["core_mass_fraction"] * parameters["m_primary_MSun"] | units.MSun
    print(f"with a {core_mass.in_(units.MSun)} core particle\n")

    giant_in_sph = convert_stellar_model_to_sph(
        None,                           # Star particle to be converted to an SPH model
        parameters['n_particles'],      # Number of gas particles in the resulting model
        pickle_file = evolved_file,       # If provided, read stellar structure from here instead of using "particle"
        with_core_particle = True,      # Model the core as a heavy, non-sph particle
        target_core_mass = core_mass,   # If (with_core_particle): target mass for the non-sph particle
        do_store_composition = False    # If set, store the local chemical composition on each particle
    )

    sph_particles = giant_in_sph.gas_particles
    core_particle = giant_in_sph.core_particle
    
    return sph_particles, core_particle

def load_snapshot_energies(dirs, parameters):
    """EXTREMELY SLOW FOR MANY PARTICLES (needs parallelisation)"""
    snapshots = sorted(os.listdir(dirs["snapshots"]))
    snapshots_gas = [f for f in snapshots if f.startswith("relaxation_gas")]
    snapshots_core = [f for f in snapshots if f.startswith("relaxation_core")]

    # Create arrays to keep track of time and energies
    time_array = [] | units.day
    E_potential = [] | units.J
    E_kinetic = [] | units.J
    E_thermal = [] | units.J

    print("Loading snapshot energies")
    for i, (snapshot_gas, snapshot_core) in tqdm(enumerate(zip(snapshots_gas, snapshots_core))):
        # Load the particles
        sph_plus_core = read_set_from_file(os.path.join(dirs["snapshots"], snapshot_gas), format="amuse")
        core = read_set_from_file(os.path.join(dirs["snapshots"], snapshot_core), format="amuse")
        sph_plus_core.add_particle(core)

        # Append snapshot energy values
        time_array.append(i * parameters["relax_timestep_day"] | units.day)
        E_potential.append(sph_plus_core.potential_energy())
        E_kinetic.append(sph_plus_core.kinetic_energy())
        E_thermal.append(sph_plus_core.thermal_energy())

    return time_array, E_potential, E_kinetic, E_thermal

def get_relaxed_parameter_string(parameters):
    """ Define parameter string for saving relaxed models """
    string = f"Mp{parameters['m_primary_MSun']}"
    string += f"_Ms{parameters['m_secondary_MSun']}"
    string += f"_D{parameters['periapsis_RSun']}"
    string += f"_N{int(parameters['n_particles']/1000)}k"
    return string

def save_relaxed_model(hydro, dirs, parameters):
    """
    Saves the relaxed giant to file in both the model and relaxed directories
    :hydro:       SPH code containing the relaxed giant
    :dirs:        Dictionary containing the paths to directories
    :parameters:  Dictionary containing the simulation parameters
    """
    parameter_string = get_relaxed_parameter_string(parameters)

    # Save files to the model directory
    snapshot_gas = os.path.join(dirs["models"],f"relaxed_gas_{parameter_string}.amuse")
    snapshot_core = os.path.join(dirs["models"],f"relaxed_core_{parameter_string}.amuse")
    write_set_to_file(hydro.gas_particles, snapshot_gas, format="amuse", overwrite_file=True)
    write_set_to_file(hydro.dm_particles, snapshot_core, format="amuse", overwrite_file=True)

    # Save files to the relaxed directory
    snapshot_gas = os.path.join(dirs["relaxed"],f"relaxed_gas_{parameter_string}.amuse")
    snapshot_core = os.path.join(dirs["relaxed"],f"relaxed_core_{parameter_string}.amuse")
    write_set_to_file(hydro.gas_particles, snapshot_gas, format="amuse", overwrite_file=True)
    write_set_to_file(hydro.dm_particles, snapshot_core, format="amuse", overwrite_file=True)

################################################################################
# Relaxation
################################################################################

def set_up_hydrodynamics(sph_code, sph_particles, core_particle, dirs, parameters):
    """
    Sets up the sph code for running the relaxation
    :sph_code:       SPH code to be used
    :sph_particles:  SPH particles representing the gas
    :core_particle:  Particle representing the core
    :dirs:           Dictionary containing the paths to directories
    :parameters:     Dictionary containing the simulation parameters
    """

    # Set up hydrodynamics code
    code_mass_unit = parameters["code_mass_unit"] | units.MSun
    code_length_unit = parameters["code_length_unit"] | units.RSun
    code_time_unit = parameters["code_time_unit"] | units.day
    unit_converter = ConvertBetweenGenericAndSiUnits(code_mass_unit, code_length_unit, code_time_unit)
    hydro_options = dict(number_of_workers=parameters["n_workers"],
                         redirection="file", # Save log to file <redirect_file>
                         redirect_file=os.path.join(dirs["data"], "sph_code_relax_out.log"))
    hydro = sph_code(unit_converter, **hydro_options)

    # Set additional hydro parameters
    hydro.parameters.epsilon_squared = (sph_particles.h_smooth)**2 # Use SPH smoothing length as epsilon

    # Time settings such that code never ends prematurely
    hydro.parameters.time_max = 2*parameters["max_relax_time_day"] | units.day   # Maximum in-code time.
    hydro.parameters.time_limit_cpu = 1 | units.yr                               # Maximum real time

    # Add particles to hydro code (AFTER SETTING THE OPTIONS!)
    hydro.gas_particles.add_particles(sph_particles)
    hydro.dm_particles.add_particle(core_particle)
    hydro.particles.move_to_center()

    return hydro

def relax_giant(hydro, dirs, parameters, t_start=0, i_next=0):
    """
    Relax the SPH particles representing a giant star.
    :hydro:          SPH code set up for running relaxation
    :dirs:           Dictionary containing the paths to directories
    :parameters:     Dictionary containing the simulation parameters
    :t_start:        Time at the start of the relaxation
    :i_next:         Value of the next step
    """

    # Create energy arrays or load them from file
    energy_file = os.path.join(dirs["data"], "relaxation_energies.txt")
    if t_start == 0:

        # Create a file to store the relaxation energies
        with open(energy_file, "w") as file:
            file.write("# Time [days] - Potential energy [J] - Kinetic energy [J] - Thermal energy [J]\n")
        
        # Create arrays to keep track of time and energies
        time_array = [] | units.day
        E_potential = [] | units.J
        E_kinetic = [] | units.J
        E_thermal = [] | units.J
    
    else:
        # Load the energies and times from file
        contents = np.loadtxt(energy_file, unpack=True)
        time_array = contents[0] | units.day
        E_potential = contents[1] | units.J
        E_kinetic = contents[2] | units.J
        E_thermal = contents[3] | units.J
        
    # Parameters for the self-determined relaxation
    min_relax_time = parameters["min_relax_time_day"] | units.day
    batch_size = parameters["batch"]

    # Define the relaxation times
    t_end = parameters["max_relax_time_day"]
    timestep = parameters["relax_timestep_day"]
    times = np.arange(t_start, t_end + timestep, timestep)[1:] | units.day
    t_offset = t_start | units.day

    # Run relaxation until relaxed or max time reached
    for i, time in tqdm(enumerate(times, start=i_next), total=len(times)):

        # Evolve the model until t=time
        hydro.evolve_model(time - t_offset)

        # Move the particles to the center (the core)
        hydro.particles.position -= hydro.dm_particles[0].position
        hydro.particles.velocity -= hydro.dm_particles[0].velocity

        # Append current values
        time_array.append(time)
        E_potential.append(hydro.potential_energy)
        E_kinetic.append(hydro.kinetic_energy)
        E_thermal.append(hydro.thermal_energy)

        # Save current values to file
        with open(energy_file, "a") as file:
            line = np.dstack((time.value_in(units.day),
                            E_potential[-1].value_in(units.J),
                            E_kinetic[-1].value_in(units.J),
                            E_thermal[-1].value_in(units.J)
                            ))[0]
            np.savetxt(file, line)

        # Save the current configuration to file
        snapshotfile = os.path.join(dirs["snapshots"], f"relaxation_gas_{i:05}_{time.value_in(units.day):.2f}d.amuse")
        write_set_to_file(hydro.gas_particles, snapshotfile, format="amuse", overwrite_file = True)
        snapshotfile = os.path.join(dirs["snapshots"], f"relaxation_core_{i:05}_{time.value_in(units.day):.2f}d.amuse")
        write_set_to_file(hydro.dm_particles, snapshotfile, format="amuse", overwrite_file = True)
        
        # Compare standard deviation between the start and now
        if time > min_relax_time and i % parameters["batch"] == 0:
            first_std = np.std(E_potential[:parameters["batch"]])
            current_std = np.std(E_potential[i-parameters["batch"]:i])
            current_ratio = current_std / first_std

            # Determine if star has relaxed
            if current_ratio < parameters["relaxed_ratio"]:
                print(f"{i:03}, time={time}: RELAXED! :D ({current_ratio:.3f})")
                break
            else:
                print(f"{i:03}, time={time}: Not yet relaxed. ({current_ratio:.3f})")

    # Save the final relaxed giant to files
    save_relaxed_model(hydro, dirs, parameters)

    # Create the energy evolution plot
    N_particles_time_string = f"N{int(parameters['n_particles']/1000)}k_t{time_array[-1].value_in(units.day):.2f}"
    filepath = os.path.join(dirs["plots"], f"energy_evolution_{N_particles_time_string}.png")
    energy_evolution_plot(time_array, E_potential, E_kinetic, E_thermal, filepath=filepath)
    
    # Create the potential energy evolution plot
    filepath = os.path.join(dirs["plots"], f"potential_energy_{N_particles_time_string}.png")
    potential_energy_plot(time_array, E_potential, filepath=filepath)

    return hydro


################################################################################
# Main
################################################################################

def B_GiantToSPH(sph_code, dirs, parameters):
    """
    Convert the evolved model to SPH particles and relax the giant
    :sph_code:       SPH code to be used
    :dirs:           Dictionary containing the paths to directories
    :parameters:     Dictionary containing the simulation parameters
    """

    # Convert evolved model to SPH particles
    evolved_model = [file for file in os.listdir(dirs["models"]) if file.startswith("evolved")][0]
    evolved_file = os.path.join(dirs["models"], evolved_model)
    sph_particles, core_particle = convert_to_sph(evolved_file, parameters)

    # Set up hydrodynamics for the SPH particles
    hydro = set_up_hydrodynamics(sph_code, sph_particles, core_particle, dirs, parameters)

    # Relax the giant
    hydro = relax_giant(hydro, dirs, parameters)

    # Clean up hydro code
    hydro.stop()

def restart_relaxation(sph_code, snapshot_gas, snapshot_core, dirs, parameters):
    """
    Restart the relaxation of the giant
    :sph_code:       SPH code to be used
    :snapshot_gas:   Snapshot containing the sph particles
    :snapshot_core:  Snapshot containing the core particle
    :dirs:           Dictionary containing the paths to directories
    :parameters:     Dictionary containing the simulation parameters
    """
    # Get the snapshot step and time
    split = snapshot_gas[:-7].split("_")
    step = int(split[-2])
    time = float(split[-1])
    
    # Load the snapshot particles
    print(f"Restarting relaxation from snapshots:\n{snapshot_gas}\n{snapshot_core}\n\nStep: {step}\nTime: {time} days\n")
    sph_particles = read_set_from_file(os.path.join(dirs["snapshots"], snapshot_gas), format="amuse")
    core_particle = read_set_from_file(os.path.join(dirs["snapshots"], snapshot_core), format="amuse")[0]

    # Set up hydrodynamics for the SPH particles
    hydro = set_up_hydrodynamics(sph_code, sph_particles, core_particle, dirs, parameters)

    # Relax the giant
    print(f"Relaxing the giant from {time} days onward")
    hydro = relax_giant(hydro, dirs, parameters, t_start=time, i_next=step+1)

    # Clean up hydro code
    hydro.stop()