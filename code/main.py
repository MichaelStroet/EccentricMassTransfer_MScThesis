import os
import json

from amuse.community.mesa.interface import MESA
from amuse.community.gadget2.interface import Gadget2
from amuse.community.huayno.interface import Huayno

from directories import getPaths, load_parameter_json, save_parameter_json
from equations import *
from visualisations import *

from A_EvolveGiant import A_EvolveGiant
from B_GiantToSPH import B_GiantToSPH, restart_relaxation, get_relaxed_parameter_string
from C_RunSimulation import C_RunSimulation, restart_simulation


def has_files_startswith(dir, start_string):
    """ Check if dir has files starting with <start_string> """
    for file in os.listdir(dir):
        if file.startswith(start_string):
            return True
    return False

def get_working_directory_name(parameters):
    """ Define work directory from parameters """
    working_directory = f"Mp{parameters['m_primary_MSun']}"
    working_directory += f"_Ms{parameters['m_secondary_MSun']}"
    working_directory += f"_D{parameters['periapsis_RSun']}"
    working_directory += f"_e{parameters['eccentricity']}"
    working_directory += f"_N{int(parameters['n_particles']/1000)}k"
    # working_directory += f"_RL{parameters['roche_multiplier']}"

    return working_directory


if __name__ == "__main__":

    """
    n_workers guide (May vary a lot with different parameters, code_units and whatever else)
    Using more risks MPI_ABORT or similar weird issues
    Values found using M=1.5+1.4, a=20, e=0.25, RL=1.1
    10k   16
    50k   16-20
    100k  24
    200k  30-32
    250k  36-40
    500k  60-63
    1M    80-90
    3M    120+
    """

    ############################################################################
    # Define working directory and parameters
    ############################################################################

    # Directory containing all relaxed models
    relaxed_dir = "/home/11293284/relaxed_stars"

    # Give the name of the work dir if restarting a run
    # existing_work_dir = "Mp1.5_Ms1.4_e0.0_D15_N50k_RL1.1_TESTING"
    existing_work_dir = "lolno"

    # Check if the above work dir exists or not
    work_path = os.path.join("/home", "11293284", "simulations", existing_work_dir)
    if not os.path.exists(work_path):
        print(f"\nDid not find {work_path}")

        # Define new parameters
        parameters = {
            "m_primary_MSun": 1.5,              # Mass of the primary star in MSun
            "m_secondary_MSun": 1.4,            # Mass of the secondary star in MSun
            "eccentricity": 0.1,                # Eccentricity of the binary orbit
            "periapsis_RSun": 15,               # Distance at closest approach in RSun
            "roche_multiplier": 1.1,            # Multiplier of the roche radius to evolve to
            "core_mass_fraction": 0.255,        # Fraction of the primary mass for the core particle
            "n_particles": 20000,              # Total number of SPH particles
            "n_workers": 10,                    # Number of workers (processes) to run hydro
            "min_relax_time_day": 20,           # Minimum time to relax in days
            "max_relax_time_day": 1,          # Maximum time to relax in days
            "relax_timestep_day": 0.1,          # Timestep of relaxation snapshots in days
            "relaxed_ratio": 0.015,             # Ratio of initial and current 'oscillation size' to call the star relaxed
            "batch": 25,                        # How many timesteps to calculate the 'oscillation size' with
            "max_simulation_time_day": 1,       # End time of the simulation in days
            "simulation_timestep_day": 0.1,     # Timestep of simulation snapshots in days
            "sink_multiplier": 2,               # Radius multiplier of the sink particle (*R_secondary)
            "escape_multiplier": 10,            # Radius multiplier of removing escaped particles (*SM-axis)
            "n_steps_escape": 25,               # Check for escaped particles every N timesteps
            "code_mass_unit": 1,                # The mass in MSun to set as 1 in code units
            "code_length_unit": 100,            # The length in RSun to set as 1 in code units
            "code_time_unit": 1                 # The time in days to set as 1 in code units
        }
        print(f"New parameters:\n{json.dumps(parameters, indent=4)}\n")

        # Create the working directory and get paths to sub dirs
        working_directory = get_working_directory_name(parameters)
        dirs = getPaths(working_directory, relaxed_dir)

        # Save the parameters as a json file
        json_file = os.path.join(dirs["data"], "parameters.json")
        save_parameter_json(json_file, parameters)

        # Check if this giant has been relaxed before
        # if so, copy the relaxed model to the model directory
        relaxed_string = get_relaxed_parameter_string(parameters)
        relaxed_gas = os.path.join(dirs["relaxed"], f"relaxed_gas_{relaxed_string}.amuse")
        relaxed_core = os.path.join(dirs["relaxed"], f"relaxed_core_{relaxed_string}.amuse")
        if os.path.exists(relaxed_gas) and os.path.exists(relaxed_gas):
            print(f"Star {relaxed_string} has been relaxed before\nCopying relaxed model to node...")
            os.system(f"touch {os.path.join(dirs['models'], 'evolved_fake_file.empty')}") # To skip evolution stage
            os.system(f"rsync -a {relaxed_gas} {dirs['models']}")
            os.system(f"rsync -a {relaxed_core} {dirs['models']}")
            print("Done\n")

    # Otherwise, restart the run in the working directory
    else:  
        print(f"\nFound {work_path}\nLoading parameters from json file")

        # load parameters from file
        parameters = load_parameter_json(os.path.join(work_path, "data", "parameters.json"))
        print(f"{json.dumps(parameters, indent=4)}\n")

        # Copy the working directory to the node
        print(f"\nCopying {existing_work_dir} to node...")
        os.system(f"rsync -a {work_path} $OUTPUT_FOLDER/simulations")
        print("Done\n")

        # Get paths to sub dirs
        dirs = getPaths(existing_work_dir, relaxed_dir)

    print(f"Working directory: {dirs['work']}")

    # Define the codes to be used
    evolution_code = MESA(version='2208')
    sph_code = Gadget2
    nbody_code = Huayno

    ############################################################################
    # Evolve the primary star until Rochelobe overflow
    ############################################################################


    has_evolved = has_files_startswith(dirs["models"], "evolved")
    print(f"\n--------------------\nHas evolved?\n{has_evolved}\n--------------------\n")
    if not has_evolved:
        A_EvolveGiant(evolution_code, dirs, parameters)
    

    ############################################################################
    # Convert the evolved model to SPH particles and relax
    ############################################################################


    has_relaxed = has_files_startswith(dirs["models"], "relaxed")
    print(f"\n--------------------\nHas relaxed?\n{has_relaxed}\n--------------------\n")

    # Skip if a relaxed model exists
    if not has_relaxed:

        # Check if there are previous snapshots
        has_snapshots = has_files_startswith(dirs["snapshots"], "relaxation")
        print(f"    Any previous relaxation snapshots?\n    {has_snapshots}\n")

        # Start relaxing from scratch
        if not has_snapshots:
            print(f"Relaxing giant of {parameters['n_particles']} particles using {parameters['n_workers']} workers:\n")
            B_GiantToSPH(sph_code, dirs, parameters)

        # Restart relaxation from snapshots
        else:
            snapshots = sorted(os.listdir(dirs["snapshots"]))
            snapshots_relax_gas = [f for f in snapshots if f.startswith("relaxation_gas")]
            snapshots_relax_core = [f for f in snapshots if f.startswith("relaxation_core")]

            restart_index = -1 # -1 restarts from last snapshot
            # parameters["max_relax_time_day"] = 20
            restart_gas = snapshots_relax_gas[restart_index]
            restart_core = snapshots_relax_core[restart_index]

            restart_relaxation(sph_code, restart_gas, restart_core, dirs, parameters)

        # Create plots of the relaxation
        post_relaxation_visualisations(dirs, parameters)


    ############################################################################
    # Add the companion star and run simulation
    ############################################################################
    

    has_simulated = has_files_startswith(dirs["snapshots"], "simulation")
    print(f"\n--------------------\nHas simulation snapshots?\n{has_simulated}\n--------------------\n")

    if not has_simulated:
        print(f"Running simulation for {parameters['max_simulation_time_day']} days using {parameters['n_workers']} workers:\n")
        C_RunSimulation(sph_code, nbody_code, dirs, parameters)

        # Create plots of the simulation
        post_simulation_visualisations(dirs, parameters)

    else:
        snapshots = sorted(os.listdir(dirs["snapshots"]))
        snapshots_sim_gas = [f for f in snapshots if f.startswith("simulation_gas")]
        snapshots_sim_dm = [f for f in snapshots if f.startswith("simulation_dm")]

        restart_index = -1 # -1 restarts from last snapshot
        # parameters["max_simulation_time_day"] = 10
        restart_gas = snapshots_sim_gas[restart_index]
        restart_dm = snapshots_sim_dm[restart_index]

        restart_simulation(sph_code, nbody_code, restart_gas, restart_dm, dirs, parameters)

        # Create plots of the simulation
        post_simulation_visualisations(dirs, parameters)
    