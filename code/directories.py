import os
import json

def load_parameter_json(filepath):
    """ Load a json file """
    with open(filepath, "r") as f:
        parameters = json.load(f)
    return parameters

def save_parameter_json(filepath, parameters):
    """ Save a json file """
    with open(filepath, "w") as f:
        json.dump(parameters, f, indent=4)

def create_if_not_exists(dir):
    """ Create a dictionary if it does not exist """
    if not os.path.exists(dir):
        os.mkdir(dir)

def getPaths(work_dir_name, relaxed_dir, sim_dir="simulations", existing_dir=False):
    """
    Defines the path toward several important directories and creates them if they dont exist.
    :work_dir_name: Name of the working directory
    :relaxed_dir:   Path to the directory containing all relaxed stars
    :sim_dir:       Name of the directory containing the work dir in /home/user/
    :existing_dir:  If the work dir has to exist already or not
    """

    dirs = {}

    # os.getcwd() returns $OUTPUT_FOLDER
    work = os.path.join(os.getcwd(), sim_dir, work_dir_name)
    dirs["work"] = work

    if existing_dir and not os.path.exists(dirs["work"]):
        print(f"Error: directory {dirs['work']} does not exist")
        exit()

    models = os.path.join(work, "models")
    dirs["models"] = models

    plots = os.path.join(work, "plots")
    dirs["plots"] = plots

    data = os.path.join(work, "data")
    dirs["data"] = data

    snapshots = os.path.join(work, "snapshots")
    dirs["snapshots"] = snapshots

    # Creates the directories if they dont exist
    for dir in dirs:
        create_if_not_exists(dirs[dir])

    # Set the relaxed models directory back in home
    dirs["relaxed"] = relaxed_dir

    return dirs