import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation

from tqdm import tqdm

from amuse.units import units, constants
from amuse.datamodel import Particle
from amuse.io import write_set_to_file, read_set_from_file
from amuse.plot import pynbody_column_density_plot, HAS_PYNBODY, scatter
from amuse.ext.star_to_sph import convert_stellar_model_to_sph, StellarModel2SPH

from directories import getPaths, load_parameter_json
from equations import *

FRAME_PRINT = 100 # Print animation update every x frames

############################################################################
### A: Evolution
############################################################################

def animate_HR(HR_data, dirs, parameters):
    """
    Creates an animation of the evolution of the giant on the HR diagram.
    :HR_data:      Collection of data during evolution [T, L, t, R]
    :dirs:         Paths to the relevant directories
    :parameters:   Parameters of the simulation
    """

    temps = HR_data[0].value_in(units.K)
    lums = HR_data[1].value_in(units.LSun)
    times = HR_data[2].value_in(10**6 * units.yr)
    radii = HR_data[3].value_in(units.RSun)

    temp_lim = [0.6*np.min(temps), 1.4*np.max(temps)]
    lum_lim = [0.6*np.min(lums), 1.4*np.max(lums)]

    # Create the initial figure
    fig, ax = plt.subplots(dpi=300)

    HR = ax.plot(temps[0], lums[0])[0]
    point = ax.plot(temps[0], lums[0], marker = "o", color = "orange")[0]
    text = ax.annotate(f"{0}: {int(times[0]):7.1f} Myr  | {radii[0]:6.2f} RSun",
                       xy = (0.05, 0.95), xycoords = "axes fraction")
    def L(R,T):
        return (constants.four_pi_stefan_boltzmann * R**2 * T**4).value_in(units.LSun)

    # Plot the lines of equal radius
    equal_R = [1, 3, 10, 30, 100] | units.RSun
    temp_equal_R = np.linspace(temp_lim, 100)
    for i, R in enumerate(equal_R):
        lum_equal_R = L(R, temp_equal_R | units.K)
        ax.plot(temp_equal_R, lum_equal_R, color = "lightgray", linestyle = "dashed")

    # Plot the line of equal radius for the estimated Roche lobe
    roche_radius = estimate_roche_radius_periapsis_parameters(parameters)
    stop_radius = parameters["roche_multiplier"] * roche_radius
    ax.plot(temp_equal_R, L(stop_radius, temp_equal_R | units.K),
             color = "gray", linestyle = "dashed")
    ax.plot(temp_equal_R, L(roche_radius, temp_equal_R | units.K),
             color = "gray", linestyle = "solid")
    ax.annotate(f"Evolving until R = {stop_radius.value_in(units.RSun):.1f} RSun",
                xy = (0.05, 0.90), xycoords = "axes fraction")

    ax.set(xlim = temp_lim, ylim = lum_lim)
    ax.set(xlabel = f"Temperature [K]", ylabel = f"Luminosity [LSun]")
    ax.set(xscale = "log", yscale = "log")

    ax.invert_xaxis()

    # Define the update function that Update the data and annotation for each frame
    def update(frame):
        HR.set_data(temps[:frame], lums[:frame])
        point.set_data([temps[frame]], [lums[frame]])

        text.set_text(f"{frame}: {int(times[frame]):7.1f} Myr  | {radii[frame]:6.1f} RSun")

    # Create the animation
    N_frames = len(temps)
    gif = animation.FuncAnimation(fig=fig, func=update, frames=N_frames)

    # Save the animation
    print(f"Animating stellar evolution in HR diagram ({N_frames} frames)")
    gif.save(os.path.join(dirs["plots"], "HR_diagram_evolution.gif"), writer = animation.PillowWriter(fps=30))

    plt.close()

############################################################################
### B: Relaxation
############################################################################

# old
def plot_particles_2D(core, gas, parameters, filepath="test_relax_plot2D.png"):

    core_pos = core.position.value_in(units.RSun)
    core_radius = core.radius.value_in(units.RSun)

    roche_radius = estimate_roche_radius_periapsis_parameters(parameters).value_in(units.RSun)
    stop_radius = roche_radius * parameters["roche_multiplier"]
    limit = 2 * roche_radius
    

    plt.figure(figsize=(5,5))
    ax = plt.gca()

    ax.add_patch(plt.Circle(core_pos[:2], core_radius, color="red", fill=False, zorder=10))
    ax.add_patch(plt.Circle(core_pos[:2], roche_radius, color="black", fill=False, zorder=10))
    ax.add_patch(plt.Circle(core_pos[:2], stop_radius, color="black", linestyle = "dashed", fill=False, zorder=10))

    xs = gas.x.value_in(units.RSun)# - gas_com[0]
    ys = gas.y.value_in(units.RSun)# - gas_com[1]
    zs = gas.z.value_in(units.RSun)# - gas_com[2]
    plt.scatter(xs, ys, alpha=0.5, s=1.5, linewidth=0)

    plt.xlabel("X")
    plt.ylabel("Y")

    
    plt.xlim([-limit, limit])
    plt.ylim([-limit, limit])

    plt.savefig(filepath, dpi=300)
    plt.close()

# old
def plot_particles_3D(gas, roche_radius, filepath="test_relax_plot3D.png"):

    roche_radius = roche_radius.value_in(units.RSun)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1,1,1])

    # Draw the SPH particles
    xs = gas.x.value_in(units.RSun)# - gas_com[0]
    ys = gas.y.value_in(units.RSun)# - gas_com[1]
    zs = gas.z.value_in(units.RSun)# - gas_com[2]
    ax.scatter(xs, ys, zs, alpha=0.3, s=2, linewidth=0)

    # Draw Roche sphere mesh
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    x_RL = roche_radius * np.cos(u)*np.sin(v)
    y_RL = roche_radius * np.sin(u)*np.sin(v)
    z_RL = roche_radius * np.cos(v)
    ax.plot_wireframe(x_RL, y_RL, z_RL, color="black", linewidth=0.3, alpha=0.4)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.savefig(filepath, dpi=300)
    plt.close()

def animate_relaxation_2D(core_snapshots, gas_snapshots, dirs, parameters, fps=15, filepath="test_relax_animation2D.gif"):
    """
    Creates an animation of the relaxation in a top-down xy view.
    :core_snapshots:  Snapshots of the core particle
    :gas_snapshots:   Snapshots of the SPH particles
    :dirs:            Paths to the relevant directories
    :parameters:      Parameters of the simulation
    :fps:             Frames per second of animation
    :filepath:        File to save the animation to
    """

    # Calculate Roche and stop radii
    roche_radius = estimate_roche_radius_periapsis_parameters(parameters).value_in(units.RSun)
    stop_radius = roche_radius * parameters["roche_multiplier"]

    # Plot the first frame
    fig, ax = plt.subplots(dpi=300, figsize=(5,5))

    # Plot the SPH particles
    gas_file = os.path.join(dirs["snapshots"], gas_snapshots[0])
    gas = read_set_from_file(gas_file, format="amuse")
    particles = ax.scatter(gas.x.value_in(units.RSun), gas.y.value_in(units.RSun), alpha=0.5, s=1.5, linewidth=0)

    # Draw a circle representing the core
    core_file = os.path.join(dirs["snapshots"], core_snapshots[0])
    core = read_set_from_file(core_file, format="amuse")[0]
    core_radius = core.radius.value_in(units.RSun)
    core_pos = (core.x.value_in(units.RSun), core.y.value_in(units.RSun))
    core_circ = ax.add_patch(plt.Circle(core_pos[:2], core_radius, color="red", fill=False, zorder=10))

    # Draw a circle representing the Roche radius and the stop-evolve radius
    roche_circ = ax.add_patch(plt.Circle(core_pos[:2], roche_radius, color="black", fill=False, zorder=10))
    stop_circ = ax.add_patch(plt.Circle(core_pos[:2], stop_radius, color="black", linestyle = "dashed", fill=False, zorder=10))

    ax.set_xlabel(r"X [R$_\odot$]")
    ax.set_ylabel(r"Y [R$_\odot$]")
    ax.set_title(f"Step {0:05}; Time {parameters['relax_timestep_day'] * 0:.2f}")
    
    limit = 2 * stop_radius
    plt.xlim([-limit, limit])
    plt.ylim([-limit, limit])

    plt.tight_layout()

    # Define the update function that updates the plot
    def update(frame):
        if frame > 0 and frame % FRAME_PRINT == 0:
            print(f"{frame:5}: {frame * parameters['relax_timestep_day']:.2f} days")

        # Load the initial SPH particles
        gas_file = os.path.join(dirs["snapshots"], gas_snapshots[frame])
        gas = read_set_from_file(gas_file, format="amuse")
        gas_pos = np.dstack((gas.x.value_in(units.RSun), gas.y.value_in(units.RSun)))[0]

        # Load the initial core particle
        core_file = os.path.join(dirs["snapshots"], core_snapshots[frame])
        core = read_set_from_file(core_file, format="amuse")[0]
        core_pos = (core.x.value_in(units.RSun), core.y.value_in(units.RSun))

        # Update the particles and circles
        particles.set_offsets(gas_pos)
        core_circ.set(center=core_pos)
        roche_circ.set(center=core_pos)
        stop_circ.set(center=core_pos)

        ax.set_title(f"Step {frame:05}; Time {parameters['relax_timestep_day'] * frame:.2f}")

    # Create the animation
    N_frames = len(core_snapshots)
    gif = animation.FuncAnimation(fig=fig, func=update, frames=N_frames)

    # Save the animation
    print(f"Animating 2D relaxation ({N_frames} frames)")
    gif.save(filepath, writer = animation.PillowWriter(fps=fps))

    plt.close()

# OUTDATED
def animate_relaxation_3D(core_snapshots, gas_snapshots, roche_radius, dirs, filepath="test_relax_animation3D.gif"):
    """
    OUTDATED FUNCTION
    """
    
    # Load the initial SPH particles
    gas_file = os.path.join(dirs["snapshots"], gas_snapshots[0])
    gas = read_set_from_file(gas_file, format="amuse")

    # Load the initial core particle
    core_file = os.path.join(dirs["snapshots"], core_snapshots[0])
    core = read_set_from_file(core_file, format="amuse")[0]
    core_pos = (core.x.value_in(units.RSun), core.y.value_in(units.RSun), core.z.value_in(units.RSun))

    # Plot the first frame
    print(f"plotting initial frame")
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1,1,1])

    # Plot the SPH particles
    gas_x = gas.x.value_in(units.RSun)
    gas_y = gas.y.value_in(units.RSun)
    gas_z = gas.z.value_in(units.RSun)
    particles = ax.scatter(gas_x, gas_y, gas_z, alpha=0.5, s=1, linewidth=0)

    # Draw Roche sphere mesh
    roche_radius = roche_radius.value_in(units.RSun)
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    x_RL = roche_radius * np.cos(u)*np.sin(v)
    y_RL = roche_radius * np.sin(u)*np.sin(v)
    z_RL = roche_radius * np.cos(v)
    sphere = ax.plot_wireframe(x_RL, y_RL, z_RL, color="black", linewidth=0.3, alpha=0.4)

    ax.set_xlabel(r"X [R$_\odot$]")
    ax.set_ylabel(r"Y [R$_\odot$]")
    ax.set_zlabel(r"Z [R$_\odot$]")
    ax.set_title(f"Step {0:05}")
    
    # limit = (1.5*roche_radius | units.RSun).value_in(units.RSun)
    # plt.xlim([-limit, limit])
    # plt.ylim([-limit, limit])

    plt.tight_layout()

    # Define the update function that updates the plot
    def update(frame):
        print(f"plotting frame {frame}")

        # Load the initial SPH particles
        gas_file = os.path.join(dirs["snapshots"], gas_snapshots[frame])
        gas = read_set_from_file(gas_file, format="amuse")
        gas_x = gas.x.value_in(units.RSun)
        gas_y = gas.y.value_in(units.RSun)
        gas_z = gas.z.value_in(units.RSun)

        # Update the particles
        particles._offsets3d = (gas_x, gas_y, gas_z)

        ax.set_title(f"Step {frame:05}")

    # Create the animation
    gif = animation.FuncAnimation(fig=fig, func=update, frames=len(core_snapshots))

    # Save the animation
    gif.save(filepath, writer = animation.PillowWriter(fps=fps))

    plt.close()

############################################################################
### Relaxation energies
############################################################################

def energy_evolution_plot(times, potential, kinetic, thermal, filepath="energy_evolution.png"):

    times = times.value_in(units.day)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10,8))
    ax1, ax2, ax3, ax4 = axes

    ax1.plot(times, kinetic.value_in(units.J), color="blue")
    ax1.set_ylabel("Kinetic [J]")

    ax2.plot(times, potential.value_in(units.J), color="orange")
    ax2.set_ylabel("Potential [J]")

    ax3.plot(times, thermal.value_in(units.J), color="green")
    ax3.set_ylabel("Thermal [J]")

    ax4.plot(times, (kinetic+potential+thermal).value_in(units.J), color="red")
    ax4.set_ylabel("Total [J]")
    ax4.set_xlabel("Time [days]")

    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"created and saved {filepath}")

def potential_energy_plot(time, potential, filepath="potential_energy_evolution.png"):

    time = time.value_in(units.day)

    plt.figure(figsize = (10, 5))

    plt.plot(time, potential.value_in(units.J), color = "orange")

    plt.xlabel("Time [days]")
    plt.ylabel("Potential energy [J]")

    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"created and saved {filepath}")

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

    for i, (snapshot_gas, snapshot_core) in tqdm(enumerate(zip(snapshots_gas, snapshots_core)), total=len(snapshots_core)):

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

def plot_relaxation_energies(dirs, parameters):
    # from B_GiantToSPH import load_snapshot_energies

    print(f"\nLoading previous energy values...")
    times, potential, kinetic, thermal = load_snapshot_energies(dirs, parameters)

    # Create the energy evolution plot
    print("plotting energy evolution")
    filepath = os.path.join(dirs["plots"], f"energy_evolution_{parameter_string}.png")
    energy_evolution_plot(times, potential, kinetic, thermal, filepath=filepath)
    
    # Create the potential energy evolution plot
    print("plotting potential energy evolution")
    filepath = os.path.join(dirs["plots"], f"potential_energy_{parameter_string}.png")
    potential_energy_plot(time_array, E_potential, filepath=filepath)


############################################################################
### C: Simulation
############################################################################

# OUTDATED
def plot_simulation_density(particles, roche_radius, filepath="test_sim_plot_density.png"):
    """
    OUTDATED FUNCTION
    """
    
    sec = particles[0]
    core = particles[1]
    gas = particles[2:]

    # plt.figure()

    vmin = 5e28
    vmax = 2e33
    limit = 10*roche_radius
    test = pynbody_column_density_plot(gas, cmap="inferno", vmin=vmin, vmax=vmax, width=limit)
    # scatter(core.x, core.y, color = "white", edgecolor="black")
    # scatter(sec.x, sec.y, color = "white", edgecolor="black")

    # plt.title(f"Step {i+1:03}: Time = {time.value_in(units.day):.3f} days")

    print(test)


    # plt.savefig(filepath, dpi=300)
    # plt.close()

    # vmin = 1e24
    # vmax = 2e32
    plt.figure("test")
    plt.imshow(test, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap="inferno")
    plt.colorbar()
    pynbody_column_density_plot(gas, cmap="inferno", vmin=vmin, vmax=vmax, width=limit)
    plt.savefig("test.png", dpi=300)
    plt.close()

def animate_simulation_particles(gas_snapshots, dm_snapshots, dirs, parameters, fps=15, filepath="test_sim_animation.gif", do_trail=False):
    """
    Creates an animation of simulation in particles in both top-down xy and side-on xz views.
    :gas_snapshots:  Snapshots of the SPH particles
    :dm_snapshots:   Snapshots of the core and secondary particles
    :dirs:           Paths to the relevant directories
    :parameters:     Parameters of the simulation
    :fps:            Frames per second of animation
    :filepath:       File to save the animation to
    :do_trail:       Boolean to add trails to the stars
    """

    # Load the initial sph particles
    gas_file = os.path.join(dirs["snapshots"], gas_snapshots[0])
    gas = read_set_from_file(gas_file, format="amuse")

    # Load the initial secondary and core particles
    dm_file = os.path.join(dirs["snapshots"], dm_snapshots[0])
    dm_particles = read_set_from_file(dm_file, format="amuse")
    sec = dm_particles[0]
    core = dm_particles[1]

    # If trails are turned on, keep track of the positions
    if do_trail:
        core_xs = [core.x.value_in(units.RSun)]
        core_ys = [core.y.value_in(units.RSun)]
        core_zs = [core.z.value_in(units.RSun)]
        sec_xs = [sec.x.value_in(units.RSun)]
        sec_ys = [sec.y.value_in(units.RSun)]
        sec_zs = [sec.z.value_in(units.RSun)]

    core_pos = core.position.value_in(units.RSun)
    sec_pos = sec.position.value_in(units.RSun)

    core_radius = core.radius.value_in(units.RSun)
    sec_radius = sec.mass.value_in(units.MSun)**0.8 # Main-sequence mass-radius relationship

    gas_x = gas.x.value_in(units.RSun)
    gas_y = gas.y.value_in(units.RSun)
    gas_z = gas.z.value_in(units.RSun)

    roche_radius = estimate_roche_radius_periapsis_parameters(parameters).value_in(units.RSun)
    limit = (1.1 * apoapsis_from_periapsis(parameters["eccentricity"], parameters["periapsis_RSun"] | units.RSun)).value_in(units.RSun)

    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, sharex = True, figsize = (10,5))
    fig.suptitle(f"Step {0:05}; Time {0 * parameters['simulation_timestep_day']:.2f} days")

    ######################
    # Plot top-down view #
    ######################

    particles_xy = ax_xy.scatter(gas_x, gas_y, alpha=1, s=2, linewidth=0, zorder = 2)

    core_circ_xy = ax_xy.add_patch(plt.Circle(core_pos[:2], core_radius, color="red", fill=False, zorder=9))
    roche_circ_xy = ax_xy.add_patch(plt.Circle(core_pos[:2], roche_radius, color="black", fill=False, zorder=9))
    sec_circ_xy = ax_xy.add_patch(plt.Circle(sec_pos[:2], sec_radius, color="black", fill=True, zorder=9))

    if do_trail:
        trail_sec_xy = ax_xy.plot(sec_xs, sec_ys, color="forestgreen", linewidth=0.5, linestyle="solid")[0]
        trail_core_xy = ax_xy.plot(core_xs, core_ys, color="red", linewidth=0.5, linestyle="solid")[0]

    ax_xy.set_xlabel(r"X [$R_\odot$]")
    ax_xy.set_ylabel(r"Y [$R_\odot$]")

    ax_xy.set_xlim([-limit, limit])
    ax_xy.set_ylim([-limit, limit])

    ax_xy.set_aspect("equal")

    #####################
    # Plot side-on view #
    #####################

    particles_xz = ax_xz.scatter(gas_x, gas_z, alpha=1, s=2, linewidth=0, zorder = 2)

    # Make the secondary go behind the giant
    if sec.y > core.y:
        sec_zorder = 1
    else:
        sec_zorder = 10

    core_circ_xz = ax_xz.add_patch(plt.Circle(core_pos[::2], core_radius, color="red", fill=False, zorder=9))
    roche_circ_xz = ax_xz.add_patch(plt.Circle(core_pos[::2], roche_radius, color="black", fill=False, zorder=9))
    sec_circ_xz = ax_xz.add_patch(plt.Circle(sec_pos[::2], sec_radius, color="black", fill=True, zorder=sec_zorder))

    if do_trail:
        trail_sec_xz = ax_xz.plot(sec_xs, sec_zs, color="forestgreen", linewidth=0.5, linestyle="solid")[0]
        trail_core_xz = ax_xz.plot(core_xs, core_zs, color="red", linewidth=0.5, linestyle="solid")[0]

    ax_xz.set_xlabel(r"X [$R_\odot$]")
    ax_xz.set_ylabel(r"Z [$R_\odot$]")

    ax_xz.set_xlim([-limit, limit])
    ax_xz.set_ylim([-limit, limit])

    ax_xz.set_aspect("equal")

    plt.tight_layout()

    # Define the update function that updates the plot
    def update(frame):
        if frame > 0 and frame % FRAME_PRINT == 0:
            print(f"{frame:5}: {frame * parameters['simulation_timestep_day']:.2f} days")

        # Load the next sph particles
        gas_file = os.path.join(dirs["snapshots"], gas_snapshots[frame])
        gas = read_set_from_file(gas_file, format="amuse")

        # Load the next secondary and core particles
        dm_file = os.path.join(dirs["snapshots"], dm_snapshots[frame])
        dm_particles = read_set_from_file(dm_file, format="amuse")
        sec = dm_particles[0]
        core = dm_particles[1]

        # Update the particles and circles of the top-down view
        gas_xy = np.dstack((gas.x.value_in(units.RSun), gas.y.value_in(units.RSun)))[0]
        core_xy = np.array((core.x.value_in(units.RSun), core.y.value_in(units.RSun)))
        sec_xy = np.dstack((sec.x.value_in(units.RSun), sec.y.value_in(units.RSun)))[0][0]

        particles_xy.set_offsets(gas_xy)
        core_circ_xy.set(center=core_xy)
        roche_circ_xy.set(center=core_xy)
        sec_circ_xy.set(center=sec_xy)

        # Update the particles and circles of the side-on view
        gas_xz = np.dstack((gas.x.value_in(units.RSun), gas.z.value_in(units.RSun)))[0]
        core_xz = np.dstack((core.x.value_in(units.RSun), core.z.value_in(units.RSun)))[0][0]
        sec_xz = np.dstack((sec.x.value_in(units.RSun), sec.z.value_in(units.RSun)))[0][0]

        # We be fancy
        if sec.y > core.y:
            sec_zorder = 1
        else:
            sec_zorder = 10

        particles_xz.set_offsets(gas_xz)
        core_circ_xz.set(center=core_xz)
        roche_circ_xz.set(center=core_xz)
        sec_circ_xz.set(center=sec_xz, zorder=sec_zorder)

        if do_trail:
            core_xs.append(core.x.value_in(units.RSun))
            core_ys.append(core.y.value_in(units.RSun))
            core_zs.append(core.z.value_in(units.RSun))
            sec_xs.append(sec.x.value_in(units.RSun))
            sec_ys.append(sec.y.value_in(units.RSun))
            sec_zs.append(sec.z.value_in(units.RSun))

            trail_sec_xy.set_xdata(sec_xs)
            trail_sec_xy.set_ydata(sec_ys)
            trail_core_xy.set_xdata(core_xs)
            trail_core_xy.set_ydata(core_ys)

            trail_sec_xz.set_xdata(sec_xs)
            trail_sec_xz.set_ydata(sec_zs)
            trail_core_xz.set_xdata(core_xs)
            trail_core_xz.set_ydata(core_zs)

        fig.suptitle(f"Step {frame:05}; Time {frame * parameters['simulation_timestep_day']:.2f} days")

    # Create the animation
    N_frames = len(gas_snapshots)
    gif = animation.FuncAnimation(fig=fig, func=update, frames=N_frames)

    # Save the animation
    print(f"Animating simulation ({N_frames} frames; trails={do_trail})")
    gif.save(filepath, writer = animation.PillowWriter(fps=fps))

    plt.close()

def animate_simulation_faraway(gas_snapshots, dm_snapshots, dirs, parameters, fps=15, filepath="test_sim_animation_far.gif"):
    """
    Creates a special version of the particles animation, showing far-away particles and their deletion.
    :gas_snapshots:  Snapshots of the SPH particles
    :dm_snapshots:   Snapshots of the core and secondary particles
    :dirs:            Paths to the relevant directories
    :parameters:      Parameters of the simulation
    :fps:             Frames per second of animation
    :filepath:        File to save the animation to
    """

    # Load the initial sph particles
    gas_file = os.path.join(dirs["snapshots"], gas_snapshots[0])
    gas = read_set_from_file(gas_file, format="amuse")

    # Load the initial secondary and core particles
    dm_file = os.path.join(dirs["snapshots"], dm_snapshots[0])
    dm_particles = read_set_from_file(dm_file, format="amuse")
    sec = dm_particles[0]
    core = dm_particles[1]

    core_pos = core.position.value_in(units.AU)
    sec_pos = sec.position.value_in(units.AU)

    core_radius = core.radius.value_in(units.AU)
    sec_radius = (sec.mass.value_in(units.MSun)**0.8 | units.RSun).value_in(units.AU) # Main-sequence mass-radius relationship

    gas_x = gas.x.value_in(units.AU)
    gas_y = gas.y.value_in(units.AU)

    roche_radius = estimate_roche_radius_periapsis_parameters(parameters).value_in(units.AU)
    sm_axis = smaxis_from_periapsis(parameters["eccentricity"], parameters["periapsis_RSun"] | units.RSun)
    escape = (parameters["escape_multiplier"] * (parameters["periapsis_RSun"]) | units.RSun).value_in(units.AU)
    limit = 2 * escape

    fig, ax = plt.subplots(figsize = (5,5), dpi=300)
    
    # Plot top-down view
    scatter = ax.scatter(gas_x, gas_y, s=2.5, linewidth=1, zorder = 2)

    core_circ = ax.add_patch(plt.Circle(core_pos[:2], core_radius, color="red", fill=False, zorder=9))
    roche_circ = ax.add_patch(plt.Circle(core_pos[:2], roche_radius, color="black", fill=False, zorder=9))
    sec_circ = ax.add_patch(plt.Circle(sec_pos[:2], sec_radius, color="black", fill=True, zorder=9))
    ax.add_patch(plt.Circle([0,0], escape, color="black", fill=False, zorder=9))

    ax.set_title(f"Step {0:05}; Time {0 * parameters['simulation_timestep_day']:.2f} days")
    ax.set_xlabel("X [AU]")
    ax.set_ylabel("Y [AU]")

    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])

    ax.set_aspect("equal")

    plt.tight_layout()

    # Define the update function that updates the plot
    def update(frame):
        if frame > 0 and frame % FRAME_PRINT == 0:
            print(f"{frame:5}: {frame * parameters['simulation_timestep_day']:.2f} days")

        # Load the next sph particles
        gas_file = os.path.join(dirs["snapshots"], gas_snapshots[frame])
        gas = read_set_from_file(gas_file, format="amuse")

        # Load the next secondary and core particles
        dm_file = os.path.join(dirs["snapshots"], dm_snapshots[frame])
        dm_particles = read_set_from_file(dm_file, format="amuse")
        sec = dm_particles[0]
        core = dm_particles[1]

        # Update the particles and circles of the top-down view
        gas_xy = np.dstack((gas.x.value_in(units.AU), gas.y.value_in(units.AU)))[0]
        core_xy = np.array((core.x.value_in(units.AU), core.y.value_in(units.AU)))
        sec_xy = np.dstack((sec.x.value_in(units.AU), sec.y.value_in(units.AU)))[0][0]

        scatter.set_offsets(gas_xy)
        core_circ.set(center=core_xy)
        roche_circ.set(center=core_xy)
        sec_circ.set(center=sec_xy)

        ax.set_title(f"Step {frame:05}; Time {frame * parameters['simulation_timestep_day']:.2f} days")

    # Create the animation
    N_frames = len(gas_snapshots)
    gif = animation.FuncAnimation(fig=fig, func=update, frames=N_frames)

    # Save the animation
    print(f"Animating simulation faraway ({N_frames} frames)")
    gif.save(filepath, writer = animation.PillowWriter(fps=fps))

    plt.close()

############################################################################
### Orbital evolution
############################################################################

def plot_all_orbital_evolutions(snapshots_sim_gas, snapshots_sim_dm, dirs, parameters):
    """
    Collects data from the snapshots and plots several plots of values over time
    :snapshots_sim_gas:  Snapshots of the SPH particles
    :snapshots_sim_dm:   Snapshots of the core and secondary particles
    :dirs:               Paths to the relevant directories
    :parameters:         Parameters of the simulation
    """

    time = [] | units.day
    m_sec = [] | units.MSun
    pos_sec = [] | units.RSun
    vel_sec = [] | units.km/units.s
    m_giant = [] | units.MSun
    pos_giant = [] | units.RSun
    vel_giant = [] | units.km/units.s
    sm_axis = [] | units.RSun
    ecc = []
    periods = [] | units.day

    print(f"\nGathering snapshot data for orbital evolution plots")
    for i, (snapshot_gas, snapshot_dm) in enumerate(zip(snapshots_sim_gas, snapshots_sim_dm)):

        # Load the SPH particles
        gas_file = os.path.join(dirs["snapshots"], snapshot_gas)
        sph_particles = read_set_from_file(gas_file, format="amuse")

        # Load the secondary and core particles
        dm_file = os.path.join(dirs["snapshots"], snapshot_dm)
        dm_particles = read_set_from_file(dm_file, format="amuse")
        secondary_particle = dm_particles[0]
        core_particle = dm_particles[1]

        # Append the positions and velocities
        pos_giant.append(core_particle.position)
        vel_giant.append(core_particle.velocity)
        pos_sec.append(secondary_particle.position)
        vel_sec.append(secondary_particle.velocity)

        # Calculate the time
        time.append(i * (parameters['simulation_timestep_day'] | units.day))

        # Calculate the masses
        Mg = core_particle.mass + np.sum(sph_particles.mass)
        Ms = secondary_particle.mass
        m_giant.append(Mg)
        m_sec.append(Ms)
    	
        # Calculate the semi-major axis (a) and eccentricity (e)
        a, e = orbit_from_particles(sph_particles, dm_particles)
        sm_axis.append(a)
        ecc.append(e)

        # Calculate the orbital period
        p = smaxis_to_period(Mg, Ms, a)
        periods.append(p)

    # Calculate the mass and time differences per timestep
    mass_changes = m_sec[1:] - m_sec[:-1]
    time_changes = time[1:] - time[:-1]

    # Calculate the rate of accretion 
    accretion_rates = (mass_changes / time_changes)
    accretion_times = (time[:-1] + time[1:]) / 2

    # Make the plots
    print("Making plots...")
    plot_evolution_masses(time, m_giant, m_sec, accretion_times, accretion_rates, os.path.join(dirs["plots"], "evolution_masses.png"))
    plot_evolution_orbit(time, sm_axis, ecc, periods, accretion_times, accretion_rates, os.path.join(dirs["plots"], "evolution_orbit.png"))
    plot_evolution_positions(time, pos_giant, pos_sec, os.path.join(dirs["plots"], "evolution_positions.png"))
    plot_evolution_velocities(time, vel_giant, vel_sec, os.path.join(dirs["plots"], "evolution_velocities.png"))

def plot_evolution_masses(time, m_giant, m_sec, Mdot_time, Mdot_rate, filepath):
    """
    Plots the masses of the stars plus accretion rate over time
    :time:      Array of times
    :m_giant:   Array of giant masses
    :m_sec:     Array of secondary masses
    :Mdot_time: Array of times for accretion
    :Mdot_rate: Array of accretion rates
    :filepath:  File to save the plot to
    """
    print("Plotting evolution of the masses")

    fig, axes = plt.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0.03}, figsize=(10,8))
    ax1, ax2, ax3 = axes

    ax1.plot(time.value_in(units.day), m_giant.value_in(units.MSun))
    ax1.set_ylabel(r"Primary mass [M$_\odot$]")

    ax2.plot(time.value_in(units.day), m_sec.value_in(units.MSun))
    ax2.set_ylabel(r"Secondary mass [M$_\odot$]")

    ax3.plot(Mdot_time.value_in(units.day), Mdot_rate.value_in(units.MSun/units.yr))
    ax3.set_ylabel(r"$\dot{M}_{\rm accr}$ [M$_\odot$/year]")
    ax3.set_xlabel("Time [days]")

    for ax in axes:
        ax.grid(linewidth=0.2)

    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_evolution_orbit(time, sm_axis, ecc, periods, Mdot_time, Mdot_rate, filepath):
    """
    Plots the orbital elements plus accretion rate over time
    :time:      Array of times
    :sm_axis:   Array of semi-major axes
    :ecc:       Array of eccentricities
    :periods:   Array of periods
    :Mdot_time: Array of times for accretion
    :Mdot_rate: Array of accretion rates
    :filepath:  File to save the plot to
    """
    print("Plotting evolution of the orbit")

    fig, axes = plt.subplots(4, 1, sharex=True, gridspec_kw={"hspace": 0.03}, figsize=(10,10))
    ax1, ax2, ax3, ax4 = axes

    ax1.plot(time.value_in(units.day), sm_axis.value_in(units.RSun))
    ax1.set_ylabel(r"Semi-major axis [R$_\odot$]")

    ax2.plot(time.value_in(units.day), ecc)
    ax2.set_ylabel("Eccentricity")

    ax3.plot(time.value_in(units.day), periods.value_in(units.day))
    ax3.set_ylabel("Period [days]")

    ax4.plot(Mdot_time.value_in(units.day), Mdot_rate.value_in(units.MSun/units.yr))
    ax4.set_ylabel(r"$\dot{M}_{\rm accr}$ [M$_\odot$/year]")
    ax4.set_xlabel("Time [days]")

    for ax in axes:
        ax.grid(linewidth=0.2)

    plt.savefig(filepath, dpi=300)
    plt.close()
    
def plot_evolution_positions(time, pos_giant, pos_sec, filepath):
    """
    Plots the positions of the stars over time
    :time:      Array of times
    :pos_giant: Array of giant position
    :pos_sec:   Array of secondary position
    :filepath:  File to save the plot to
    """
    print("Plotting evolution of the positions")

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, gridspec_kw={"hspace": 0.03}, figsize=(10,8))
    ax1, ax2, ax3 = axes

    ax1.plot(time.value_in(units.day), pos_giant.x.value_in(units.RSun), color = "blue")
    ax1.plot(time.value_in(units.day), pos_sec.x.value_in(units.RSun), color = "red")
    ax1.set_ylabel(r"X$\rm position$ [R$_\odot$]")

    ax2.plot(time.value_in(units.day), pos_giant.y.value_in(units.RSun), color = "blue")
    ax2.plot(time.value_in(units.day), pos_sec.y.value_in(units.RSun), color = "red")
    ax2.set_ylabel(r"Y$\rm position$ [R$_\odot$]")

    ax3.plot(time.value_in(units.day), pos_giant.z.value_in(units.RSun), color = "blue", label = "giant")
    ax3.plot(time.value_in(units.day), pos_sec.z.value_in(units.RSun), color = "red", label = "companion")
    ax3.set_ylabel(r"Z$\rm position$ [R$_\odot$]")
    ax3.legend()

    for ax in axes:
        ax.grid(linewidth=0.2)
        ax.axhline(0, color="black", linestyle="dashed")

    plt.savefig(filepath, dpi=300)
    plt.close()
    
def plot_evolution_velocities(time, vel_giant, vel_sec, filepath):
    """
    Plots the velocities of the stars over time
    :time:      Array of times
    :vel_giant: Array of giant velocities
    :vel_sec:   Array of secondary velocities
    :filepath:  File to save the plot to
    """
    print("Plotting evolution of the velocities")

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, gridspec_kw={"hspace": 0.03}, figsize=(10,8))
    ax1, ax2, ax3 = axes

    ax1.plot(time.value_in(units.day), vel_giant.x.value_in(units.km/units.s), color = "blue")
    ax1.plot(time.value_in(units.day), vel_sec.x.value_in(units.km/units.s), color = "red")
    ax1.set_ylabel(r"X$\rm velocity$ [R$_\odot$]")

    ax2.plot(time.value_in(units.day), vel_giant.y.value_in(units.km/units.s), color = "blue")
    ax2.plot(time.value_in(units.day), vel_sec.y.value_in(units.km/units.s), color = "red")
    ax2.set_ylabel(r"Y$\rm velocity$ [R$_\odot$]")

    ax3.plot(time.value_in(units.day), vel_giant.z.value_in(units.km/units.s), color = "blue", label = "giant")
    ax3.plot(time.value_in(units.day), vel_sec.z.value_in(units.km/units.s), color = "red", label = "companion")
    ax3.set_ylabel(r"Z$\rm velocity$ [R$_\odot$]")
    ax3.legend()

    for ax in axes:
        ax.grid(linewidth=0.2)
        ax.axhline(0, color="black", linestyle="dashed")

    plt.savefig(filepath, dpi=300)
    plt.close()

############################################################################
### Density profile
############################################################################

def plot_density_profile_residuals(sph_particles, evolved_file, parameters, filepath="test_densityresiduals.png"):

    core_mass = parameters["core_mass_fraction"] * parameters["m_primary_MSun"] | units.MSun
    roche_radius = estimate_roche_radius_periapsis_parameters(parameters)
    stop_radius = roche_radius * parameters["roche_multiplier"]

    # Get the MESA radius and density profiles
    SPH_converter = StellarModel2SPH(
        None,                           # Star particle to be converted to an SPH model
        parameters['n_particles'],      # Number of gas particles in the resulting model
        pickle_file = evolved_file,       # If provided, read stellar structure from here instead of using "particle"
        with_core_particle = True,      # Model the core as a heavy, non-sph particle
        target_core_mass = core_mass,   # If (with_core_particle): target mass for the non-sph particle
        do_store_composition = False    # If set, store the local chemical composition on each particle
    )
    SPH_converter.unpickle_stellar_structure()
    mesa_radii = SPH_converter.radius_profile
    mesa_densities = SPH_converter.density_profile

    # Units to be used
    radius_unit = units.RSun
    weight_unit = units.g
    length_unit = units.cm
    density_unit = weight_unit / length_unit**3

    # Get the mass and distances of the sph particles
    sph_mass = sph_particles.mass[0].value_in(weight_unit)
    sph_distances = sph_particles.position.lengths().value_in(length_unit)

    # Get the points to calculate the particle density for
    steps = 3
    sample_edges = np.insert(mesa_radii[::steps].value_in(length_unit), 0, 0)
    sample_densities = mesa_densities[::steps].value_in(density_unit)

    # Get the particle counts for each bin
    counts, bin_edges = np.histogram(sph_distances, bins=sample_edges)

    # Calculate the bin centre values and their volumes
    bin_centres = ((bin_edges[1:] + bin_edges[:-1]) / 2 | length_unit).value_in(radius_unit)
    bin_volumes = 4 / 3 * np.pi * (np.power(bin_edges[1:], 3) - np.power(bin_edges[:-1], 3))
    
    # Calculate the densities and residuals
    densities = (sph_mass * counts) / bin_volumes
    residuals = densities / sample_densities

    # Create the plot
    gridspec = {"hspace": 0.01, "height_ratios": [7,2]}
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw=gridspec, figsize=(8,7), layout="constrained")

    ax1.plot(mesa_radii.value_in(radius_unit), mesa_densities.value_in(density_unit), color="blue", label="MESA")
    ax1.axvline(stop_radius.value_in(radius_unit), color="black", linestyle = "dashed", linewidth=1, label=f"{parameters['roche_multiplier']}*RL")
    ax1.scatter(bin_centres, densities, s=25, marker="x", color="orange", label="SPH particles")

    ax1.set_ylabel(f"Density [{density_unit}]")

    ax1.set_yscale("log")
    ax1.legend()

    ax2.plot(bin_centres, residuals, color = "orange")
    ax2.axvline(stop_radius.value_in(radius_unit), color="black", linestyle = "dashed", linewidth=1)
    ax2.axhline(1.0, color="black", linestyle="dashed", linewidth=1)

    ax2.set_ylabel("Residuals [SPH/MESA]")
    ax2.set_xlabel(f"Radius [{radius_unit}]")

    ax2.set_yscale("log")

    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_density_profile_particles(sph_particles, evolved_file, parameters, filepath="test_densityprofile.png"):

    core_mass = parameters["core_mass_fraction"] * parameters["m_primary_MSun"] | units.MSun
    roche_radius = estimate_roche_radius_periapsis_parameters(parameters)
    stop_radius = roche_radius * parameters["roche_multiplier"]

    # Get the MESA radius and density profiles
    SPH_converter = StellarModel2SPH(
        None,                           # Star particle to be converted to an SPH model
        parameters['n_particles'],      # Number of gas particles in the resulting model
        pickle_file = evolved_file,       # If provided, read stellar structure from here instead of using "particle"
        with_core_particle = True,      # Model the core as a heavy, non-sph particle
        target_core_mass = core_mass,   # If (with_core_particle): target mass for the non-sph particle
        do_store_composition = False    # If set, store the local chemical composition on each particle
    )
    SPH_converter.unpickle_stellar_structure()
    mesa_radii = SPH_converter.radius_profile
    mesa_densities = SPH_converter.density_profile

    # Units to be used
    radius_unit = units.RSun
    weight_unit = units.g
    length_unit = units.cm
    density_unit = weight_unit / length_unit**3

    # Get the mass and distances of the sph particles
    sph_mass = sph_particles.mass[0].value_in(weight_unit)
    sph_distances = sph_particles.position.lengths().value_in(length_unit)

    # Get the particle counts for each bin
    counts, bin_edges = np.histogram(sph_distances, bins=100)

    # Calculate the bin centre values and their volumes
    bin_centres = ((bin_edges[1:] + bin_edges[:-1]) / 2 | length_unit).value_in(radius_unit)
    bin_volumes = 4 / 3 * np.pi * (np.power(bin_edges[1:], 3) - np.power(bin_edges[:-1], 3))
    
    # Calculate the densities and residuals
    densities = (sph_mass * counts) / bin_volumes

    # Create the plot
    plt.figure()

    plt.plot(mesa_radii.value_in(radius_unit), mesa_densities.value_in(density_unit), color="blue", label="MESA")
    plt.axvline(stop_radius.value_in(radius_unit), color="black", linestyle = "dashed", linewidth=1, label=f"{parameters['roche_multiplier']}*RL")
    plt.scatter(bin_centres, densities, s=25, marker="x", color="orange", label="SPH particles")

    plt.xlabel(f"Distance [{radius_unit}]")
    plt.ylabel(f"Density [{density_unit}]")
    plt.yscale("log")
    plt.legend()

    plt.savefig(filepath, dpi=300)
    plt.close()

def animate_density_profile(snapshots_sph, dirs, parameters, fps=15, filepath="test_animated_densityprofile.gif"):

    # Units to be used
    radius_unit = units.RSun
    weight_unit = units.g
    length_unit = units.cm
    density_unit = weight_unit / length_unit**3
    
    core_mass = parameters["core_mass_fraction"] * parameters["m_primary_MSun"] | units.MSun
    roche_radius = estimate_roche_radius_periapsis_parameters(parameters)
    stop_radius = (roche_radius * parameters["roche_multiplier"]).value_in(radius_unit)

    # Get the MESA radius and density profiles
    evolved_model = [file for file in os.listdir(dirs["models"]) if file.startswith("evolved")][0]
    evolved_file = os.path.join(dirs["models"], evolved_model)
    SPH_converter = StellarModel2SPH(
        None,                           # Star particle to be converted to an SPH model
        parameters['n_particles'],      # Number of gas particles in the resulting model
        pickle_file = evolved_file,       # If provided, read stellar structure from here instead of using "particle"
        with_core_particle = True,      # Model the core as a heavy, non-sph particle
        target_core_mass = core_mass,   # If (with_core_particle): target mass for the non-sph particle
        do_store_composition = False    # If set, store the local chemical composition on each particle
    )
    SPH_converter.unpickle_stellar_structure()
    mesa_radii = SPH_converter.radius_profile
    mesa_densities = SPH_converter.density_profile

    # Load the initial sph particles
    sph_file = os.path.join(dirs["snapshots"], snapshots_sph[0])
    sph_particles = read_set_from_file(sph_file, format="amuse")
    sph_particles.move_to_center()

    def calculate_densities(sph_particles, weight_unit, length_unit, radius_unit):

        # Get the mass and distances of the sph particles
        sph_mass = sph_particles.mass[0].value_in(weight_unit)
        sph_distances = sph_particles.position.lengths().value_in(length_unit)

        # Get the particle counts for each bin
        counts, bin_edges = np.histogram(sph_distances, bins=100)

        # Calculate the bin centre values and their volumes
        bin_centres = ((bin_edges[1:] + bin_edges[:-1]) / 2 | length_unit).value_in(radius_unit)
        bin_volumes = 4 / 3 * np.pi * (np.power(bin_edges[1:], 3) - np.power(bin_edges[:-1], 3))
        
        # Calculate the densities
        densities = (sph_mass * counts) / bin_volumes

        return densities, bin_centres

    # Calculate the densities
    densities, radii = calculate_densities(sph_particles, weight_unit, length_unit, radius_unit)

    # Plot the first frame
    fig, ax = plt.subplots(dpi=300, figsize=(8,5))#, layout="constrained")

    ax.plot(mesa_radii.value_in(radius_unit), mesa_densities.value_in(density_unit), color="blue", label="MESA")
    ax.axvline(stop_radius, color="black", linestyle = "dashed", linewidth=1, label=f"{parameters['roche_multiplier']}*RL")
    density_profile = ax.scatter(radii, densities, s=25, marker="x", color="orange", label="SPH particles")

    ax.set_title(f"Step {0:05}; Time {0:.2f} days")
    ax.set_xlabel(f"Distance [{radius_unit}]")
    ax.set_ylabel(f"Density [{density_unit}]")
    ax.set_yscale("log")

    ax.legend()
    ax.set_xlim(left = 0, right = 1.75 * stop_radius)

    def update(frame):
        if frame > 0 and frame % FRAME_PRINT == 0:
            print(f"{frame:5}: {frame * parameters['relax_timestep_day']:.2f} days")

        # Load the sph particles
        sph_file = os.path.join(dirs["snapshots"], snapshots_sph[frame])
        sph_particles = read_set_from_file(sph_file, format="amuse")
        sph_particles.move_to_center()

        # Calculate the densities
        densities, radii = calculate_densities(sph_particles, weight_unit, length_unit, radius_unit)
        profile = np.dstack((radii, densities))[0]

        # Update the particles and circles
        density_profile.set_offsets(profile)

        ax.set_title(f"Step {frame:05}; Time {parameters['relax_timestep_day'] * frame:.2f}")
        ax.relim()

    # Create the animation
    N_frames = len(snapshots_sph)
    gif = animation.FuncAnimation(fig=fig, func=update, frames=N_frames)

    # Save the animation
    print(f"Animating relaxation density profile ({N_frames} frames)")
    gif.save(filepath, writer = animation.PillowWriter(fps=fps))

    plt.close()

############################################################################

def post_relaxation_visualisations(dirs, parameters):
    """ Plots for after relaxation """
    snapshots = sorted(os.listdir(dirs["snapshots"]))
    snapshots_core = [f for f in snapshots if f.startswith("relaxation_core")]
    snapshots_gas = [f for f in snapshots if f.startswith("relaxation_gas")]

    # Plots relaxation
    animate_relaxation_2D(snapshots_core, snapshots_gas, dirs, parameters, filepath=os.path.join(dirs["plots"], "animation_relaxation_2D.gif"))
    animate_density_profile(snapshots_gas, dirs, parameters, filepath=os.path.join(dirs["plots"], "animation_density_profile.gif"))


def post_simulation_visualisations(dirs, parameters):
    """ Plots for after simulation """
    snapshots = sorted(os.listdir(dirs["snapshots"]))
    snapshots_sim_gas = [f for f in snapshots if f.startswith("simulation_gas")]
    snapshots_sim_dm = [f for f in snapshots if f.startswith("simulation_dm")]
    
    # Plots simulation
    plot_all_orbital_evolutions(snapshots_sim_gas, snapshots_sim_dm, dirs, parameters)
    animate_simulation_particles(snapshots_sim_gas, snapshots_sim_dm, dirs, parameters, filepath=os.path.join(dirs["plots"], "animation_simulation_particles.gif"))
    animate_simulation_particles(snapshots_sim_gas, snapshots_sim_dm, dirs, parameters, filepath=os.path.join(dirs["plots"], "animation_simulation_particles_trail.gif"), do_trail=True)
    animate_simulation_faraway(snapshots_sim_gas, snapshots_sim_dm, dirs, parameters, filepath=os.path.join(dirs["plots"], "animation_simulation_particles_far.gif"))

############################################################################

if __name__ == "__main__":

    os.chdir("/home/11293284")

    # for working_directory in os.listdir("completed_simulations"):
    working_directory = "Mp1.5_Ms1.4_e0.0_a19.472_N250k_RL1.1_predefined-RL"
    print(f"Making plots for {working_directory}")

    dirs = getPaths(working_directory, "")
    # dirs = getPaths(working_directory, sim_dir = "completed_simulations")
    
    print("Loading parameters")
    parameter_json = os.path.join(dirs["data"], "parameters.json")
    parameters = load_parameter_json(parameter_json)

    print("Loading snapshots")
    snapshots = sorted(os.listdir(dirs["snapshots"]))
    snapshots_core = [f for f in snapshots if f.startswith("relaxation_core")]
    snapshots_gas = [f for f in snapshots if f.startswith("relaxation_gas")]
    snapshots_sim_gas = [f for f in snapshots if f.startswith("simulation_gas")]
    snapshots_sim_dm = [f for f in snapshots if f.startswith("simulation_dm")]
    
    ##################
    ### RELAXATION ###
    ##################

    if len(snapshots_gas) > 0:
        print("Plotting relaxation plots")
        plot_relaxation_energies(dirs, parameters)
        animate_relaxation_2D(snapshots_core, snapshots_gas, dirs, parameters, filepath=os.path.join(dirs["plots"], "animation_relaxation_2D.gif"))
        # animate_density_profile(snapshots_gas, dirs, parameters, filepath=os.path.join(dirs["plots"], "animation_density_profile.gif"))
        
    
    ##################
    ### SIMULATION ###
    ##################

    if len(snapshots_sim_gas) > 0:
        print("Plotting simulation plots")
        plot_all_orbital_evolutions(snapshots_sim_gas, snapshots_sim_dm, dirs, parameters)
        animate_simulation_particles(snapshots_sim_gas, snapshots_sim_dm, dirs, parameters, filepath=os.path.join(dirs["plots"], "animation_simulation_particles.gif"))
        animate_simulation_particles(snapshots_sim_gas, snapshots_sim_dm, dirs, parameters, filepath=os.path.join(dirs["plots"], "animation_simulation_particles_trail.gif"), do_trail=True)
        animate_simulation_faraway(snapshots_sim_gas, snapshots_sim_dm, dirs, parameters, filepath=os.path.join(dirs["plots"], "animation_simulation_particles_far.gif"))