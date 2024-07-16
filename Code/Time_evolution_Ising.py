# Simulation of the Ising model on Networks
#
# To run the following code you need to create the desire network as the 
# atoms_networks object, all the variables of the simulation are initalized below.
# At the end of the simulation graphs with the rieslts are produced.

import networkx as nx
from networkx.algorithms import approximation as approx
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

rng = np.random.default_rng()  # Random number generator

# Lattice proprieties
dimension_x = 25
dimension_y = 5
dimension_z = 2

# Simulation variables
initial_thermalization_time = (
    40000000  # Steps used to thermalize the system after its generation
)
evolution_time = 600000  # Steps for evolution each time we change temperature
number_of_averages = 3000  # Averages taken at the end of evolution_time for each temperature
interval_of_averages = 100  # Step between each measure
J = 50
initial_temperature = 1
final_temperature = 1000
temperature_step = 10
atoms_to_remove = 0 # Atoms removed in the "Broken lattice" simulation 
GRAPH_MEASURES = True # If true, the code will measure the graph proprieties of aligned spin netowrks

type_of_simulation = "Herdos-Renyi (P=3/4)"  # Text to be added in title of graphs

# Arrays used to store relevant thermodynamic functions
TEMPERATURE = []

THDM_MEASURES = {
    "Magnetization": [],
    "Energy": [],
    "Cv": [],
    "Susceptibility": [],
    "Entropy": [],
    "Free energy": [],
}

# Store the only spins up netowrk measures
UP_SPINS_MEASURES = {
    "Connected components": [],
    "giant component size": [],
    "Link density": [],
    "Clustering coefficent": [],
    "Betweenness centrality": [],
    "Giant component diameter": [],
    "Giant component node connectivity": [],
}

# Store the only spins down netowrk measures
DOWN_SPINS_MEASURES = {
    "Connected components": [],
    "giant component size": [],
    "Link density": [],
    "Clustering coefficent": [],
    "Betweenness centrality": [],
    "Giant component diameter": [],
    "Giant component node connectivity": [],
}


##FUNCTIONS##

# Give spin to nodes
def set_initial_spin(graph):
    for i in range(graph.number_of_nodes()):
        spin = rng.integers(2)
        if spin == 0:
            spin = -1
        graph.nodes[i]["Spin"] = spin


# Put colors on nodes
def color_map(graph):
    colors = []
    for i in range(graph.number_of_nodes()):
        if graph.nodes[i]["Spin"] == 1:
            colors.append("red")
        else:
            colors.append("blue")
    return colors


# Draw the network
def draw(graph):
    nx.draw(
        graph, node_color=color_map(graph)
    )  # , labels=nx.get_node_attributes(graph,"Spin"))


# Evaluate the energy
def evaluate_hamiltonian(graph):
    H = 0
    for edge in graph.edges():  # cycles over the interacting atoms
        spin0 = graph.nodes[edge[0]]["Spin"]
        spin1 = graph.nodes[edge[1]]["Spin"]
        H -= J * (spin0 * spin1)
    return H


# Flip 1 random spin
# It returns variation of energy to be used in evolution
def flip_spin(graph, temperature):
    rnd_spin = rng.integers(atoms_network.number_of_nodes())
    delta_energy = variation_hamiltonian(graph, rnd_spin)
    # Decide if the flip occurs
    if delta_energy < 0 or math.exp(-delta_energy / temperature) > rng.random():
        atoms_network.nodes[rnd_spin]["Spin"] *= -1
        return delta_energy
    else:
        return 0


# Evaluate magnetization per particle
def evaluate_magnetization(graph):
    total_magnetization = 0
    for i in range(graph.number_of_nodes()):
        total_magnetization += graph.nodes[i]["Spin"]
    return total_magnetization / graph.number_of_nodes()


# Evaluate variation of the hamiltonian by the flip of one spin
def variation_hamiltonian(graph, changed_spin):
    Delta_H = 0
    for spin in graph.neighbors(changed_spin):
        Delta_H += 2 * J * graph.nodes[spin]["Spin"] * graph.nodes[changed_spin]["Spin"]
    return Delta_H


# Evaluate variation of the magnetization by the flip of one spin
def variation_magnetization(graph, changed_spin):
    return 2 * graph.nodes[changed_spin]["Spin"] / graph.number_of_nodes()


# Evaluate entropy per particle
def evaluate_entropy(magnetization):
    if magnetization == 1:
        return 0
    return -1 * (
        (magnetization + 1) / 2 * np.log((1 + magnetization) / 2)
        + (1 - magnetization) / 2 * np.log((1 - magnetization) / 2)
    )


# Return network with only atoms with specified spin
def one_spin_network(spin, graph):
    new_graph = graph.copy()
    to_remove_nodes = []
    for node in new_graph:
        if new_graph.nodes[node]["Spin"] != spin:
            to_remove_nodes.append(node)
    new_graph.remove_nodes_from(to_remove_nodes)
    return new_graph


# Remove non selected spins
def remove_different_nodes_from(graph, spins):
    new_graph = graph.copy()
    new_graph.remove_nodes_from(set(graph.nodes()) - set(spins))
    return new_graph


# Find critical temperture as the maximum of a list of a variable (e.g. specific heat)
def temperature_of_maximum(temperatures, thdm_variable):
    maximum = max(thdm_variable)
    temperature_index = thdm_variable.index(maximum)
    return temperatures[temperature_index]


# Remove random nodes
def remove_random_nodes(graph, number_of_nodes):
    removed_nodes = []
    for i in range(number_of_nodes):
        rnd_node = rng.integers(graph.number_of_nodes())
        if rnd_node not in removed_nodes:
            graph.nodes[rnd_node]["Spin"] = 0
            removed_nodes.append(rnd_node)


# Create more then nearest neigbor interaction networks from lattices
def mnn_interaction_from_lattices(graph):
    new_graph = graph.copy()
    for node in graph:
        for first_neighbor in graph.neighbors(node):
            for second_neighbor in graph.neighbors(first_neighbor):
                new_graph.add_edge(node,second_neighbor)
    return new_graph



##CODE##

# Generate the network on which the simulation runs
# NOTE: lattices must undergo a convertion of thei indices
#  from the position in the lattice to just progressive numeration

# atoms_network = nx.convert_node_labels_to_integers(nx.grid_2d_graph(dimension_x,dimension_y,periodic = True), 0)
# atoms_network = nx.convert_node_labels_to_integers(nx.grid_graph((dimension_x,dimension_y,dimension_z,dimension_z),periodic = False), 0)
# atoms_network = nx.convert_node_labels_to_integers(nx.triangular_lattice_graph(dimension_x, dimension_y, periodic= True))
atoms_network = nx.erdos_renyi_graph(dimension_x, 0.75)


# Setting the network
# atoms_network = mnn_interaction_from_lattices(atoms_network)
set_initial_spin(atoms_network)
remove_random_nodes(atoms_network, atoms_to_remove)

# First thermalization
print("-----FIRST THERMALIZATION-----")
for t in tqdm(range(initial_thermalization_time)):
    flip_spin(atoms_network, initial_temperature)

# Evolve at different temperatures
print("-----THERMAL EVOLUTION-----")
for temperature in tqdm(
    range(initial_temperature, final_temperature, temperature_step)
):
    TEMPERATURE.append(temperature)

    # Set of variables used to take averages
    # These will be the sums of the observables

    thdm_averages = {
        "Magnetization": 0,
        "Squared magnetization": 0,
        "Energy": 0,
        "Squared energy": 0,
        "Entropy": 0,
    }

    up_spins_averages = {
        "Connected components": 0,
        "giant component size": 0,
        "Link density": 0,
        "Clustering coefficent": 0,
        "Betweenness centrality": 0,
        "Giant component diameter": 0,
        "Giant component node connectivity": 0,
    }

    down_spins_averages = {
        "Connected components": 0,
        "giant component size": 0,
        "Link density": 0,
        "Clustering coefficent": 0,
        "Betweenness centrality": 0,
        "Giant component diameter": 0,
        "Giant component node connectivity": 0,
    }

    # Time evolution
    # Evaluate initial thermodynamic variables
    system_energy = evaluate_hamiltonian(atoms_network)
    system_magnetization = evaluate_magnetization(atoms_network)

    # Time evolution between successive temperatures
    for t in range(evolution_time):
        rnd_spin = rng.integers(atoms_network.number_of_nodes())  # Choose random spin
        delta_energy = variation_hamiltonian(atoms_network, rnd_spin)
        # Decide if the flip occur using metropolis algorithm
        if delta_energy < 0 or math.exp(-delta_energy / temperature) > rng.random():
            atoms_network.nodes[rnd_spin]["Spin"] *= -1  # Flip the spin
            system_energy += delta_energy  # New energy
            system_magnetization += variation_magnetization(
                atoms_network, rnd_spin
            )  # New magnetization

        # Last iterations evolve the system in order to simulate one ensamble and then get averages
        # Determine if in this step a measure should occur
        if (
            evolution_time - t <= number_of_averages * interval_of_averages
            and (t - number_of_averages * interval_of_averages) % interval_of_averages
            == 0
        ):  # Mesaures
            thdm_averages["Energy"] += system_energy / number_of_averages
            thdm_averages["Squared energy"] += (
                system_energy * system_energy / number_of_averages
            )
            thdm_averages["Magnetization"] += (
                abs(system_magnetization) / number_of_averages
            )
            thdm_averages["Squared magnetization"] += (
                system_magnetization * system_magnetization / number_of_averages
            )
            thdm_averages["Entropy"] += (
                evaluate_entropy(abs(system_magnetization)) / number_of_averages
            )
            if GRAPH_MEASURES == True:
                # Slpit the network in spins up and down
                spin_up = one_spin_network(1, atoms_network)
                spin_down = one_spin_network(-1, atoms_network)
                up_connected_compopnents = nx.number_connected_components(spin_up)
                down_connected_compopnents = nx.number_connected_components(spin_down)
                if up_connected_compopnents != 0:
                    up_giant_component = remove_different_nodes_from(
                        spin_up, max(nx.connected_components(spin_up), key=len)
                    )
                    up_spins_averages["Link density"] += (
                        nx.density(spin_up) / number_of_averages
                    )
                    up_spins_averages["Clustering coefficent"] += (
                        nx.average_clustering(spin_up) / number_of_averages
                    )
                    up_spins_averages["Connected components"] += (
                        up_connected_compopnents / number_of_averages
                    )
                    up_spins_averages["giant component size"] += (
                        up_giant_component.number_of_nodes() / number_of_averages
                    )
                    up_spins_averages["Betweenness centrality"] += (
                        sum(nx.betweenness_centrality(spin_up).values())
                        / spin_up.number_of_nodes()
                        / number_of_averages
                    )  # returns a ditc of nodes so we average
                    up_spins_averages["Giant component diameter"] += (
                        approx.diameter(up_giant_component) / number_of_averages
                    )
                    up_spins_averages["Giant component node connectivity"] += (
                        approx.node_connectivity(up_giant_component)
                        / number_of_averages
                    )
                if down_connected_compopnents != 0:
                    down_giant_component = remove_different_nodes_from(
                        spin_down, max(nx.connected_components(spin_down), key=len)
                    )
                    down_spins_averages["Link density"] += (
                        nx.density(spin_down) / number_of_averages
                    )
                    down_spins_averages["Clustering coefficent"] += (
                        nx.average_clustering(spin_down) / number_of_averages
                    )
                    down_spins_averages["Connected components"] += (
                        down_connected_compopnents / number_of_averages
                    )
                    down_spins_averages["giant component size"] += (
                        down_giant_component.number_of_nodes() / number_of_averages
                    )
                    down_spins_averages["Betweenness centrality"] += (
                        sum(nx.betweenness_centrality(spin_down).values())
                        / spin_down.number_of_nodes()
                        / number_of_averages
                    )  # returns a ditc of nodes so we average
                    down_spins_averages["Giant component diameter"] += (
                        approx.diameter(down_giant_component) / number_of_averages
                    )
                    down_spins_averages["Giant component node connectivity"] += (
                        approx.node_connectivity(down_giant_component)
                        / number_of_averages
                    )

    # Adding averages to the arrays to store thermodynamic functions of temperature
    THDM_MEASURES["Energy"].append(
        thdm_averages["Energy"] / atoms_network.number_of_nodes()
    )
    THDM_MEASURES["Magnetization"].append(thdm_averages["Magnetization"])
    THDM_MEASURES["Cv"].append(
        (
            thdm_averages["Squared energy"]
            - thdm_averages["Energy"] * thdm_averages["Energy"]
        )
        / (temperature * temperature)
        / atoms_network.number_of_nodes()
    )
    THDM_MEASURES["Susceptibility"].append(
        (
            thdm_averages["Squared magnetization"]
            - thdm_averages["Magnetization"] * thdm_averages["Magnetization"]
        )
        / temperature
        / atoms_network.number_of_nodes()
    )
    THDM_MEASURES["Entropy"].append(thdm_averages["Entropy"])
    THDM_MEASURES["Free energy"].append(
        thdm_averages["Energy"] / atoms_network.number_of_nodes()
        - temperature * thdm_averages["Entropy"]
    )

    if GRAPH_MEASURES == True:
        for measure in UP_SPINS_MEASURES:
            UP_SPINS_MEASURES[measure].append(up_spins_averages[measure])
        for measure in DOWN_SPINS_MEASURES:
            DOWN_SPINS_MEASURES[measure].append(down_spins_averages[measure])


# Draw Graphs and print results
T_c_from_cv = temperature_of_maximum(TEMPERATURE, THDM_MEASURES["Cv"])
T_c_from_sus = temperature_of_maximum(TEMPERATURE, THDM_MEASURES["Susceptibility"])
print(
    "Cv maximum critical temperature: "
    + str(T_c_from_cv)
    + "\nchi maximum critical temperature: "
    + str(T_c_from_sus)
)

fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(2, 3, 1)
ax.set_title("Magnetization")
ax.set_xlabel("T", loc="right")
ax.set_ylabel("m", loc="top", rotation=0, fontsize=18)
ax.plot(TEMPERATURE, THDM_MEASURES["Magnetization"], marker=".", linestyle="-")
ax.grid(True)
# ax.axvline(x=T_c, color='r', linestyle='--')
# ax.text(T_c / final_temperature, -0.1, r'$T_c$', color='r', fontsize=12, transform = ax.transAxes)

ax = fig.add_subplot(2, 3, 2)
ax.set_title("Energy")
ax.set_xlabel("T", loc="right")
ax.set_ylabel(r"$\frac{E}{N}$", loc="top", rotation=0, fontsize=18)
ax.plot(TEMPERATURE, THDM_MEASURES["Energy"], marker=".", linestyle="-")
ax.grid(True)
# ax.axvline(x=T_c, color='r', linestyle='--')
# ax.text(T_c / final_temperature, -0.1, r'$T_c$', color='r', fontsize=12, transform = ax.transAxes)

ax = fig.add_subplot(2, 3, 3)
ax.set_title("Heat capacity")
ax.set_xlabel("T", loc="right")
ax.set_ylabel("$C_V$", loc="top", rotation=0)
ax.plot(TEMPERATURE, THDM_MEASURES["Cv"], marker=".", linestyle="-")
ax.grid(True)
ax.axvline(x=T_c_from_cv, color="r", linestyle="--")
ax.text(
    (T_c_from_cv - initial_temperature) / (final_temperature - initial_temperature),
    -0.1,
    r"$T_c$",
    color="r",
    fontsize=12,
    transform=ax.transAxes,
)

ax = fig.add_subplot(2, 3, 4)
ax.set_title("Susceptibility")
ax.set_xlabel("T", loc="right")
ax.set_ylabel(r"$\chi$", loc="top", rotation=0)
ax.plot(TEMPERATURE, THDM_MEASURES["Susceptibility"], marker=".", linestyle="-")
ax.grid(True)
ax.axvline(x=T_c_from_sus, color="r", linestyle="--")
ax.text(
    (T_c_from_sus - initial_temperature) / (final_temperature - initial_temperature),
    -0.1,
    r"$T_c$",
    color="r",
    fontsize=12,
    transform=ax.transAxes,
)

ax = fig.add_subplot(2, 3, 5)
ax.set_title("Entropy")
ax.set_xlabel("T", loc="right")
ax.set_ylabel(r"$\frac{S}{N}$", loc="top", rotation=0, fontsize=18)
ax.plot(TEMPERATURE, THDM_MEASURES["Entropy"], marker=".", linestyle="-")
ax.grid(True)
# ax.axvline(x=T_c, color='r', linestyle='--')
# ax.text(T_c / final_temperature, -0.1, r'$T_c$', color='r', fontsize=12, transform = ax.transAxes)

ax = fig.add_subplot(2, 3, 6)
ax.set_title("Free energy")
ax.set_xlabel("T", loc="right")
ax.set_ylabel(r"$\frac{F}{N}$", loc="top", rotation=0, fontsize=18)
ax.plot(TEMPERATURE, THDM_MEASURES["Free energy"], marker=".", linestyle="-")
ax.grid(True)
# ax.axvline(x=T_c, color='r', linestyle='--')
# ax.text(T_c / final_temperature, -0.1, r'$T_c$', color='r', fontsize=12, transform = ax.transAxes)

fig.suptitle(
    "Simulated Ising model on " + type_of_simulation + " at different temperatures",
    fontsize=20,
    x=0.4,
    y=0.94,
)
description = (
    "N="
    + str(atoms_network.number_of_nodes())
    + " J="
    + str(J)
    + "\nInitial therm. steps: "
    + str(initial_thermalization_time)
    + "\nEvolution steps: "
    + str(evolution_time)
    + "\n"
    + str(number_of_averages)
    + " averages every "
    + str(interval_of_averages)
    + " steps."
)
fig.text(
    0.755,
    0.89,
    description,
    fontsize=14,
    bbox=dict(facecolor="white", edgecolor="black", pad=10.0),
)

fig.tight_layout()
fig.subplots_adjust(bottom=0.1, top=0.84)

# Plotting network related graphs
if GRAPH_MEASURES == True:
    fig2 = plt.figure(figsize=(15, 10))

    ax = fig2.add_subplot(2, 3, 1)
    ax.set_title("Connected components")
    ax.set_xlabel("T", loc="right")
    ax.set_ylabel("Number of connected components")
    ax.plot(
        TEMPERATURE,
        UP_SPINS_MEASURES["Connected components"],
        marker=".",
        linestyle="-",
        color="orange",
    )
    ax.plot(
        TEMPERATURE,
        DOWN_SPINS_MEASURES["Connected components"],
        marker=".",
        linestyle="-",
        color="skyblue",
    )
    ax.grid(True)

    ax = fig2.add_subplot(2, 3, 2)
    ax.set_title("Giant component")
    ax.set_xlabel("T", loc="right")
    ax.set_ylabel("Size of the giant component")
    ax.plot(
        TEMPERATURE,
        UP_SPINS_MEASURES["giant component size"],
        marker=".",
        linestyle="-",
        color="orange",
    )
    ax.plot(
        TEMPERATURE,
        DOWN_SPINS_MEASURES["giant component size"],
        marker=".",
        linestyle="-",
        color="skyblue",
    )
    ax.grid(True)

    ax = fig2.add_subplot(2, 3, 3)
    ax.set_title("Graph density")
    ax.set_xlabel("T", loc="right")
    ax.set_ylabel("Density")
    ax.plot(
        TEMPERATURE,
        UP_SPINS_MEASURES["Link density"],
        marker=".",
        linestyle="-",
        color="orange",
    )
    ax.plot(
        TEMPERATURE,
        DOWN_SPINS_MEASURES["Link density"],
        marker=".",
        linestyle="-",
        color="skyblue",
    )
    ax.grid(True)

    ax = fig2.add_subplot(2, 3, 4)
    ax.set_title("Betweenness centrality")
    ax.set_xlabel("T", loc="right")
    ax.set_ylabel("Average betweenness centrality")
    ax.plot(
        TEMPERATURE,
        UP_SPINS_MEASURES["Betweenness centrality"],
        marker=".",
        linestyle="-",
        color="orange",
    )
    ax.plot(
        TEMPERATURE,
        DOWN_SPINS_MEASURES["Betweenness centrality"],
        marker=".",
        linestyle="-",
        color="skyblue",
    )
    ax.grid(True)

    ax = fig2.add_subplot(2, 3, 5)
    ax.set_title("Giant component diameter")
    ax.set_xlabel("T", loc="right")
    ax.set_ylabel("Links")
    ax.plot(
        TEMPERATURE,
        UP_SPINS_MEASURES["Giant component diameter"],
        marker=".",
        linestyle="-",
        color="orange",
    )
    ax.plot(
        TEMPERATURE,
        DOWN_SPINS_MEASURES["Giant component diameter"],
        marker=".",
        linestyle="-",
        color="skyblue",
    )
    ax.grid(True)

    ax = fig2.add_subplot(2, 3, 6)
    ax.set_title("Giant component node connectivity")
    ax.set_xlabel("T", loc="right")
    ax.set_ylabel("Average node connectivity")
    ax.plot(
        TEMPERATURE,
        UP_SPINS_MEASURES["Giant component node connectivity"],
        marker=".",
        linestyle="-",
        color="orange",
        label="Spin up",
    )
    ax.plot(
        TEMPERATURE,
        DOWN_SPINS_MEASURES["Giant component node connectivity"],
        marker=".",
        linestyle="-",
        color="skyblue",
        label="Spin down",
    )
    ax.grid(True)

    fig2.suptitle(
        "Simulated Ising model on " + type_of_simulation + " at different temperatures",
        fontsize=20,
        x=0.4,
        y=0.94,
    )
    description = (
        "N="
        + str(atoms_network.number_of_nodes())
        + " J="
        + str(J)
        + "\nInitial therm. steps: "
        + str(initial_thermalization_time)
        + "\nEvolution steps: "
        + str(evolution_time)
        + "\n"
        + str(number_of_averages)
        + " averages every "
        + str(interval_of_averages)
        + " steps."
    )
    fig2.text(
        0.755,
        0.89,
        description,
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="black", pad=10.0),
    )
    fig2.legend(
        loc="lower right", ncol=2, bbox_transform=fig.transFigure, fontsize="15"
    )

    fig2.tight_layout()
    fig2.subplots_adjust(bottom=0.1, top=0.84)
plt.show()
