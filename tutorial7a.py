import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction

    In the previous tutorials, we saw how to set up and run an age-resolved ODE-based SECIR-type model. However, one limiting assumption of simple ODE-based models is the assumption of homogenous mixing within the population. To overcome this limitation and incorporate spatial heterogeneity, in this example we show how to use MEmilio's graph-based metapopulation model. This model realizes mobility between regions via graph edges, while every region is represented by a graph node containing it's own ODE-based model.

    The example requires to have the memilio-simulation package installed which can be accessed under https://github.com/SciCompMod/memilio/tree/main/pycode/memilio-simulation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Setup

    We first import the needed functions from the memilio-simulation package:
    """)
    return


@app.cell
def _():
    import memilio.simulation as mio
    import memilio.simulation.osecir as osecir
    from memilio.simulation import AgeGroup, LogLevel, set_log_level
    set_log_level(LogLevel.Off)
    return AgeGroup, mio, osecir


@app.cell
def _(mo):
    mo.md(r"""
    We set the simulation start time `t0`, the end time `tmax` and the initial step size `dt` as:
    """)
    return


@app.cell
def _():
    t0 = 0
    tmax = 100
    dt = 0.1
    return t0, tmax


@app.cell
def _(mo):
    mo.md(r"""
    Next, we need to specify the parameters. We will initialize a metapopulation model with two regions. The total population as well as the epidemiological parameters will be the same for both regions.
    """)
    return


@app.cell
def _():
    # Initialize total population per region
    total_population_per_region = 100000
    return (total_population_per_region,)


@app.cell
def _(mo):
    mo.md(r"""
    We use a model with three age groups for both regions:
    """)
    return


@app.cell
def _(osecir):
    num_age_groups = 3
    model = osecir.Model(3)
    return model, num_age_groups


@app.cell
def _(mo):
    mo.md(r"""
    Now, we have to set the epidemiological model parameters which are dependent on age group. A list of all parameters can be found at https://memilio.readthedocs.io/en/latest/cpp/models/osecir.html.

    We choose an increasing risk of severe and critical infections for age group 2 and 3 compared to age group 1. The other parameters are equal for all age groups.
    """)
    return


@app.cell
def _(AgeGroup, model, np, num_age_groups):
    for ag in range(num_age_groups):
        # Set infection state stay times (in days)
        model.parameters.TimeExposed[AgeGroup(ag)] = 3.2
        model.parameters.TimeInfectedNoSymptoms[AgeGroup(ag)] = 2.
        model.parameters.TimeInfectedSymptoms[AgeGroup(ag)] = 6.
        model.parameters.TimeInfectedSevere[AgeGroup(ag)] = 12.
        model.parameters.TimeInfectedCritical[AgeGroup(ag)] = 8.

        # Set infection state transition probabilities
        model.parameters.RelativeTransmissionNoSymptoms[AgeGroup(ag)] = 0.67
        model.parameters.TransmissionProbabilityOnContact[AgeGroup(ag)] = 0.1
        model.parameters.RecoveredPerInfectedNoSymptoms[AgeGroup(ag)] = 0.2
        model.parameters.RiskOfInfectionFromSymptomatic[AgeGroup(ag)] = 0.25
        model.parameters.DeathsPerCritical[AgeGroup(ag)] = 0.3

    # The groups have an increasing risk of severe and critical infections
    model.parameters.SeverePerInfectedSymptoms[AgeGroup(0)] = 0.2
    model.parameters.SeverePerInfectedSymptoms[AgeGroup(1)] = 0.2 * 1.5
    model.parameters.SeverePerInfectedSymptoms[AgeGroup(2)] = 0.2 * 2
    model.parameters.CriticalPerSevere[AgeGroup(0)] = 0.25
    model.parameters.CriticalPerSevere[AgeGroup(1)] = 0.25 * 1.5
    model.parameters.CriticalPerSevere[AgeGroup(2)] = 0.25 * 2

    # Set contact frequency
    model.parameters.ContactPatterns.cont_freq_mat[0].baseline = np.ones((num_age_groups, num_age_groups)) * 10
    return


@app.cell
def _(mo):
    mo.md(r"""
    Next, we create the graph via:
    """)
    return


@app.cell
def _(osecir):
    graph = osecir.MobilityGraph()
    return (graph,)


@app.cell
def _(mo):
    mo.md(r"""
    In the graph-based metapopulation model, every graph node gets it's own ODE-based model which is copied when adding a graph node and handing the model to it as parameter. Therefore we can choose different initial conditions (as well as differing parameters) for different graph nodes. In our example, we simulate two regions with only one region having initially infected individuals. We choose 1% initially infected for that region while the other region starts with a totally susceptible population.

    The model compartments for the first node are initialized via:
    """)
    return


@app.cell
def _(AgeGroup, model, num_age_groups, osecir, total_population_per_region):
    # The population is equally distributed among the age groups
    for group in range(num_age_groups):
        # 1% of the population is initially infected, 0.5% Exposed and 0.5% in the pre- or asymptomatic state
        model.populations[AgeGroup(group), osecir.InfectionState.Exposed] = 0.005 * total_population_per_region / num_age_groups
        model.populations[AgeGroup(group), osecir.InfectionState.InfectedNoSymptoms] = 0.005 * total_population_per_region / num_age_groups
        # The rest of the population is Susceptible
        model.populations.set_difference_from_group_total_AgeGroup(
            (AgeGroup(group), osecir.InfectionState.Susceptible), total_population_per_region / num_age_groups)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Then, we add a node to the graph created above and hand over the model as follows:
    """)
    return


@app.cell
def _(graph, model, t0):
    # Add node with id 0 and copy beforehand initialized model to it
    graph.add_node(id=0, model=model, t0=t0) 
    return


@app.cell
def _(mo):
    mo.md(r"""
    Before adding the second node to the graph, we set the model's initial populations to totally susceptible:
    """)
    return


@app.cell
def _(AgeGroup, model, num_age_groups, osecir, total_population_per_region):
    # The population is equally distributed among the age groups
    for age in range(num_age_groups):
        # No infected individuals
        model.populations[AgeGroup(age), osecir.InfectionState.Exposed] = 0
        model.populations[AgeGroup(age), osecir.InfectionState.InfectedNoSymptoms] = 0
        # The total population is Susceptible
        model.populations.set_difference_from_group_total_AgeGroup(
            (AgeGroup(age), osecir.InfectionState.Susceptible), total_population_per_region / num_age_groups)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Subsequently, we add the second node to the graph:
    """)
    return


@app.cell
def _(graph, model, t0):
    # Add node with id 0 and copy beforehand initialized model to it
    graph.add_node(id=1, model=model, t0=t0) 
    return


@app.cell
def _(mo):
    mo.md(r"""
    If we would simulate the graph-based metapopulation model now, we would just have two independent ODE-based SECIR-type models running with different initial conditions. In reality, there is usually exchange between regions through individuals travelling or commuting from one region to another. This can be realized via graph edges.

    We here use a symmetric mobility i.e. we have the same number of individuals that travel from node 0 to node 1 as vice versa. We let 10% of the population commute via the edges twice a day - except dead individuals - which is specified by the mobility coefficient and the exchange time step `dt_exchange`:
    """)
    return


@app.cell
def _(graph, mio, model, np, osecir):
    # One coefficient per (age group x compartment)
    mobility_coefficients = 0.1 * np.ones(model.populations.numel())
    # Dead individuals do not commute
    mobility_coefficients[osecir.InfectionState.Dead] = 0
    mobility_params = mio.MobilityParameters(mobility_coefficients)
    # Add two edges to graph
    graph.add_edge(0, 1, mobility_params)
    graph.add_edge(1, 0, mobility_params)
    # Individuals are exchanged every half day
    dt_exchange = 0.5
    return


@app.cell
def _(mo):
    mo.md(r"""
    We now have finished initializing the metapopulation model. The graph-based simulation is created and advanced until `tmax` via:
    """)
    return


@app.cell
def _(graph, osecir, t0, tmax):
    # Create graph simulation and advance until tmax
    sim = osecir.MobilitySimulation(graph, t0, dt=0.5)
    sim.advance(tmax)
    return (sim,)


@app.cell
def _(mo):
    mo.md(r"""
    As every graph node has its own model, we get one result time series per node. Those can be accessed as follows:
    """)
    return


@app.cell
def _(sim):
    result_region0 = sim.graph.get_node(0).property.result
    result_region1 = sim.graph.get_node(1).property.result
    return result_region0, result_region1


@app.cell
def _(mo):
    mo.md(r"""
    As already seen in the previous tutorials, the time series do not have equidistant time steps because we use an adaptive integrator. Therefore, we interpolate the time series of both regions using the linea interpolation function:
    """)
    return


@app.cell
def _(osecir, result_region0, result_region1):
    result_region0_interpolated = osecir.interpolate_simulation_result(result_region0)
    result_region1_interpolated = osecir.interpolate_simulation_result(result_region1)
    return result_region0_interpolated, result_region1_interpolated


@app.cell
def _(mo):
    mo.md(r"""
    Finally, we can compare the trajectories of all infection states for both regions. In the following, we plot the number of `InfectedNoSymptoms` aggregated over all age groups for both regions:
    """)
    return


@app.cell
def _(osecir, plt, result_region0_interpolated, result_region1_interpolated):
    # Convert time series to array
    result_array_0 = result_region0_interpolated.as_ndarray()
    result_array_1 = result_region1_interpolated.as_ndarray()

    # Plot the number of non-symptomatically infected for both regions
    fig, ax = plt.subplots()
    time = result_array_0[0, :]
    InfectedNoSymptoms_0 = result_array_0[1 + int(osecir.InfectionState.InfectedNoSymptoms), :] + result_array_0[1 + int(osecir.InfectionState.InfectedNoSymptoms) + int(osecir.InfectionState.Dead) + 1, :] + result_array_0[1 + int(osecir.InfectionState.InfectedNoSymptoms) + 2 * (int(osecir.InfectionState.Dead) + 1), :]
    InfectedNoSymptoms_1 = result_array_1[1 + int(osecir.InfectionState.InfectedNoSymptoms), :] + result_array_1[1 + int(osecir.InfectionState.InfectedNoSymptoms) + int(osecir.InfectionState.Dead) + 1, :] + result_array_1[1 + int(osecir.InfectionState.InfectedNoSymptoms) + 2 * (int(osecir.InfectionState.Dead) + 1), :]
    ax.plot(time, InfectedNoSymptoms_0, label='Infected No Symptoms Region 1')
    ax.plot(time, InfectedNoSymptoms_1, label='Infected No Symptoms Region 2')
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Individuals [#]')
    ax.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
