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
    # Simulating an ODE-based metapopulation model with implicit mobility from Python
    ## Introduction

    In the previous tutorials, we saw how to set up and run an age-resolved ODE-based SECIR-type model. However, one limiting assumption of simple ODE-based models is the assumption of homogenous mixing within the population. To overcome this limitation and incorporate spatial heterogeneity, in this example we show how to use MEmilio's equation-based metapopulation model. This model realizes mobility between regions in an implicit manner, by including its effects into the ODEs. Currently, this model is only available for the SEIR model.

    The example requires to have the memilio-simulation package installed which can be accessed under https://github.com/SciCompMod/memilio/tree/main/pycode/memilio-simulation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Setup

    We first import the needed functions from the memilio-simulation package:
    """)
    return


@app.cell
def _():
    import memilio.simulation as mio
    import memilio.simulation.oseir_metapop as oseir_metapop
    from memilio.simulation import AgeGroup, Region, LogLevel, set_log_level
    set_log_level(LogLevel.Off)
    return AgeGroup, Region, mio, oseir_metapop


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
    return dt, t0, tmax


@app.cell
def _(mo):
    mo.md(r"""
    Next, we initialize a metapopulation model with two regions and three age groups:
    """)
    return


@app.cell
def _(oseir_metapop):
    num_regions = 2
    num_age_groups = 3
    model = oseir_metapop.Model(num_regions, num_age_groups)
    return model, num_age_groups, num_regions


@app.cell
def _(mo):
    mo.md(r"""
    Then, we set the populations for both regions and age groups and seed the infection in region 0 and age group 1:
    """)
    return


@app.cell
def _(AgeGroup, mio, model, num_age_groups, num_regions, oseir_metapop):
    # Initialize total population per region
    total_population_per_region = 100000

    for _region in range(num_regions):
        for _ag in range(num_age_groups):
            model.populations[mio.Region(_region), AgeGroup(
                _ag), oseir_metapop.InfectionState.Susceptible] = total_population_per_region / num_age_groups

    model.populations[mio.Region(0), AgeGroup(
        1), oseir_metapop.InfectionState.Exposed] = 10
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now, we have to set the epidemiological model parameters which are dependent on the age group.

    We choose an increasing risk of transmission for age group 2 and 3 compared to age group 1. The other parameters are equal for all age groups.
    """)
    return


@app.cell
def _(AgeGroup, Region, model, np, num_age_groups, num_regions):
    for _region in range(num_regions):
        for _ag in range(num_age_groups):
            # Set infection state stay times (in days)
            model.parameters.TimeExposed[Region(_region), AgeGroup(_ag)] = 3.2
            model.parameters.TimeInfected[Region(_region), AgeGroup(_ag)] = 2.

        # Set infection state transition probabilities
        model.parameters.TransmissionProbabilityOnContact[Region(
            _region), AgeGroup(0)] = 0.05
        model.parameters.TransmissionProbabilityOnContact[Region(
            _region), AgeGroup(1)] = 0.1
        model.parameters.TransmissionProbabilityOnContact[Region(
            _region), AgeGroup(2)] = 0.135

    # Set contact frequency
    model.parameters.ContactPatterns.cont_freq_mat[0].baseline = np.ones(
        (num_age_groups, num_age_groups)) * 10
    return


@app.cell
def _(mo):
    mo.md(r"""
    The mobility between regions is described by a matrix giving the fraction of individuals moving from one region to the other during a day. In this model, commuting or movement of people is not performed explicitly. Instead, we include its effect into the ODEs.

    We here use a symmetric mobility, i.e., we have the same number of individuals that travel from region 0 to region 1 and vice versa:
    """)
    return


@app.cell
def _(model, np):
    mobility_data_commuter = np.array([[0.9, 0.1], [0.1, 0.9]])
    model.set_commuting_strengths(mobility_data_commuter)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Model simulation

    We now have finished initializing the metapopulation model. The graph-based simulation is created and advanced until `tmax` via:
    """)
    return


@app.cell
def _(dt, model, oseir_metapop, t0, tmax):
    result = oseir_metapop.simulate(t0, tmax, dt, model)
    return (result,)


@app.cell
def _(mo):
    mo.md(r"""
    As a result, we get one time series including both regions. We can use the linear interpolation function to get a time series with equidistant time steps:
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    As already seen in the previous tutorials, the time series do not have equidistant time steps because we use an adaptive integrator. Therefore, we interpolate the time series of both regions using the linea interpolation function:
    """)
    return


@app.cell
def _(oseir_metapop, result):
    result_interpolated = oseir_metapop.interpolate_simulation_result(result)
    return (result_interpolated,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualization of model output

    Finally, we can compare the trajectories of all infection states for both regions. In the following, we plot the number of `Infected` aggregated over all age groups for both regions:
    """)
    return


@app.cell
def _(num_age_groups, oseir_metapop, plt, result_interpolated):
    # Convert time series to array
    region_offset = (
        int(oseir_metapop.InfectionState.Recovered) + 1) * num_age_groups
    result_array_0 = result_interpolated.as_ndarray()[:region_offset, :]
    result_array_1 = result_interpolated.as_ndarray(
    )[region_offset:2 * region_offset, :]

    # Plot the number of non-symptomatically infected for both regions
    fig, ax = plt.subplots()
    time = result_array_0[0, :]
    Infected_0 = result_array_0[1 + int(oseir_metapop.InfectionState.Infected), :] + result_array_0[1 + int(oseir_metapop.InfectionState.Infected) + int(
        oseir_metapop.InfectionState.Recovered) + 1, :] + result_array_0[1 + int(oseir_metapop.InfectionState.Infected) + 2 * (int(oseir_metapop.InfectionState.Recovered) + 1), :]
    Infected_1 = result_array_1[1 + int(oseir_metapop.InfectionState.Infected), :] + result_array_1[1 + int(oseir_metapop.InfectionState.Infected) + int(
        oseir_metapop.InfectionState.Recovered) + 1, :] + result_array_1[1 + int(oseir_metapop.InfectionState.Infected) + 2 * (int(oseir_metapop.InfectionState.Recovered) + 1), :]
    ax.plot(time, Infected_0, label='Infected Region 1')
    ax.plot(time, Infected_1, label='Infected Region 2')
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Individuals [#]')
    ax.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
