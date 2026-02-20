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

    In Tutorial 1, we created, initialized and simulated MEmilio's ODE-based SECIR-type model without any sociodemographic resolution. All ODE-based models have the possibility to add an arbitrary number of sociodemographic groups which can represent certain certain risk groups, like vaccination or age groups. Adding those groups can have a relevant impact on the simulation outcome. If for example older people have a higher risk of severe and critical infections, that can have an impact on ICU occupancy.
    In the following, we initialize and simulate an ODE-based SECIR-type model with three age groups.

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
    import memilio.simulation.osecir as osecir
    from memilio.simulation import AgeGroup, LogLevel, set_log_level
    set_log_level(LogLevel.Off)
    return AgeGroup, osecir


@app.cell
def _(mo):
    mo.md(r"""
    Next, we need to specify the parameters. The non-epidemiological parameters are the total population size, the start day `t0`, the simulation time frame `tmax` and the contact frequency. We should also choose an initial time step, though the ODE solver will use adaptive time steps later on.
    """)
    return


@app.cell
def _():
    # Initialize total population, simulation start time, simulation time frame and initial step size
    total_population = 100000
    t0 = 0
    tmax = 100
    dt = 0.1
    contact_frequency = 10
    return dt, t0, tmax, total_population


@app.cell
def _(mo):
    mo.md(r"""
    Then, we create a model with three age groups via:
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
    In this example, 0.5% of the population is initially in `Exposed` and 0.5% is initially in `InfectedNoSymptoms` while the remaining 99% of the population is `Susceptible`. The fractions are equally distributed to all age groups by:
    """)
    return


@app.cell
def _(AgeGroup, model, num_age_groups, osecir, total_population):
    # The population is equally distributed among the age groups
    for group in range(num_age_groups):
        # 1% of the population is initially infected, 0.5% Exposed and 0.5% in the pre- or asymptomatic state
        model.populations[AgeGroup(group), osecir.InfectionState.Exposed] = 0.005 * total_population / num_age_groups
        model.populations[AgeGroup(group), osecir.InfectionState.InfectedNoSymptoms] = 0.005 * total_population / num_age_groups
        # The rest of the population is Susceptible
        model.populations.set_difference_from_group_total_AgeGroup(
            (AgeGroup(group), osecir.InfectionState.Susceptible), total_population / num_age_groups)
    return


@app.cell
def _(mo):
    mo.md(r"""
    To get reasonable results, the model needs to have plausible parameter values e.g. average stay times or transition probabilities have to be greater zero. MEmilio provides the possibility to check and automatically correct the initialized parameters by applying the `apply_constraints` function. If a value has to be corred a warning is printed and the function returns `True`, otherwise it returns `False`.
    """)
    return


@app.cell
def _(model):
    # Apply mathematical constraints to parameters
    model.apply_constraints()
    return


@app.cell
def _(mo):
    mo.md(r"""
    After having initialized the model, dynamics can be simulated. The simulation output is a time series containing the evolution of all compartments per age group over time. In the following we simulate the model from `t0` to `tmax` with initial step size `dt` and subsequently print the time series result:
    """)
    return


@app.cell
def _(dt, model, osecir, t0, tmax):
    # Simulate model from t0 to tmax with initial step size dt
    result = osecir.simulate(t0, tmax, dt, model)
    result.print_table()
    return (result,)


@app.cell
def _(mo):
    mo.md(r"""
    Because we use an adaptive integrator as default, the result time series does not have equidistant time steps. If we however want to have values at predefined time points, MEmilio provides the functionality to linearly interpolate a time series. You can give the function the specific time points you want the results to be interpolated to or use the default setting which interpolates to full days.
    """)
    return


@app.cell
def _(osecir, result):
    # Interpolate result to full days
    interpolated_result = osecir.interpolate_simulation_result(result)
    interpolated_result.print_table()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Single result time points and its values can be accessed via the `get_time` and `get_value` functions. The whole time series can be converted to an array with the first row the time points and the following rows the compartment sizes using `as_ndarray`. Let's have a look at the simulated trajectories for severe and critical infection of all age groups:
    """)
    return


@app.cell
def _(osecir, plt, result):
    # Convert time series to array
    result_array = result.as_ndarray()

    # Plot the number of infected with symptoms over time
    fig, ax = plt.subplots()
    time = result_array[0, :]
    ax.plot(time, result_array[1 + int(osecir.InfectionState.InfectedSevere), :], label='Infected Severe Group 1')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.InfectedCritical), :], label='Infected Critical Group 1')
    ax.plot(time, result_array[1 + (int(osecir.InfectionState.Dead) + 1) + int(osecir.InfectionState.InfectedSevere), :], label='Infected Severe Group 2')
    ax.plot(time, result_array[1 + (int(osecir.InfectionState.Dead) + 1) + int(osecir.InfectionState.InfectedCritical), :], label='Infected Critical Group 2')
    ax.plot(time, result_array[1 + 2 * (int(osecir.InfectionState.Dead) + 1) + int(osecir.InfectionState.InfectedSevere), :], label='Infected Severe Group 3')
    ax.plot(time, result_array[1 + 2 * (int(osecir.InfectionState.Dead) + 1) + int(osecir.InfectionState.InfectedCritical), :], label='Infected Critical Group 3')
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Individuals [#]')
    ax.legend()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    So far, we only cosidered one homogeneously mixed population. However, disease spread is often influenced by regional patterns resulting in spatial heterogeneity among the distribution of infections. To include spatial resolution, MEmilio's ODE-based Graph-Metapopulation Model can be used. This is further explained in the next tutorial.
    """)
    return


if __name__ == "__main__":
    app.run()
