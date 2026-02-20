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

    MEmilio implements various models based on ordinary differential equations (ODEs). ODE-based models are a subclass of compartmental models in which individuals are grouped into subpopulations called compartments.

    In this tutorial we will setup and run MEmilio's ODE-based SECIR-type model. This model is particularly suited for pathogens with pre- or asymptomatic infection states and when severe or critical symptoms are possible. The model assumes perfect immunity after recovery. The used infection states or compartments are Susceptible (S), Exposed(E), Non-symptomatically Infected (Ins), Symptomatically Infected (Isy), Severely Infected (Isev), Critically Infected (Icri), Dead (D) and Recovered (R).

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
    Next, we need to specify the parameters. In this tutorial, we use a simple model without spatial or age resolution i.e. we only consider one (age) group. The non-epidemiological parameters are the total population size, the start day `t0`, the simulation time frame `tmax` and the contact frequency. We should also choose an initial time step, though the ODE solver will use adaptive time steps later on.
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
    return contact_frequency, dt, t0, tmax, total_population


@app.cell
def _(mo):
    mo.md(r"""
    A model without any further stratification i.e. with only one (sociodemographic) group is created via:
    """)
    return


@app.cell
def _(osecir):
    model = osecir.Model(1)
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""
    Next, we have to set the epidemiological model parameters which include the average stay times per infection state, the state transition probabilities and the contact frequency. A list of all parameters can be found at https://memilio.readthedocs.io/en/latest/cpp/models/osecir.html. The parameters can be set as follows:
    """)
    return


@app.cell
def _(AgeGroup, contact_frequency, model, np):
    # Set infection state stay times (in days)
    group = AgeGroup(0)
    model.parameters.TimeExposed[group] = 3.2
    model.parameters.TimeInfectedNoSymptoms[group] = 2.
    model.parameters.TimeInfectedSymptoms[group] = 6.
    model.parameters.TimeInfectedSevere[group] = 12.
    model.parameters.TimeInfectedCritical[group] = 8.

    # Set infection state transition probabilities
    model.parameters.RelativeTransmissionNoSymptoms[group] = 0.67
    model.parameters.TransmissionProbabilityOnContact[group] = 0.1
    model.parameters.RecoveredPerInfectedNoSymptoms[group] = 0.2
    model.parameters.RiskOfInfectionFromSymptomatic[group] = 0.25
    model.parameters.SeverePerInfectedSymptoms[group] = 0.2
    model.parameters.CriticalPerSevere[group] = 0.25
    model.parameters.DeathsPerCritical[group] = 0.3

    # Set contact frequency
    model.parameters.ContactPatterns.cont_freq_mat[0].baseline = np.ones((1, 1)) * contact_frequency
    return (group,)


@app.cell
def _(mo):
    mo.md(r"""
    In addition to the parameters, the initial number of individuals in each compartment has to be set. If a compartment is not set, its initial value is zero. In this example we start our simulation with 1% of the population initially infected. We have to distribute the infected individuals to the different states one can have when being infected. We do this by distributing them equally to the `Exposed` and the `InfectedNoSymptoms` state which contains pre- and asymptomatically infectious individuals. That means 0.5% of the population is exposed, 0.5% is non-symptomatically infected and the rest (99%) is susceptible.
    """)
    return


@app.cell
def _(group, model, osecir, total_population):
    # 1% of the population is initially infected, 0.5% Exposed and 0.5% in the pre- or asymptomatic state
    model.populations[group, osecir.InfectionState.Exposed] = 0.005 * total_population
    model.populations[group, osecir.InfectionState.InfectedNoSymptoms] = 0.005 * total_population
    # The rest of the population is Susceptible
    model.populations.set_difference_from_total(
        (group, osecir.InfectionState.Susceptible), total_population)
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
    After having initialized the model, dynamics can be simulated. The simulation output is a time series containing the evolution of all compartments over time. In the following we simulate the model from `t0` to `tmax` with initial step size `dt` and subsequently print the time series result:
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
    Single result time points and its values can be accessed via the `get_time` and `get_value` functions. The whole time series can be converted to an array with the first row the time points and the following rows the compartment sizes using `as_ndarray`. Let's have a look at the simulated trajectories of all compartments:
    """)
    return


@app.cell
def _(osecir, plt, result):
    # Convert time series to array
    result_array = result.as_ndarray()

    # Plot the number of infected with symptoms over time
    fig, ax = plt.subplots()
    time = result_array[0, :]
    ax.plot(time, result_array[1 + int(osecir.InfectionState.Susceptible), :], label='Susceptible')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.Exposed), :], label='Exposed')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.InfectedNoSymptoms), :], label='Infected No Symptoms')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.InfectedSymptoms), :], label='Infected Symptoms')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.InfectedSevere), :], label='Infected Severe')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.InfectedCritical), :], label='Infected Critical')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.Recovered), :], label='Recovered')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.Dead), :], label='Deaths')
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Individuals [#]')
    ax.legend()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    In the simulated scenario, ~90% of the population got infected within the simulated time frame of 100 days. In a real pandemic, non-pharmaceutical interventions (NPIS) can be implemented to prevent an outbreak from occurring or mitigate its strength. These include for example mask wearing or physical distancing which aims at decreasing the effective contacts and hence the infection risk. NPIs can be realized in MEmilio's ODE-based models through applying dampings. How to do that will be shown in the next tutorial.
    """)
    return


if __name__ == "__main__":
    app.run()
