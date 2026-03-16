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
    # Simulating a first ODE-based model from Python

    ## Introduction

    MEmilio implements various models based on ordinary differential equations (ODEs). ODE-based models are a subclass of compartmental models in which individuals are grouped into subpopulations called compartments.

    In this tutorial we will setup and run MEmilio's ODE-based SECIR-type model. This model is particularly suited for pathogens with pre- or asymptomatic infection states and when severe or critical symptoms are possible. The model assumes perfect immunity after recovery. The used infection states or compartments are Susceptible (S), Exposed(E), Non-symptomatically Infected (Ins), Symptomatically Infected (Isy), Severely Infected (Isev), Critically Infected (Icri), Dead (D) and Recovered (R). The transitions are depicted in the following figure.

    The example requires to have the memilio-simulation package installed which can be accessed under https://github.com/SciCompMod/memilio/tree/main/pycode/memilio-simulation.
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(
    """
    graph TD
        S["**S**\nSusceptible"] -->|infection| E["**E**\nExposed, Not Infectious"]
        E -->|incubation| INS["**Ins**\nInfectious, No Symptoms"]

        INS -->|develops symptoms| ISy["**Isy**\nInfectious, Symptoms"]
        INS -->|recovers| R["**R**\nRecovered"]

        ISy -->|worsens| ISev["**Isev**\nInfected, Severe"]
        ISy -->|recovers| R

        ISev -->|worsens| ICr["**Icr**\nInfected, Critical"]
        ISev -->|recovers| R

        ICr -->|recovers| R
        ICr -->|dies| D["**D**\nDead"]

        style S fill:#d4943a,color:#000,stroke:#000
        style E fill:#d4943a,color:#000,stroke:#000
        style INS fill:#d4943a,color:#000,stroke:#000
        style ISy fill:#d4943a,color:#000,stroke:#000
        style ISev fill:#d4943a,color:#000,stroke:#000
        style ICr fill:#d4943a,color:#000,stroke:#000
        style R fill:#d4943a,color:#000,stroke:#000
        style D fill:#d4943a,color:#000,stroke:#000
        """
        )
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
    import memilio.simulation.osecir as osecir
    from memilio.simulation import AgeGroup, LogLevel, set_log_level
    set_log_level(LogLevel.Off)
    return AgeGroup, osecir


@app.cell
def _(mo):
    mo.md(r"""
    Next, we need to specify basic parameters. In this tutorial, we use a simple model without spatial resolution and with only one age group. We first define the `total_population` size and the simulation horizong through the start day `t0`, and the simulation's end point `tmax`. By default, the ODE is solved with adaptive time stepping and the initial time step is `dt`.
    """)
    return


@app.cell
def _():
    total_population = 100000
    t0 = 0
    tmax = 100
    dt = 0.1
    return dt, t0, tmax, total_population


@app.cell
def _(mo):
    mo.md(r"""
    To set up a model without any further stratification, i.e., with only one age group is created a follows:
    """)
    return


@app.cell
def _(osecir):
    model = osecir.Model(1)
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""
    Next, we have to set the epidemiological model parameters which include the average stay times per infection state, the state transition probabilities, and the contact frequency. A list of all parameters can be found at https://memilio.readthedocs.io/en/latest/cpp/models/osecir.html. The parameters can be set as follows:
    """)
    return


@app.cell
def _(AgeGroup, model, np):
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
    contact_frequency = 10
    model.parameters.ContactPatterns.cont_freq_mat[0].baseline = np.ones((1, 1)) * contact_frequency
    return (group,)


@app.cell
def _(mo):
    mo.md(r"""
    In addition to the parameters, the initial number of individuals in each compartment has to be set. If a compartment is not set, its initial value is zero by default. In this example, we start our simulation with 1 % of the population initially infected, distributing them equally to the `Exposed` and the `InfectedNoSymptoms` state, where the latter contains pre- and asymptomatic infectious individuals. With the last line, we set the remaining part of the population (99%) to be susceptible.
    """)
    return


@app.cell
def _(group, model, osecir, total_population):
    # 1% of the population is initially infected, 0.5 % Exposed and 0.5 % in the pre- or asymptomatic state
    model.populations[group, osecir.InfectionState.Exposed] = 0.005 * total_population
    model.populations[group, osecir.InfectionState.InfectedNoSymptoms] = 0.005 * total_population
    # The rest of the population is Susceptible
    model.populations.set_difference_from_total(
        (group, osecir.InfectionState.Susceptible), total_population)
    return


@app.cell
def _(mo):
    mo.md(r"""
    To check that all initial parameter and compartmental values are in a meaningful range, MEmilio provides the `check_constraints` function. If a value exceeds its meaningful range, a warning is printed and the function returns `True`, otherwise it returns `False`.
    """)
    return


@app.cell
def _(model):
    model.check_constraints()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Model simulation

    After having initialized the model, dynamics can be simulated. The simulation output is a time series containing the evolution of all compartments over time. We now simulate the model from `t0` to `tmax` -- using an initial step size `dt`.
    """)
    return


@app.cell
def _(dt, model, osecir, t0, tmax):
    result = osecir.simulate(t0, tmax, dt, model)
    return (result,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualization of model output

    Since we have used an adaptive integration scheme, the `result` time series contains irregular time points. We can look at the time points with `get_times`.
    """)
    return


@app.cell
def _(result):
    # print the first 10 time points
    print(result.get_times()[0:10])
    return


@app.cell
def _(mo):
    mo.md(r"""
    We can also look at the compartmental values at these time points using `print_table`.
    """)
    return


@app.cell
def _(result):
    print(result.print_table(return_string=True))
    return


@app.cell
def _(mo):
    mo.md(r"""
    We can use MEmilio's functionality to linearly interpolate `TimeSeries` objects. By default, interpolation is conducted to to full days.
    """)
    return


@app.cell
def _(osecir, result):
    interpolated_result = osecir.interpolate_simulation_result(result)
    # print the first 10 time points
    print(interpolated_result.get_times()[0:10])
    return (interpolated_result,)


@app.cell
def _(mo):
    mo.md(r"""
    The interpolated result can be printed as before.
    """)
    return


@app.cell
def _(interpolated_result):
    print(interpolated_result.print_table(return_string=True))
    return


@app.cell
def _(mo):
    mo.md(r"""
    Single time points and result values can be accessed via the `get_time` and `get_value` functions. The whole time series can be converted to an array with the first row the time points and the following rows the compartment sizes using `as_ndarray`. Let's have a look at the simulated trajectories of all compartments:
    """)
    return


@app.cell
def _(osecir, plt, result):
    # Convert time series to array
    result_array = result.as_ndarray()

    # Plot the compartment trajectories over time
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
    ## Summary and next steps

    In the simulated scenario, ~90% of the population got infected within the simulated time frame of 100 days. In a real pandemic, non-pharmaceutical interventions (NPIS) can be implemented to prevent an outbreak from occurring or mitigate its strength. For instance, these include mask wearing or physical distancing which aims at decreasing the effective contacts and hence the infection risk. NPIs can be realized in MEmilio's (ODE-based) models through applying `Dampings`. How to do that will be shown in Tutorial 3. In Tutorial 2, we show how to extract information on daily new infections or hospitalizations through MEmilio's built-in flow formulation.
    """)
    return


if __name__ == "__main__":
    app.run()
