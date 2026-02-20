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

    In the previous tutorial, we created, initialized and simulated MEmilio's ODE-based SECIR-type model with one (age) group. In this tutorial, we will show how to incorporate non-pharmaceutical interventions (NPIs) through the use of `Dampings` in the ODE-based SECIR-type model.

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
    from memilio.simulation import AgeGroup, Damping, LogLevel, set_log_level
    set_log_level(LogLevel.Off)
    return AgeGroup, Damping, osecir


@app.cell
def _(mo):
    mo.md(r"""
    Next, we create and initialize a SECIR-type model with one age group. For a detailed description on taht, see Tutorial 1.
    """)
    return


@app.cell
def _(AgeGroup, np, osecir):
    # Initialize total population, simulation start time, simulation time frame and initial step size
    total_population = 100000
    t0 = 0
    tmax = 100
    dt = 0.1
    contact_frequency = 10

    # Create model with one age group
    model = osecir.Model(1)

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
    model.parameters.ContactPatterns.cont_freq_mat[0].baseline = np.ones((1, 1)) * 10
    return dt, group, model, t0, tmax, total_population


@app.cell
def _(mo):
    mo.md(r"""
    After the model initialization, we add a contact reduction (`Damping`) that represents an NPI like e.g. mask wearing or social distancing. Dampings are a factor applied to the contact frequency and can be added to the model at fixed simulation time points before simulating. They have a *Level* and a *Type*. A damping with a given level and type replaces the previously active one with the same level and type, while all currently active dampings of one level and different types are summed up. If two dampings have different levels (independent of the type) they are combined multiplicatively. In the following we apply a `Damping` of 0.9 after 10 days and another damping of 0.6 after 20 days which means that the contacts are reduced by 10% and 40%, respectively. To always retain a minimum level of contacts, a minimum contact frequency can be set that is never deceeded. In our example we set this minimum contact rate to 0.
    """)
    return


@app.cell
def _(Damping, model, np):
    # Set minimum contact frequency
    model.parameters.ContactPatterns.cont_freq_mat[0].minimum = np.zeros((1, 1))

    # Add contact reduction by 10% after 10 days
    model.parameters.ContactPatterns.cont_freq_mat.add_damping(Damping(coeffs=np.ones((1, 1)) * 0.9, t=10.0, level=0, type=0))
    # Add contact reduction by 40% after 20 days
    model.parameters.ContactPatterns.cont_freq_mat.add_damping(Damping(coeffs=np.ones((1, 1)) * 0.6, t=20.0, level=0, type=0))
    return


@app.cell
def _(mo):
    mo.md(r"""
    Again, we start with 0.5% of the population initially in `Exposed` and 0.5% initially in `InfectedNoSymptoms` while the remaining 99% is `Susceptible`.
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
    We simulate the model from `t0` to `tmax` with initial step size `dt` and subsequently print the time series result:
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
    Subsequently, we interpolate the simulated time series to full days using the linear interpolation function:
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
    We plot the trajectories of all infected compartments i.e. `Exposed`, `InfectedNoSymptoms`, `InfectedSymptoms`, `InfectedSevere` and `InfectedCritical`.
    """)
    return


@app.cell
def _(osecir, plt, result):
    # Convert time series to array
    result_array = result.as_ndarray()

    # Plot the number of infected with symptoms over time
    fig, ax = plt.subplots()
    time = result_array[0, :]
    ax.plot(time, result_array[1 + int(osecir.InfectionState.Exposed), :], label='Exposed')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.InfectedNoSymptoms), :], label='Infected No Symptoms')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.InfectedSymptoms), :], label='Infected Symptoms')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.InfectedSevere), :], label='Infected Severe')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.InfectedCritical), :], label='Infected Critical')
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Individuals [#]')
    ax.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
