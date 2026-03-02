import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise 1

    This exercise aligns with tutorial 1, so please see have a look at that tutorial for a detailed description of the code.

    Please set the average time individuals spend in the exposed state (`TimeExposed`) to 4 days and simulate the model starting with 2% of the population initially infected. The initially infected should be distributed to the compartpartments as follows: 0.5% is initially exposed (state `Exposed`), 0.5% is non-symptomatically infected (state `InfectedNosymptoms`) and 1% is symptomatically infected (state `InfectedSymptoms`).
    The positions where you should insert the missing code is marked with *TODO*.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    return mo, np, plt


@app.cell
def _():
    import memilio.simulation.osecir as osecir
    from memilio.simulation import AgeGroup, LogLevel, set_log_level
    set_log_level(LogLevel.Off)
    return AgeGroup, osecir


@app.cell
def _():
    total_population = 100000
    t0 = 0
    tmax = 100
    dt = 0.1
    return dt, t0, tmax, total_population


@app.cell
def _(osecir):
    model = osecir.Model(1)
    return (model,)


@app.cell
def _(AgeGroup, model, np):
    # Set infection state stay times (in days)
    group = AgeGroup(0)

    ### TODO ###

    ############################

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
def _(group, model, osecir, total_population):
    # 2% of the population is initially infected
    model.populations[group, osecir.InfectionState.Exposed] = 0.005 * total_population
    model.populations[group, osecir.InfectionState.InfectedNoSymptoms] = 0.005 * total_population

    ### TODO ###

    ############################

    # The rest of the population is Susceptible
    model.populations.set_difference_from_total(
        (group, osecir.InfectionState.Susceptible), total_population)
    return


@app.cell
def _(model):
    model.check_constraints()
    return


@app.cell
def _(dt, model, osecir, t0, tmax):
    result = osecir.simulate(t0, tmax, dt, model)
    return (result,)


@app.cell
def _(result):
    # print the first 10 time points
    print(result.get_times()[0:10])
    return


@app.cell
def _(result):
    result.print_table()
    return


@app.cell
def _(osecir, result):
    interpolated_result = osecir.interpolate_simulation_result(result)
    # print the first 10 time points
    print(interpolated_result.get_times()[0:10])
    return (interpolated_result,)


@app.cell
def _(interpolated_result):
    interpolated_result.print_table()
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


if __name__ == "__main__":
    app.run()
