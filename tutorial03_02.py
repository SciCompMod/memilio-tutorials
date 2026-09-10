import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Combining Multiple Interventions
    ## Introduction

    In the previous tutorial, we applied non-pharmaceutical interventions by reducing the contacts in a population via a multiplicative factor. For more advanced simulations and studies of interventions, we can model several interventions in place at the same time. In this tutorial we want to show how we combine multiple interventions. Every `Damping` has a **Level** and a **Type**:

    - Dampings with the same **Level** but different **Type** are **summed**. This fits measures that each affect a different, non-overlapping part of the population, e.g., a school closure and an office closure, since neither affects the contacts the other one does.
    - Dampings on different **Levels** are combined multiplicatively, i.e., two dampings `a` and `b` on different levels combine to `1 - (1-a)(1-b)`. This fits measures of a fundamentally different kind acting on the *same* population, e.g., mask-wearing (which reduces the risk of whatever contacts happen) and homeschooling (which reduces how many contacts happen in the first place). Since mask-wearing's effect only applies to the contacts homeschooling leaves behind, the two effects compound rather than simply add.
    - A new damping with the same **Level and Type** as an existing one **replaces** it from its start time onward instead of adding to it, e.g., strengthening or relaxing a single measure over time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Setup

    We use the same SECIR-type model with one age group as for the previous tutorial. For a detailed description on that, see Tutorial 1.
    """)
    return


@app.cell
def _():
    import memilio.simulation.osecir as osecir
    from memilio.simulation import AgeGroup, Damping, LogLevel, set_log_level
    set_log_level(LogLevel.Off)
    return AgeGroup, Damping, osecir


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
    model.parameters.ContactPatterns.cont_freq_mat[0].baseline = np.ones((1, 1)) * contact_frequency

    # 1% of the population is initially infected, 0.5% Exposed and 0.5% in the pre- or asymptomatic state
    model.populations[group, osecir.InfectionState.Exposed] = 0.005 * total_population
    model.populations[group, osecir.InfectionState.InfectedNoSymptoms] = 0.005 * total_population
    # The rest of the population is Susceptible
    model.populations.set_difference_from_total(
        (group, osecir.InfectionState.Susceptible), total_population)
    return dt, model, t0, tmax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Same Level, Different Types: Additive

    Suppose two measures start on day 10, each affecting a different, non-overlapping part of the population: a partial school closure reduces contacts by 20%, and an office closure reduces contacts by a further 40%. We give them the same `level=0` but different `type`, and MEmilio sums their values: `0.2 + 0.4 = 0.6`.
    """)
    return


@app.cell
def _(Damping, model, np):
    model.parameters.ContactPatterns.cont_freq_mat.add_damping(
        Damping(coeffs=np.ones((1, 1)) * 0.2, t=10.0, level=0, type=0))  # partial school closure
    model.parameters.ContactPatterns.cont_freq_mat.add_damping(
        Damping(coeffs=np.ones((1, 1)) * 0.4, t=10.0, level=0, type=1))  # office closure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A Different Level: Multiplicative

    On day 25, a mask mandate takes effect, reducing the risk of whatever contacts still happen by 30%, regardless of how many contacts the school/office closures already removed. Since it's a fundamentally different kind of measure acting on the same population, we place it on a new `level=1`. It combines multiplicatively with the level-0 total: `1 - (1 - 0.6)(1 - 0.3) = 0.72`
    """)
    return


@app.cell
def _(Damping, model, np):
    model.parameters.ContactPatterns.cont_freq_mat.add_damping(
        Damping(coeffs=np.ones((1, 1)) * 0.3, t=25.0, level=1, type=0))  # mask mandate
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Same Level and Type: Replaces

    At day 40, the partial school closure is escalated to full homeschooling, now reducing contacts by 40% instead of 20%. Because this damping shares its `level=0, type=0` with the day-10 school closure, it **replaces** it rather than adding to it. From day 40 onward, level 0 is the sum of this new value and the still-active office-closure damping: `0.4 + 0.4 = 0.8`, which then combines multiplicatively with the mask mandate on level 1: `1 - (1 - 0.8)(1 - 0.3) = 0.86`.
    """)
    return


@app.cell
def _(Damping, model, np):
    model.parameters.ContactPatterns.cont_freq_mat.add_damping(
        Damping(coeffs=np.ones((1, 1)) * 0.4, t=40.0, level=0, type=0))  # full homeschooling
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verifying the Combined Effect

    We sample the effective contact rate at a few time points and compare against the values
    computed by hand above.
    """)
    return


@app.cell
def _(model):
    baseline = model.parameters.ContactPatterns.cont_freq_mat[0].baseline[0, 0]
    sample_times = [0, 15, 20, 30, 39, 45]
    for t_sample in sample_times:
        effective = model.parameters.ContactPatterns.cont_freq_mat.get_matrix_at(t_sample)[0, 0]
        reduction = 1 - effective / baseline
        print(f"t = {t_sample:>3}: effective contacts = {effective:.2f}, "
              f"combined reduction = {reduction:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Expected values:

    - `t = 0`: no damping active, reduction = 0.
    - `t = 15`: only the day-10 school/office closures are active, reduction = 0.6 (`0.2 + 0.4`).
    - `t = 20`: unchanged, still 0.6.
    - `t = 30`: the mask mandate is active too, reduction = 0.72 (`1 - (1-0.6)(1-0.3)`).
    - `t = 39`: unchanged, still 0.72.
    - `t = 45`: full homeschooling has replaced the partial school closure, reduction = 0.86
      (`1 - (1 - (0.4+0.4))(1-0.3)`).

    ## Model Simulation
    """)
    return


@app.cell
def _(dt, model, osecir, t0, tmax):
    result = osecir.simulate(t0, tmax, dt, model)
    print(result.print_table(return_string=True))
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualization of Model Output
    """)
    return


@app.cell
def _(osecir, plt, result):
    result_array = result.as_ndarray()
 
    fig, ax = plt.subplots()
    time = result_array[0, :]
    ax.plot(time, result_array[1 + int(osecir.InfectionState.Exposed), :], label='Exposed')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.InfectedNoSymptoms), :], label='Infected No Symptoms')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.InfectedSymptoms), :], label='Infected Symptoms')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.InfectedSevere), :], label='Infected Severe')
    ax.plot(time, result_array[1 + int(osecir.InfectionState.InfectedCritical), :], label='Infected Critical')

    for t_damping, label in [(10, 'school + office closure'), (25, 'mask mandate'),
                              (40, 'full homeschooling')]:
        ax.axvline(t_damping, color='gray', linestyle='--', linewidth=0.8)
        ax.text(t_damping, ax.get_ylim()[1] * 0.95, label, rotation=90,
                verticalalignment='top', fontsize=8, color='gray')

    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Individuals [#]')
    ax.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
