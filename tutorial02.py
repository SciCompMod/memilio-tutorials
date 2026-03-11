import marimo

__generated_with = "0.20.4"
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
    # Simulating Daily Incidences with Flow-Based Models

    ## Introduction

    In Tutorial 1, we learned how to set up and simulate an ODE-based SECIR-type model. The output of the standard simulation gave us the exact number of individuals in each compartment (e.g., Susceptible, InfectedSymptoms, Recovered) at any given time.

    However, public health data usually reports *incidence* -- such as **daily new cases** or **daily new hospitalizations**. If we simply look at the difference in the `InfectedSymptoms` compartment between today and yesterday, we do not get the true number of new cases. This is because, during that same day, other individuals may have recovered or progressed to a more severe state, leaving the compartment.

    To obtain the exact number of new transitions (flows) between compartments, MEmilio provides a **flow-based formulation**.

    The computational overhead of this flow-based formulation is minimal. The transition rates between compartments have to be evaluated at every step in the standard compartmental formulation anyway. Therefore, calculating the cumulative flows simultaneously only adds a small amount of overhead due to the mapping of these variables and a slightly higher memory usage to store the additional values.

    In this tutorial, we will show how to use `simulate_flows` to obtain these transition counts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Setup

    We set up the exact same model as in Tutorial 1. We define the simulation timeframe, initialize the model parameters, and set the initial populations.
    """)
    return


@app.cell
def _(np):
    import memilio.simulation.osecir as osecir
    from memilio.simulation import AgeGroup, LogLevel, set_log_level
    set_log_level(LogLevel.Off)

    # Initialize total population, simulation start time, timeframe, and step size
    total_population = 100000
    t0 = 0
    tmax = 100
    dt = 0.1

    # Create model with one age group
    model = osecir.Model(1)
    group = AgeGroup(0)

    # Set infection state stay times (in days)
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

    # Initialize compartments
    model.populations[group, osecir.InfectionState.Exposed] = 0.005 * total_population
    model.populations[group, osecir.InfectionState.InfectedNoSymptoms] = 0.005 * total_population
    model.populations.set_difference_from_total(
        (group, osecir.InfectionState.Susceptible), total_population)
    return dt, model, osecir, t0, tmax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before running the simulation, we check if all initial values and parameters are within a valid range.
    *Note: MEmilio's `check_constraints()` returns `True` if a constraint is violated, and `False` if everything is fine.*
    """)
    return


@app.cell
def _(model):
    constraints_violated = model.check_constraints()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulating Flows

    Instead of using `osecir.simulate`, we now use `osecir.simulate_flows`.

    This function integrates the transition rates between compartments. The result is a vector which containts the compartment states as first element, and the cumulative flows between compartments as the second element. The cumulative flows are the total number of transitions that have occurred between compartments up to that point in time. The compartment states are the same as the output of `osecir.simulate`. Note that the entries of the vectors are `TimeSeries` objects.
    """)
    return


@app.cell
def _(dt, model, osecir, t0, tmax):
    # Simulate flows instead of just compartment states
    results = osecir.simulate_flows(t0, tmax, dt, model)
    return (results,)


@app.cell
def _(results):
    compartments = results[0]  # The first element of results contains the compartment data
    flows = results[1]  # The second element of results contains the flow data
    return compartments, flows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extracting Daily Incidences

    The `flow_result` contains continuous cumulative data. To extract daily metrics (like daily new cases), we need to:
    1. Interpolate the flow results to full integer days.
    2. Calculate the difference between consecutive days to get the discrete daily incidence.

    Since the flow values are stored in a `TimeSeries` object, we can directly use the interpolation method to achieve this.

    Note: the adaptive integrator may make time steps which are larger than 1 day. Following, `interpolate_simulation_result` then fills missing days by linear interpolation of the cumulative flows, producing constant values in the daily differences.
    """)
    return


@app.cell
def _(flows, np, osecir):
    # 1. Interpolate flows to full days
    interpolated_flows = osecir.interpolate_simulation_result(flows)

    # Transform the interpolated flows into a NumPy array
    flow_array = interpolated_flows.as_ndarray()

    # Row 0 contains the time points
    time_days = flow_array[0, :]

    # 2. Calculate daily differences
    # We slice [1:, :] to remove the time row. Now row 0 is Flow 0!
    daily_flows = np.diff(flow_array[1:, :], axis=1)

    # Slice time_days to match the dimension of daily_flows (N-1)
    plot_time = time_days[1:]

    print(flow_array.shape)
    print("ji")
    return daily_flows, plot_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extracting Specific Transitions

    To plot specific transitions, we need to extract the correct rows from our `daily_flows` array. MEmilio flattens all possible flows into a 1D list. The sorting follows two clear rules:

    1. **By Age Group:** The array contains all flows for the first age group, followed by all flows for the second age group, and so on.
    2. **By Infection Progression:** Within each age group, the sequence strictly follows the chronological progression of the disease (including parallel paths for confirmed cases).

    For a model with $N_G$ age groups and $N_{T_G}$ flows per group, the flat index for flow $f$ in age group $g$ is simply calculated as $g \times N_{T_G} + f$.

    Since our tutorial model only has one age group ($g=0$), we just need the base indices of the flows. The SECIR model defines 15 specific flows. The first few are ordered like this:
    * Index 0: `Susceptible` $\rightarrow$ `Exposed` (Total new infections)
    * Index 1: `Exposed` $\rightarrow$ `InfectedNoSymptoms`
    * **Index 2: `InfectedNoSymptoms` $\rightarrow$ `InfectedSymptoms` (New symptomatic cases)**
    * Index 3: `InfectedNoSymptoms` $\rightarrow$ `Recovered`
    * ...
    * **Index 6: `InfectedSymptoms` $\rightarrow$ `InfectedSevere` (New hospitalizations)**

    The full list of flows can be found in the [MEmilio documentation](https://memilio.readthedocs.io/en/latest/cpp/models/osecir.html#flows).
    We will now hardcode indices 2 and 6 to extract the daily incidences for new symptomatic cases and new hospitalizations.
    """)
    return


@app.cell
def _(compartments, daily_flows, osecir, plot_time, plt):
    # 1. Prepare compartment data for comparison
    # Interpolate to match the time grid of our flows
    comp_array = osecir.interpolate_simulation_result(compartments).as_ndarray()

    # Extract the "InfectedSevere" compartment (Currently Hospitalized)
    # Adding 1 because row 0 is the time axis
    state_severe_idx = 1 + int(osecir.InfectionState.InfectedSevere)
    current_severe = comp_array[state_severe_idx, 1:] 

    # 2. Flow indices
    new_symptomatic_idx = 2
    new_hospitalized_idx = 6

    # 3. Create the plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # --- Subplot 1: Disease Progression (The Time Lag) ---
    ax[0].bar(plot_time, daily_flows[new_symptomatic_idx, :], color='coral', alpha=0.5, label='New Symptomatic Cases')
    ax[0].bar(plot_time, daily_flows[new_hospitalized_idx, :], color='darkred', alpha=0.8, label='New Hospitalizations')
    ax[0].set_title('Disease Progression: Infection to Hospitalization')
    ax[0].set_xlabel('Time [days]')
    ax[0].set_ylabel('Daily New Cases [#]')
    ax[0].legend()

    # --- Subplot 2: Incidence vs. Bed Occupancy ---
    # Axis 1: Daily New Cases (Incidence)
    ax[1].bar(plot_time, daily_flows[new_hospitalized_idx, :], color='steelblue', alpha=0.5, label='Daily Admissions (Incidence)')
    ax[1].set_xlabel('Time [days]')
    ax[1].set_ylabel('Daily New Hospitalizations [#]', color='steelblue')
    ax[1].tick_params(axis='y', labelcolor='steelblue')

    # Axis 2: Current Occupancy
    ax2 = ax[1].twinx()
    ax2.plot(plot_time, current_severe, color='black', linewidth=3, label='Currently Hospitalized')
    ax2.set_ylabel('Total Hospital Bed Occupancy [#]', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    ax[1].set_title('Incidence vs. Bed Occupancy')

    # Combine legends from both axes
    lines_1, labels_1 = ax[1].get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    By utilizing MEmilio's flow-based simulation, we can easily extract daily transition values, such as daily new cases or hospitalizations, without losing information to simultaneous compartmental outflows.

    In **Tutorial 3**, we will explore how to influence these trajectories by applying non-pharmaceutical interventions (NPIs) using `Damping` parameters.
    """)
    return


if __name__ == "__main__":
    app.run()
