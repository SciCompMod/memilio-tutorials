import marimo

__generated_with = "0.19.11"
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
    # Dynamic NPIs

    ## Introduction

    In Tutorial 10, we applied NPIs at a **predefined fixed time** (day 20). In practice, interventions are often activated reactively and triggered when the number of infections exceeds a critical threshold, such as a specific incidence per 100,000 individuals.

    MEmilio supports this pattern through **dynamic NPIs**: a set of contact dampings that are automatically activated whenever a specified infection threshold is exceeded, remain active for a defined duration, and are then automatically lifted if the incidence is below the threshold again.

    Key parameters of a dynamic NPI:

    | Parameter     | Meaning                                                                          |
    |---------------|----------------------------------------------------------------------------------|
    | `threshold`   | Incidence limit that triggers the NPI        |
    | `base_value`  | Reference population size for the incidence calculation (typically 100,000)      |
    | `duration`    | Minimum number of days the NPI stays active once triggered                       |
    | `interval`    | How often (in days) the incidence is re-evaluated                                |
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
    import memilio.simulation.osecir as osecir
    from memilio.simulation import AgeGroup, Damping, LogLevel, set_log_level
    set_log_level(LogLevel.Off)
    return AgeGroup, mio, osecir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We reuse the four contact locations from Tutorial 10 and keep the same simulation parameters:
    """)
    return


@app.cell
def _():
    from enum import IntEnum

    class Location(IntEnum):
        Home   = 0
        School = 1
        Work   = 2
        Other  = 3

    total_population = 100000
    t0   = 0
    tmax = 100
    dt   = 0.1
    return Location, dt, t0, tmax, total_population


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We reuse the `create_model()` helper from Tutorial 10, which already sets up location-specific contact matrices (baseline sum = 10 contacts/day):
    """)
    return


@app.cell
def _(AgeGroup, Location, mio, np, osecir, total_population):
    def create_model():
        """Create an OSECIR model with location-specific contact matrices."""
        m = osecir.Model(1)
        g = AgeGroup(0)

        # Set infection state stay times (in days)
        m.parameters.TimeExposed[g]             = 3.2
        m.parameters.TimeInfectedNoSymptoms[g]  = 2.
        m.parameters.TimeInfectedSymptoms[g]    = 6.
        m.parameters.TimeInfectedSevere[g]      = 12.
        m.parameters.TimeInfectedCritical[g]    = 8.

        # Set infection state transition probabilities
        m.parameters.RelativeTransmissionNoSymptoms[g]    = 0.67
        m.parameters.TransmissionProbabilityOnContact[g]  = 0.1
        m.parameters.RecoveredPerInfectedNoSymptoms[g]    = 0.2
        m.parameters.RiskOfInfectionFromSymptomatic[g]    = 0.25
        m.parameters.SeverePerInfectedSymptoms[g]         = 0.2
        m.parameters.CriticalPerSevere[g]                 = 0.25
        m.parameters.DeathsPerCritical[g]                 = 0.3

        # Create ContactMatrixGroup: 4 location matrices, each of size 1x1 (one age group)
        contacts = mio.ContactMatrixGroup(4, 1)
        contacts[Location.Home]   = mio.ContactMatrix(np.array([[4.0]]), np.array([[1.0]]))
        contacts[Location.School] = mio.ContactMatrix(np.array([[3.0]]), np.array([[0.0]]))
        contacts[Location.Work]   = mio.ContactMatrix(np.array([[2.0]]), np.array([[0.0]]))
        contacts[Location.Other]  = mio.ContactMatrix(np.array([[1.0]]), np.array([[0.0]]))
        m.parameters.ContactPatterns.cont_freq_mat = contacts

        # Initial populations: 0.5% Exposed, 0.5% InfectedNoSymptoms, rest Susceptible
        m.populations[g, osecir.InfectionState.Exposed]            = 0.005 * total_population
        m.populations[g, osecir.InfectionState.InfectedNoSymptoms] = 0.005 * total_population
        m.populations.set_difference_from_total(
            (g, osecir.InfectionState.Susceptible), total_population)

        return m

    return (create_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline Simulation Without NPIs

    We first run the model without any interventions:
    """)
    return


@app.cell
def _(create_model, dt, osecir, t0, tmax):
    model_baseline = create_model()
    result_baseline = osecir.simulate(t0, tmax, dt, model_baseline)
    return (result_baseline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining Dynamic NPIs

    Dynamic NPIs are attached to the model via `model.parameters.DynamicNPIsInfectedSymptoms`. The simulator checks every `interval` days whether the current number of `InfectedSymptoms` relative to `base_value` exceeds a threshold. If so, the corresponding set of `DampingSampling` objects is applied for `duration` days.

    Each `DampingSampling` describes one location-specific contact reduction and is constructed as:

    ```python
    mio.DampingSampling(
        value          = mio.UncertainValue(damping_coefficient),
        level          = 0,            # damping level (for combining multiple dampings)
        type           = 0,            # damping type  (for combining multiple dampings)
        time           = 0.0,          # time offset within the NPI duration
        matrix_indices = [loc_index],  # which location matrix to damp
        group_weights  = np.ones(1)    # one entry per age group
    )
    ```

    We define two escalation levels:

    | Level | Threshold (per 100k) | School | Work | Other |
    |-------|---------------------|--------|------|-------|
    | Mild  | 500                 | 0.3    | 0.3  | 0.3   |
    | Strict| 5000                | 1.0    | 0.6  | 0.8   |
    """)
    return


@app.cell
def _(Location, mio, np):
    # Helper: create a DampingSampling for one location
    def loc_damping(coefficient, location_index):
        return mio.DampingSampling(
            value          = mio.UncertainValue(coefficient),
            level          = 0,
            type           = 0,
            time           = 0.0,
            matrix_indices = [int(location_index)],
            group_weights  = np.ones(1)
        )

    # Mild restrictions (threshold: 500 per 100k)
    mild_npis = [
        loc_damping(0.3, Location.School),   # school contacts reduced by 30 %
        loc_damping(0.3, Location.Work),     # work contacts reduced by 30 %
        loc_damping(0.3, Location.Other),    # other contacts reduced by 30 %
    ]

    # EXERCISE: Define strict_npis for the high-incidence threshold (5000 per 100k):
    #   School closure:               D = 1.0  (schools fully closed)
    #   Home-office mandate:          D = 0.6  (work contacts reduced by 60 %)
    #   Public transport restriction: D = 0.8  (other contacts reduced by 80 %)
    # Hint: follow the same pattern as mild_npis above using loc_damping().
    # strict_npis = [???]
    strict_npis = []  # TODO: replace with your implementation
    return mild_npis, strict_npis


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting Up the Model with Dynamic NPIs

    We create a new model and set the two dynamic NPIs. The simulator will automatically select the **highest exceeded threshold** at each check interval:
    """)
    return


@app.cell
def _(create_model, mild_npis, strict_npis):
    model_dynamic = create_model()

    dynamic_npis = model_dynamic.parameters.DynamicNPIsInfectedSymptoms
    # EXERCISE: Configure the dynamic NPI mechanism and register both thresholds.
    #   interval   = 3       (how often the incidence is checked, in days)
    #   duration   = 14      (minimum time an NPI stays active once triggered, in days)
    #   base_value = 100000  (reference population for the incidence calculation)
    #   threshold 1:  500 per 100k -> mild_npis
    #   threshold 2: 5000 per 100k -> strict_npis
    # Hint: use dynamic_npis.interval / .duration / .base_value / .set_threshold(threshold, npis_list)
    # ???
    return (model_dynamic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulation with Dynamic NPIs

    To benefit from dynamic NPI checking, we must use `osecir.Simulation` first. The `Simulation` class overrides `advance()`. We create the simulation object and advance it to `tmax`.
    """)
    return


@app.cell
def _(dt, model_dynamic, osecir, t0, tmax):
    sim = osecir.Simulation(model_dynamic, t0, dt)
    sim.advance(tmax)
    result_dynamic = sim.result
    return (result_dynamic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualization of Model Output

    We compare the `InfectedSymptoms` trajectories from the baseline and the dynamic-NPI scenario. The shaded horizontal bands mark the two thresholds:
    """)
    return


@app.cell
def _(osecir, plt, result_baseline, result_dynamic, total_population):
    arr_baseline = result_baseline.as_ndarray()
    arr_dynamic  = result_dynamic.as_ndarray()

    inf_idx = 1 + int(osecir.InfectionState.InfectedSymptoms)

    # Convert absolute numbers to incidence per 100k for the y-axis
    scale = 100000 / total_population

    fig, ax = plt.subplots()
    ax.plot(arr_baseline[0], arr_baseline[inf_idx] * scale,
            label='Without NPIs', color='tab:blue')
    ax.plot(arr_dynamic[0],  arr_dynamic[inf_idx] * scale,
            label='With dynamic NPIs', linestyle='--', color='tab:orange')

    # Mark the two thresholds
    ax.axhline(y=500,  color='gold',   linestyle=':', linewidth=1.5, label='Mild threshold (500 / 100k)')
    ax.axhline(y=5000, color='tab:red', linestyle=':', linewidth=1.5, label='Strict threshold (5000 / 100k)')

    ax.set_xlabel('Time [days]')
    ax.set_ylabel('InfectedSymptoms')
    ax.set_title('Dynamic NPIs: Automatic Threshold-based Interventions')
    ax.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary and Next Steps

    In this tutorial, we introduced **dynamic NPIs** contact reductions which are triggered automatically when an incidence threshold is exceeded. Key takeaways:

    - Dynamic NPIs are configured via `model.parameters.DynamicNPIsInfectedSymptoms`.
    - Three control parameters determine the mechanism: `interval` (check frequency), `duration` (minimum active time), and `base_value` (reference population).
    - Each threshold is paired with a list of `DampingSampling` objects that specify **which location** and **how much** to damp.
    - If multiple thresholds are defined, MEmilio automatically selects the highest exceeded threshold at each check.
    - Dynamic NPIs require using `osecir.Simulation(model, t0, dt)` and `sim.advance(tmax)`, since the threshold check is embedded in `advance()`.
    """)
    return


if __name__ == "__main__":
    app.run()
