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
    # Location-specific Contact Patterns and NPIs

    ## Introduction

    In Tutorial 3, we applied contact reductions (`Dampings`) uniformly to a single contact matrix. In reality, contacts between individuals occur in different locations: **at home**, **at school**, **at work**, and in **other** places such as transport. Different non-pharmaceutical interventions (NPIs) target different settings, for instance a school closure reduces school contacts, while a home-office mandate mainly reduces work contacts.

    Large-scale studies, such as the POLYMOD project, have measured social contact patterns across different locations (like home, school, work, and others) in various countries. For instance, see

    * **Mossong2008**: Mossong J, Hens N, Jit M, Beutels P, Auranen K, et al. (2008) *Social Contacts and Mixing Patterns Relevant to the Spread of Infectious Diseases*. PLoS Med 5(3): e74. https://doi.org/10.1371/journal.pmed.0050074
    * **Prem2017**: Prem K, Cook AR, Jit M (2017) *Projecting social contact matrices in 152 countries using contact surveys and demographic data*. PLoS Comput Biol 13(9): e1005697. https://doi.org/10.1371/journal.pcbi.1005697

    In this tutorial, we extend the approach from Tutorial 3 by splitting the contact matrix into **location-specific contact matrices** using `ContactMatrixGroup`. This allows us to apply NPIs to individual locations, allowing a more realistic and detailed representation of intervention effects.
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
    return AgeGroup, Damping, mio, osecir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We define four contact locations and use an `IntEnum` to refer to them. We also set the basic simulation parameters - population size, time horizon, and step size - as in Tutorial 3:
    """)
    return


@app.cell
def _():
    from enum import IntEnum

    class Location(IntEnum):
        Home = 0
        School = 1
        Work = 2
        Other = 3

    total_population = 100000
    t0 = 0
    tmax = 100
    dt = 0.1
    return Location, dt, t0, tmax, total_population


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Location-specific Contact Matrices

    Instead of a single contact matrix, we now use a `ContactMatrixGroup`, containing one `ContactMatrix` per location. Each `ContactMatrix` has two components:

    - **Baseline**: the typical number of daily contacts at the location under regular conditions.
    - **Minimum**: the minimal contact rate even under the strictest restrictions, as some contacts, especially at home, cannot be fully avoided.

    The effective contact rate at simulation time $t$ for a location $\ell$ is:

    $$C_\ell(t) = C_{\ell,\text{baseline}} - D_\ell(t) \cdot \left(C_{\ell,\text{baseline}} - C_{\ell,\text{minimum}}\right)$$

    where $D_\ell(t) \leq 1$ is the damping coefficient. For $D > 0$ contacts are reduced; for $D = 1$ they reach the minimum; and for $D < 0$ contacts are **increased above the baseline**. The upper bound $D \leq 1$ is enforced by MEmilio's C++ core. The ODE model uses the **sum** of all location-specific matrices as the total contact rate.

    The total baseline contact rate (sum over all locations) equals $4 + 3 + 2 + 1 = 10$ contacts/day, matching Tutorial 3.
    """)
    return


@app.cell
def _(AgeGroup, Location, mio, np, osecir, total_population):
    def create_model():
        """Create an OSECIR model with location-specific contact matrices."""
        m = osecir.Model(1)
        g = AgeGroup(0)

        # Set infection state stay times (in days)
        m.parameters.TimeExposed[g] = 3.2
        m.parameters.TimeInfectedNoSymptoms[g] = 2.
        m.parameters.TimeInfectedSymptoms[g] = 6.
        m.parameters.TimeInfectedSevere[g] = 12.
        m.parameters.TimeInfectedCritical[g] = 8.

        # Set infection state transition probabilities
        m.parameters.RelativeTransmissionNoSymptoms[g] = 0.67
        m.parameters.TransmissionProbabilityOnContact[g] = 0.1
        m.parameters.RecoveredPerInfectedNoSymptoms[g] = 0.2
        m.parameters.RiskOfInfectionFromSymptomatic[g] = 0.25
        m.parameters.SeverePerInfectedSymptoms[g] = 0.2
        m.parameters.CriticalPerSevere[g] = 0.25
        m.parameters.DeathsPerCritical[g] = 0.3

        # Create ContactMatrixGroup: 4 location matrices, each of size 1x1 (one age group)
        contacts = mio.ContactMatrixGroup(4, 1)

        # Home contacts: baseline 4/day, minimum 1/day (irreducible household contacts)
        contacts[Location.Home] = mio.ContactMatrix(
            np.array([[4.0]]), np.array([[1.0]]))
        # School contacts: baseline 3/day
        contacts[Location.School] = mio.ContactMatrix(
            np.array([[3.0]]), np.array([[0.0]]))
        # Work contacts: baseline 2/day
        contacts[Location.Work] = mio.ContactMatrix(
            np.array([[2.0]]), np.array([[0.0]]))
        # Other contacts: baseline 1/day
        contacts[Location.Other] = mio.ContactMatrix(
            np.array([[1.0]]), np.array([[0.0]]))

        m.parameters.ContactPatterns.cont_freq_mat = contacts

        # Initial populations: 0.5% Exposed, 0.5% InfectedNoSymptoms, rest Susceptible
        m.populations[g,
                      osecir.InfectionState.Exposed] = 0.005 * total_population
        m.populations[g, osecir.InfectionState.InfectedNoSymptoms] = 0.005 * total_population
        m.populations.set_difference_from_total(
            (g, osecir.InfectionState.Susceptible), total_population)

        return m

    return (create_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline Simulation Without NPIs

    We first create and simulate the model without any interventions to obtain a baseline:
    """)
    return


@app.cell
def _(create_model, dt, osecir, t0, tmax):
    model_no_npi = create_model()
    result_no_npi = osecir.simulate(t0, tmax, dt, model_no_npi)
    return (result_no_npi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adding Location-specific NPIs

    We now model a lockdown scenario in which three NPIs are activated simultaneously on **day 20**:

    | Intervention             | Location | Damping $D$ | Effect                                     |
    |--------------------------|----------|-------------|--------------------------------------------|
    | School closure           | School   | 1.0         | All school contacts eliminated             |
    | Home-office mandate      | Work     | 0.5         | Work contacts reduced by 50 %              |
    | Public transport restriction    | Other    | 0.8         | Other contacts reduced by 80 %        |

    The key difference from Tutorial 3 is that each `Damping` is applied to a **specific location matrix** by indexing `cont_freq_mat[location_index]` before calling `add_damping`. In Tutorial 3, `add_damping` was called on the whole group, reducing **all** contact matrices simultaneously. Home contacts are left unrestricted here, but they cannot fall below the minimum of 1 contact/day regardless.
    """)
    return


@app.cell
def _(Damping, Location, create_model, np):
    t_npi_start = 20.0   # day at which interventions start

    model_with_npi = create_model()

    # School closure: damping of 1.0 -> effective school contacts reach the minimum (0)
    model_with_npi.parameters.ContactPatterns.cont_freq_mat[Location.School].add_damping(
        Damping(coeffs=np.ones((1, 1)) * 1.0, t=t_npi_start, level=0, type=0))
    # Home-office mandate: damping of 0.5 -> work contacts halved
    model_with_npi.parameters.ContactPatterns.cont_freq_mat[Location.Work].add_damping(
        Damping(coeffs=np.ones((1, 1)) * 0.5, t=t_npi_start, level=0, type=0))
    # Restrictions in public transport: damping of 0.8 -> other contacts reduced by 80 %
    model_with_npi.parameters.ContactPatterns.cont_freq_mat[Location.Other].add_damping(
        Damping(coeffs=np.ones((1, 1)) * 0.8, t=t_npi_start, level=0, type=0))
    return model_with_npi, t_npi_start


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can verify the effective total contact rates before and after the NPIs by evaluating the contact matrix group at a specific point in time:
    """)
    return


@app.cell
def _(model_with_npi, t_npi_start):
    contacts_before = model_with_npi.parameters.ContactPatterns.cont_freq_mat.get_matrix_at(
        t_npi_start - 1)
    contacts_after = model_with_npi.parameters.ContactPatterns.cont_freq_mat.get_matrix_at(
        t_npi_start + 1)
    print(
        f"Total contacts before NPIs (day {t_npi_start - 1:.0f}): {contacts_before[0, 0]} / day")
    print(
        f"Total contacts after  NPIs (day {t_npi_start + 1:.0f}): {contacts_after[0, 0]} / day")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We simulate the NPI scenario from `t0` to `tmax`:
    """)
    return


@app.cell
def _(dt, model_with_npi, osecir, t0, tmax):
    result_with_npi = osecir.simulate(t0, tmax, dt, model_with_npi)
    return (result_with_npi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualization of Model Output

    We plot and compare the `InfectedSymptoms` trajectories from both scenarios. The vertical dashed line marks the onset of the NPIs on day 20:
    """)
    return


@app.cell
def _(osecir, plt, result_no_npi, result_with_npi, t_npi_start):
    arr_no_npi = result_no_npi.as_ndarray()
    arr_with_npi = result_with_npi.as_ndarray()

    inf_idx = 1 + int(osecir.InfectionState.InfectedSymptoms)

    fig, ax = plt.subplots()
    ax.plot(arr_no_npi[0],   arr_no_npi[inf_idx],
            label='Without NPIs', color='tab:blue')
    ax.plot(
        arr_with_npi[0],
        arr_with_npi[inf_idx],
        label='With location-specific NPIs', linestyle='--',
        color='tab:orange')
    ax.axvline(x=t_npi_start, color='gray', linestyle=':',
               label=f'NPI start (day {int(t_npi_start)})')
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Individuals [#]')
    ax.set_title('Effect of Location-specific NPIs on InfectedSymptoms')
    ax.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The NPIs substantially flatten the infection curve. After day 20, school contacts drop to zero, work contacts are halved, and other contacts are reduced by 80 %. Only household contacts remain, bounded from below by their minimum of 1 contact/day.

    ## Summary and Next Steps

    In this tutorial, we have introduced location-specific contact patterns via `ContactMatrixGroup` and applied targeted NPIs to individual location matrices. Key takeaways:

    - A `ContactMatrixGroup` collects one `ContactMatrix` per location (Home, School, Work, Other).
    - Each `ContactMatrix` has a **baseline** (normal contacts) and a **minimum** (irreducible lower bound).
    - Location-specific NPIs are applied with `cont_freq_mat[location_index].add_damping(...)`. In contrast, Tutorial 3 used `cont_freq_mat.add_damping(...)` which applies to **all** matrices simultaneously.
    - The effective contact rate per location is $C_\text{eff} = C_\text{baseline} - D \cdot (C_\text{baseline} - C_\text{minimum})$, and the ODE model receives the **sum** over all locations.

    In the next tutorial, we will extend this framework with **dynamic NPIs** that are triggered automatically when infection thresholds are exceeded.
    """)
    return


if __name__ == "__main__":
    app.run()
