import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    return mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Breaking news 📺

    Today, the German federal health authority RKI announced that the recently observed sharp increase in ICU case numbers after acute respiratory infections is due to a new lineage of the influenza virus. It was first sequenced at the university hospital of cologne and thus called Influenza B/Colognia/314/2026. After it's detection, laboratories in the whole of Germany rushed to test stored samples of recent patients s.t. we now have a good overview on the death numbers and numbers of patients in ICUs.

    The first patients fell sick shortly after excessively celebrating carneval on Rose Monday, which was 2026-02-16 this year.

    The RKI today published all available data and asked for modellers around the world to estimate paramaters and make predictions on the course of the disease.

    First forecasts already predicted an exponential growth of the number of deaths and ICU patients. Non-pharmaceutical interventions (NPIs) should be implementes as soon as possible, but also as tailored as possible. To do that, we need to better understand the mechanics of our endemic and look at the infections per age group.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction

    In this tutorial we try to infer model parameters based on "real" data. This allows us to make predictions on the course of an epidemic and study sets of potential interventions.

    This is the second notebook in a series of three that introduce the usage of Approximate Bayesian computation (ABC) with MEmilio. Having started with a simple ODE model in the first part, we now want to add multiple age groups to our model.

    Note that this is not a tutorial on ABC. For details on ABC, we refer to the tutorials of pyabc and Bayesflow as listed below. Here, we just show how to connect these tools with the MEmilio software framework.

    In the first two tutorials we use the package [pyabc](https://pyabc.readthedocs.io/en/latest/) for likelihood-free inference. While more suitable approaches might be available for the model considered here, we use a simple model to demonstrate the coupling. For more advanced and stochastic models, you only need to replace the model and accept longer run times.
    For the last tutorial, we use [Bayesflow](https://bayesflow.org/main/index.html), a state of the art python library for Bayesian inference with deep learning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Setup

    We will reuse the SECIR-type ODE-based model that was already used in the last tutorials. The model already has all the compartments for which we have data available, so we do not need to implement anything from scratch. Let's first import the necessary functions:
    """)
    return


@app.cell
def _():
    import memilio.simulation.osecir as osecir
    from memilio.simulation import AgeGroup, Damping, LogLevel, set_log_level
    from memilio.simulation import ContactMatrixGroup, ContactMatrix, read_mobility_plain
    set_log_level(LogLevel.Warning)
    return (
        AgeGroup,
        ContactMatrix,
        ContactMatrixGroup,
        osecir,
        read_mobility_plain,
    )


@app.cell
def _(mo):
    mo.md(r"""
    Next, we need to define the setting and the model parameters. As we have six different age groups in our data (the standard for data provided by the RKI), the number of unknown parameters is multiplied by six.
    The parameters that were already known during the last fitting process are the total population of Germany, day 0 (rose monday proved to be a good guess) and thus the length of the simulation (the last reported day in the data). We should also choose an initial time step, but the ODE solver will use adaptive time steps later on. Previous studys give us good values for the contact rates in Germany, even for different age groups, that we will just reuse here.

    Luckily, our endemic has only been going on for a few weeks and "only" about 200 people died. Unfortunately, however, this implies that there is not a lot of data to fit to. A parameter space of dimension 72 will not be easy to infer, if at all. Thus we need to fall back to use some of the preliminary data that is provided by the RKI and to only infer the values that are relevant to our current research question - how can we implement NPIs in the most efficient way? To do that, we want to know how likely it is for each age group to get hospitalized and how easily they infect other people. Additionally, we need to estimate the parameters around infections without symptoms, as it is too early for big studies to find them. Nevertheless, we can already assume that they are as infectious as people with symptoms.
    All other parameters we will just take from the rough estimates that RKI provides based on observations in the hospitals and surveys of infected individuals.
    """)
    return


@app.cell
def _():
    # known parameters:
    total_population = 83000000
    t0 = 0
    tmax = 30
    dt = 0.1
    num_age_groups = 6
    TimesExposed = [2.5, 2.3, 2.1, 1.9, 1.7, 1.5]
    TimesInfectedSymptoms = [3.4, 3.8, 4, 4.2, 4.6, 5]
    TimesInfectedSevere = [3.6, 3.8, 4, 4.2, 4.4, 4.4]
    TimesInfectedCritical = [2, 2, 2.4, 2.4, 3, 3.5]
    SeveresPerInfectedSymptoms = [0.025, 0.05, 0.1, 0.2, 0.24, 0.25]
    CriticalsPerSevere = [0.4, 0.15, 0.05, 0.15, 0.25, 0.4]
    DeathssPerCritical = [0.25, 0.2, 0.15, 0.2, 0.3, 0.4]
    RisksOfInfectionFromSymptomatic = [1, 1, 1, 1, 1, 1]
    RelativeTransmissionsNoSymptoms = [1, 1, 1, 1, 1, 1]
    return (
        CriticalsPerSevere,
        DeathssPerCritical,
        RelativeTransmissionsNoSymptoms,
        RisksOfInfectionFromSymptomatic,
        SeveresPerInfectedSymptoms,
        TimesExposed,
        TimesInfectedCritical,
        TimesInfectedSevere,
        TimesInfectedSymptoms,
        dt,
        num_age_groups,
        t0,
        tmax,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As before, for our unknown paramters we will use a dictionary which serves for sampling random realizations, but now we will also add a function to set our known parameters.
    """)
    return


@app.cell
def _(AgeGroup, num_age_groups):
    def set_parameters(model, parameters):
        for index in range(num_age_groups):
            group = AgeGroup(index)
            model.parameters.TimeInfectedNoSymptoms[
                group] = parameters[f"TimeInfectedNoSymptoms{index}"]
            model.parameters.TransmissionProbabilityOnContact[
                group] = parameters[f"TransmissionProbabilityOnContact{index}"]
            model.parameters.RecoveredPerInfectedNoSymptoms[
                group] = parameters[f"RecoveredPerInfectedNoSymptoms{index}"]

    return (set_parameters,)


@app.cell
def _(
    AgeGroup,
    CriticalsPerSevere,
    DeathssPerCritical,
    RelativeTransmissionsNoSymptoms,
    RisksOfInfectionFromSymptomatic,
    SeveresPerInfectedSymptoms,
    TimesExposed,
    TimesInfectedCritical,
    TimesInfectedSevere,
    TimesInfectedSymptoms,
):
    def set_known_parameters(model):
        for ag, val in enumerate(TimesExposed):
            model.parameters.TimeExposed[AgeGroup(ag)] = val
        for ag, val in enumerate(TimesInfectedSymptoms):
            model.parameters.TimeInfectedSymptoms[AgeGroup(ag)] = val
        for ag, val in enumerate(TimesInfectedSevere):
            model.parameters.TimeInfectedSevere[AgeGroup(ag)] = val
        for ag, val in enumerate(TimesInfectedCritical):
            model.parameters.TimeInfectedCritical[AgeGroup(ag)] = val
        for ag, val in enumerate(SeveresPerInfectedSymptoms):
            model.parameters.SeverePerInfectedSymptoms[AgeGroup(ag)] = val
        for ag, val in enumerate(CriticalsPerSevere):
            model.parameters.CriticalPerSevere[AgeGroup(ag)] = val
        for ag, val in enumerate(DeathssPerCritical):
            model.parameters.DeathsPerCritical[AgeGroup(ag)] = val
        for ag, val in enumerate(RisksOfInfectionFromSymptomatic):
            model.parameters.RiskOfInfectionFromSymptomatic[AgeGroup(ag)] = val
        for ag, val in enumerate(RelativeTransmissionsNoSymptoms):
            model.parameters.RelativeTransmissionNoSymptoms[AgeGroup(ag)] = val

    return (set_known_parameters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How should we initialize the population? As before, we do not assume the existence of cross-immunity. We also don't know how many people have been initially exposed, but we are pretty sure that they were in Age Group 2 (15 - 35 years old) due to the origin in Kölner Karneval.
    """)
    return


@app.cell
def _(AgeGroup, osecir):
    def set_population(model, parameters):
        model.populations[AgeGroup(
            0), osecir.InfectionState.Susceptible] = 3700000
        model.populations[AgeGroup(
            1), osecir.InfectionState.Susceptible] = 7920000
        model.populations[AgeGroup(
            2), osecir.InfectionState.Susceptible] = 18760000
        model.populations[AgeGroup(
            3), osecir.InfectionState.Susceptible] = 28080000
        model.populations[AgeGroup(
            4), osecir.InfectionState.Susceptible] = 17720000
        model.populations[AgeGroup(
            5), osecir.InfectionState.Susceptible] = 7390000
        model.populations[AgeGroup(
            2), osecir.InfectionState.Susceptible].value -= parameters["InitiallyExposed"]
        model.populations[AgeGroup(
            2), osecir.InfectionState.Exposed] = parameters["InitiallyExposed"]

    return (set_population,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For contact data we just use the results of established studies. We can use the `MEmilio` functionality to read in contact matrices:
    """)
    return


@app.cell
def _(ContactMatrix, ContactMatrixGroup, os, read_mobility_plain):
    def set_contact_matrices(model):
        contact_matrices = ContactMatrixGroup(1, 6)
        baseline_file = os.path.join(
            "data/contact_matrix_baseline.txt")
        minimum_file = os.path.join(
            "data/contact_matrix_minimum.txt")
        # Build a ContactMatrix from baseline and minimum files
        contact_matrices[0] = ContactMatrix(
            read_mobility_plain(baseline_file),
            read_mobility_plain(minimum_file),
        )
        model.parameters.ContactPatterns.cont_freq_mat = contact_matrices

    return (set_contact_matrices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we will combine the code fragments into one function. This is necessary for our fitting process, but it also simplifies testing everything a lot.
    """)
    return


@app.cell
def _(
    dt,
    num_age_groups,
    osecir,
    set_contact_matrices,
    set_known_parameters,
    set_parameters,
    set_population,
    t0,
    tmax,
):
    def run_simulation(parameters, tmax=tmax):
        # Create model and set parameters
        local_model = osecir.Model(num_age_groups)
        set_population(local_model, parameters)
        set_parameters(local_model, parameters)
        set_known_parameters(local_model)
        set_contact_matrices(local_model)
        # Check that the parameters can not be impossible choices like, for example, negative dwelling times
        local_model.check_constraints()
        result = osecir.simulate(t0, tmax, dt, local_model)
        return {"data": osecir.interpolate_simulation_result(result).as_ndarray()}

    return (run_simulation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fitting Setup

    We decided to use [`pyABC`](https://pyabc.readthedocs.io/en/latest/), a well-established tool and maintained tool for parameter estimation. In combination with MEmilio, it has for example been used in [this publication](https://doi.org/10.1101/2025.09.25.25336633). We will just introduce its features as needed. For advanced setups, like distributed cluster usage, additional settings, visualizations, and examples we refer to the [documentation](https://pyabc.readthedocs.io/en/latest/).

    Here, we first need to import it and load some dependencies.
    """)
    return


@app.cell
def _():
    import os
    import tempfile
    import pyabc
    import pyarrow

    return os, pyabc, tempfile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting up the prior

    Before we can run the inference process, we first of all need a prior (distribution). This is a distribution over possible parameter values from which we sample candidates, simulate the model, and then evaluate the simulations using an objective function.

    `pyABC` has functions that create priors for each parameter based on given distributions. On sampling, this function outputs a dictionary with the parameter names and values. This fits neatly into our previously defined `run_simulation` function - What a lucky coincidence!

    As priors for our parameters, we can use everything that is defined as a `scipy` random distribution. As we do not have any prior knowledge here, we will assume uniform distributions. Note that life is easier if mean values are known or can be guessed. We should mainly take care that we do not accidentally sample implausible values (for example, negative values). In these cases, the model returns a warning. In order to avoid theses problems, priors should be selected carefully.
    """)
    return


@app.cell
def _(pyabc):
    prior = pyabc.Distribution(
        TimeInfectedNoSymptoms0=pyabc.RV("uniform", 0.1, 1.9),
        TransmissionProbabilityOnContact0=pyabc.RV("uniform", 0.01, 0.1),
        RecoveredPerInfectedNoSymptoms0=pyabc.RV("uniform", 0.1, 0.3),
        TimeInfectedNoSymptoms1=pyabc.RV("uniform", 0.1, 1.9),
        TransmissionProbabilityOnContact1=pyabc.RV("uniform", 0.01, 0.1),
        RecoveredPerInfectedNoSymptoms1=pyabc.RV("uniform", 0.1, 0.3),
        TimeInfectedNoSymptoms2=pyabc.RV("uniform", 0.1, 1.9),
        TransmissionProbabilityOnContact2=pyabc.RV("uniform", 0.01, 0.1),
        RecoveredPerInfectedNoSymptoms2=pyabc.RV("uniform", 0.1, 0.3),
        TimeInfectedNoSymptoms3=pyabc.RV("uniform", 0.1, 1.9),
        TransmissionProbabilityOnContact3=pyabc.RV("uniform", 0.01, 0.1),
        RecoveredPerInfectedNoSymptoms3=pyabc.RV("uniform", 0.1, 0.3),
        TimeInfectedNoSymptoms4=pyabc.RV("uniform", 0.1, 1.9),
        TransmissionProbabilityOnContact4=pyabc.RV("uniform", 0.01, 0.1),
        RecoveredPerInfectedNoSymptoms4=pyabc.RV("uniform", 0.1, 0.3),
        TimeInfectedNoSymptoms5=pyabc.RV("uniform", 0.1, 1.9),
        TransmissionProbabilityOnContact5=pyabc.RV("uniform", 0.01, 0.1),
        RecoveredPerInfectedNoSymptoms5=pyabc.RV("uniform", 0.1, 0.3),
        InitiallyExposed=pyabc.RV("uniform", 1, 200)
    )
    return (prior,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can sample from this prior by calling the `rvs` function. If you rerun the next cell, you will always get different results:
    """)
    return


@app.cell
def _(prior):
    prior.rvs()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining the objective function

    The last step before running the fitting is the defintion of an objective (or distance) function. Here, we are given data for the ICU cases and deaths per day for multiple age groups. Thus an obvious choice for the distance function is to calculate the difference between the simulated and the observed numbers per day and adding them up.

    We need a function that takes a `data` dictionary provided by our `run_simulation` function and an `observation` dictionary, given by our input data. As with plotting in the previous tutorials, we have to access the correct columns of our simulation results by indexing as there is no name provided.
    """)
    return


@app.cell
def _(DeathssPerCritical, np, num_age_groups, pyabc, tmax):
    def distance_ICU(simulation, real_data):
        return_value = 0
        for i in range(num_age_groups):
            real_ICU = real_data[f"AgeGroup {i} InfectedCritical"]
            sim_ICU = simulation['data'][10*(i+1)-2]
            return_value += np.sum(np.abs(real_ICU - sim_ICU))
        return return_value / (2 * num_age_groups * tmax)

    def distance_Deaths(simulation, real_data):
        return_value = 0
        for i in range(num_age_groups):
            real_Deaths = real_data[f"AgeGroup {i} Dead"]
            sim_Death = simulation["data"][10*(i+1)]
            return_value += np.sum(DeathssPerCritical[i]
                                   * np.abs(real_Deaths - sim_Death))
        return return_value / (2 * num_age_groups * tmax)

    distance = pyabc.AdaptiveAggregatedDistance(
        [distance_ICU, distance_Deaths], adaptive=False, scale_function=pyabc.distance.mean)
    return (distance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data loading and testing

    Before starting the fitting process, we need to load our data and we should once check that everything works as intended.
    """)
    return


@app.cell
def _(pd):
    observation_data = pd.read_csv("data/cases_2.csv")
    return (observation_data,)


@app.cell
def _(observation_data):
    observation_data
    return


@app.cell
def _(prior, run_simulation):
    example_results = run_simulation(prior.rvs())
    return (example_results,)


@app.cell
def _(example_results):
    example_results
    return


@app.cell
def _(distance, example_results, observation_data):
    distance(example_results, observation_data)
    return


@app.cell
def _(example_results):
    example_results['data'][0, :]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nice, everything seems to work. Then we can go on with running the inference process.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inference process

    With all the previous work done, there are only four more lines of code needed to run the inference process. First, we need to create the fitting object. It is called `ABCSMC` because we perform the fitting using Approximate Bayesian Computation - Sequential Monte Carlo. The object is created by giving it our simulation function, the prior and the distance. We set the population size to 600. This will reduce the chance of numerical instabilites. There are more possible parameters for which we will just use the defaults. The full documentation is available [here](https://pyabc.readthedocs.io/en/latest/api/pyabc.inference.html#pyabc.inference.ABCSMC).

    Then we need to define a database path. `pyabc` stores all simulations in a database. This allows us to take a closer look at them after the inference. However, here we will use a temporary directory to store the database. Once we found that folder, we need to create the database before finally running the inference.
    One important question is for how long to run the inference. Here, we will simply set a `max_nr_populations` of 6. Then the inference is stopped once it has optimized for six iterations. While this is not a very good stopping criteria, we use it here to reduce the run time. You are free to replace that by any other stopping criteria as explained in the [pyabc documentation](https://pyabc.readthedocs.io/en/latest/index.html).
    The inference is automatically parallelized to all available cores. Thus the runtime depends on your machine and may range from a minute up to a 15 minutes.  To test performance, you can start with a `population_size` of 100 or less. If you encounter issues about the population size being `nan`, just try to create the `abc` object again. If it still doesn't work, try it with a bigger population size.
    """)
    return


@app.cell
def _(distance, observation_data, os, prior, pyabc, run_simulation, tempfile):
    abc = pyabc.ABCSMC(run_simulation, prior,
                       distance, population_size=600)
    db_path = "sqlite:///" + os.path.join(tempfile.gettempdir(), "tmp.db")
    abc.new(db_path, observation_data)
    return (abc,)


@app.cell
def _(abc):
    history = abc.run(minimum_epsilon=1)
    return (history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Great, that worked out well. Let's take a look at the evaluation figures by `pyabc`. Here we can see the posterior distributions for the different parameters.Rendering the figure takes a few seconds.
    """)
    return


@app.cell
def _(history, plt, pyabc):
    df, w = history.get_distribution()
    plot = pyabc.visualization.plot_kde_matrix(df, w)
    plt.show()
    return


@app.cell
def _(history, plt, pyabc):
    fig_eval, arr_ax = plt.subplots(1, 3, figsize=(12, 4))

    pyabc.visualization.plot_sample_numbers(history, ax=arr_ax[0])
    pyabc.visualization.plot_epsilons(history, ax=arr_ax[1])
    pyabc.visualization.plot_effective_sample_sizes(history, ax=arr_ax[2])

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Overall, the statistics could be worse. Let's take a look at a simulated trajectory plotted agains the real data. `pyabc` offers us a `plot_data_callback` function which is just a wrapper around plotting functions that we define, but it extracts the simulations out of the `history` for us.
    """)
    return


@app.cell
def _(np, num_age_groups):
    def plot_critical_data(sum_stat, weight, ax, **kwargs):
        for i in range(num_age_groups):
            ax.plot(range(0, 31), sum_stat['data']
                    [10*(i+1)-2, :], color='grey', alpha=0.1)

    def plot_critical_mean(sum_stats, weights, ax, **kwargs):
        for i in range(num_age_groups):
            weights = np.array(weights)
            weights /= weights.sum()
            data = np.array([sum_stat['data'][10*(i+1)-2, :]
                            for sum_stat in sum_stats])
            mean = (data * weights.reshape((-1, 1))).sum(axis=0)
            ax.plot(range(0, 31), mean,
                    color=f"C{i}", label=f"Simulation mean Agegroup {i}")

    return plot_critical_data, plot_critical_mean


@app.cell
def _(
    history,
    mo,
    num_age_groups,
    observation_data,
    plot_critical_data,
    plot_critical_mean,
    plt,
    pyabc,
):
    fig, ax = plt.subplots()
    ax = pyabc.visualization.plot_data_callback(
        history, plot_critical_data, plot_critical_mean, ax=ax)

    for _i in range(num_age_groups):
        plt.scatter(range(
            0, 31), observation_data[f"AgeGroup {_i} InfectedCritical"], color=f"C{_i}", label=f"data Age group {_i}", zorder=2)
    plt.xlabel("Time")
    plt.ylabel("# Cases")
    plt.title("Number of ICU patients")
    plt.legend()
    plt.show()
    mo.vstack([ax])
    return


@app.cell
def _(np, num_age_groups):
    def plot_dead_data(sum_stat, weight, ax, **kwargs):
        for i in range(num_age_groups):
            ax.plot(range(0, 31), sum_stat['data']
                    [10*(i+1), :], color='grey', alpha=0.1)

    def plot_dead_mean(sum_stats, weights, ax, **kwargs):
        for i in range(num_age_groups):
            weights = np.array(weights)
            weights /= weights.sum()
            data = np.array([sum_stat['data'][10*(i+1), :]
                            for sum_stat in sum_stats])
            mean = (data * weights.reshape((-1, 1))).sum(axis=0)
            ax.plot(range(0, 31), mean,
                    color=f"C{i}", label=f"Simulation mean Agegroup {i}")

    return plot_dead_data, plot_dead_mean


@app.cell
def _(
    history,
    num_age_groups,
    observation_data,
    plot_dead_data,
    plot_dead_mean,
    plt,
    pyabc,
):
    fig_dead, ax_dead = plt.subplots()
    ax_dead = pyabc.visualization.plot_data_callback(
        history, plot_dead_data, plot_dead_mean, ax=ax_dead)

    for i in range(num_age_groups):
        plt.scatter(range(
            0, 31), observation_data[f"AgeGroup {i} Dead"], color=f"C{i}", label=f"data Age group {i}", zorder=2)
    plt.xlabel("Time")
    plt.ylabel("# Cases")
    plt.title("Cumulative number of dead patients")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This fit looks okayish! Apparently, we can fit to the data well and with reasonable uncertainty. We could probably also improve by running more generations or bigger population sizes.

    Now we need to report our results back to the RKI because we do not want to be the people responsible for specific NPIs... The discussion has already started and some people claim that the inhabitants of specific German regions already restricted themselfes, making further measures unnecessary. Is that true? To investigate this question, we need to fit a model with regional resolution. That will indeed be done in the next tutorial.
    But for now, let's just plot our posteriors and take a break.
    """)
    return


@app.cell
def _(history, np, plt, prior, pyabc):
    param_names = prior.get_parameter_names()
    _fig, _ax = plt.subplots(4, 5, layout='constrained', figsize=(14, 6))
    _ax = _ax.flatten()
    for _i, param in enumerate(param_names):
        _df, _w = history.get_distribution(m=0, t=history.max_t)
        if param not in _df.columns:
            print(f'parameter in prior but not in history: {param}')
            continue
        pyabc.visualization.plot_kde_1d(
            _df,
            _w,
            x=param,
            # xname = param_names_to_types[param],
            # title=param_names_to_formulas[param],
            xmax=prior[param].distribution.support()[1] if np.isfinite(
                prior[param].distribution.support()[1]) else None,
            xmin=prior[param].distribution.support()[0] if np.isfinite(
                prior[param].distribution.support()[0]) else None,
            ax=_ax[_i],
            label=f"PDF t={history.max_t+1}",
        )
    _ax[-1].set_axis_off()
    _fig.set_constrained_layout(True)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
