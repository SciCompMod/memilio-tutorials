import marimo

__generated_with = "0.20.0"
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

    Today, the German federal health authority RKI announced that the recently observed sharp increase in treatments in intensive care units (ICU) is due to a new lineage of the influenza virus. It was first sequenced at the University Hospital of Cologne and thus called Influenza B/Colognia/314/2026. After it's detection, laboratories in the whole of Germany rushed to test stored samples of recent patients so that we now have a good overview of the death counts and numbers of patients in ICUs.

    The oldest sample found in a deceased person was a 50 year old man from Cologne, who died on 2026-03-02. His family reported that he felt sick shortly after excessively celebrating Karneval on Rose Monday, which was 2026-02-16 this year.

    The RKI today published all available data and asked modellers around the world to estimate parameters and make predictions on the course of the disease.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction

    In this tutorial, we try to infer model parameters based on "real" data. This allows us to make predictions about the course of an epidemic and study sets of potential interventions.

    This is the first notebook in a series of three that introduce the usage of Approximate Bayesian Computation (ABC) with MEmilio. We start with a simple model and progressively improve our fits by approaching the reality more closely, in particular by adding stratification by age groups and multiple regions.

    Note that this is not a tutorial on ABC. For details on ABC, we refer to the tutorials of pyabc and BayesFlow as listed below. Here, we show how to connect these tools with the MEmilio software framework.

    In the first two tutorials, we use the package [pyabc](https://pyabc.readthedocs.io/en/latest/) for likelihood-free inference. While more suitable approaches might be available for the model considered here, we use a simple model to demonstrate the coupling. For more advanced and stochastic models, you only need to replace the model and accept longer run times.
    For the last tutorial, we use [Bayesflow](https://bayesflow.org/main/index.html), a state of the art python library for Bayesian inference with deep learning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Setup

    We will reuse the SECIR-type ODE-based model that was introduced in the previous tutorials. The model already has all the compartments for which we have data available, so we do not need to implement anything from scratch. Let's first import the necessary functions:
    """)
    return


@app.cell
def _():
    import memilio.simulation.osecir as osecir
    from memilio.simulation import AgeGroup, Damping, LogLevel, set_log_level
    # deactivate the log level to avoid warning messages from adaptive step sizing
    set_log_level(LogLevel.Warning)
    return AgeGroup, osecir


@app.cell
def _(mo):
    mo.md(r"""
    Next, we need to define the setting and the model parameters. We will first (i.e., in this notebook) try to fit a simple model with neither age groups nor spatial resolution. This, luckily, reduces our parameter space to just a few unknowns. We already know the total population size of Germany being aproximately 83 million. We set the initial day to 0 (Rose Monday is a good guess here) and ignore seasonality effects for now. The length of the simulation corresponds to the period of day 0 to the date of the last reported case. Previous studies give us good estimates for the contact rates in Germany, which we will just reuse here.
    """)
    return


@app.cell
def _():
    # known parameters:
    total_population = 83000000
    t0 = 0
    tmax = 30
    contact_frequency = 7.95
    return contact_frequency, t0, tmax, total_population


@app.cell
def _(mo):
    mo.md(r"""
    First, we create the model with one age group.
    """)
    return


@app.cell
def _(osecir):
    model = osecir.Model(1)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Instead of fixing the parameters, we will use a dictionary called `parameters`, which serves for sampling random realizations.
    """)
    return


@app.cell
def _(AgeGroup):
    def set_parameters(model, parameters):
        group = AgeGroup(0)
        model.parameters.TimeExposed[group] = parameters["TimeExposed"]
        model.parameters.TimeInfectedNoSymptoms[group] = parameters["TimeInfectedNoSymptoms"]
        model.parameters.TimeInfectedSymptoms[group] = parameters["TimeInfectedSymptoms"]
        model.parameters.TimeInfectedSevere[group] = parameters["TimeInfectedSevere"]
        model.parameters.TimeInfectedCritical[group] = parameters["TimeInfectedCritical"]

        model.parameters.RelativeTransmissionNoSymptoms[group] = parameters["RelativeTransmissionNoSymptoms"]
        model.parameters.TransmissionProbabilityOnContact[group] = parameters["TransmissionProbabilityOnContact"]
        model.parameters.RecoveredPerInfectedNoSymptoms[group] = parameters["RecoveredPerInfectedNoSymptoms"]
        model.parameters.RiskOfInfectionFromSymptomatic[group] = parameters["RiskOfInfectionFromSymptomatic"]
        model.parameters.SeverePerInfectedSymptoms[group] = parameters["SeverePerInfectedSymptoms"]
        model.parameters.CriticalPerSevere[group] = parameters["CriticalPerSevere"]
        model.parameters.DeathsPerCritical[group] = parameters["DeathsPerCritical"]

    return (set_parameters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How should we initialize the population? Even though Influenza B(Colognia) has never been observed before, it is quite likely that there is some type of cross-immunity from other Influenza viruses. Nevertheless, as we do not have data on that, we will first of all assume that the whole population starts off susceptible. On day 0, we then have an unknown number of exposed people that got infected celebrating Karneval. One of them is our first deceased patient. Patient 0, however, likely did recover without being tested and will never be found.
    """)
    return


@app.cell
def _(AgeGroup, osecir, total_population):
    def set_population(model, parameters):
        group = AgeGroup(0)
        model.populations[group, osecir.InfectionState.Exposed] = parameters["InitiallyExposed"]
        model.populations.set_difference_from_total((group, osecir.InfectionState.Susceptible), total_population)

    return (set_population,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we will combine the code fragments into one function. This is necessary for our fitting process, but it also simplifies testing everything a lot.
    """)
    return


@app.cell
def _(contact_frequency, np, osecir, set_parameters, set_population, t0, tmax):
    def run_simulation(parameters, tmax = tmax):
        # Create model and set parameters
        influenza_model = osecir.Model(1)
        set_population(influenza_model, parameters)
        set_parameters(influenza_model, parameters)
        influenza_model.parameters.ContactPatterns.cont_freq_mat[0].baseline = np.ones((1,1)) * contact_frequency
        # Check that the parameters are meaningful, i.e., no negative dwelling times
        influenza_model.check_constraints()

        result = osecir.simulate(t0, tmax, 0.1, influenza_model)
        return {"data": osecir.interpolate_simulation_result(result).as_ndarray()}

    return (run_simulation,)


@app.cell
def _(mo):
    mo.md(r"""
    Why did we choose to return a dictionary instead of just the time series? We will see that in the next chapter.
    """)
    return


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
    ## Setting up the prior distribution

    Before we can run the inference process, we first of all need a prior (distribution). This is a distribution over possible parameter values from which we sample candidates, simulate the model, and then evaluate the simulations using an objective function.

    `pyABC` has functions that create priors for each parameter based on given distributions. On sampling, this function outputs a dictionary with the parameter names and values. This fits neatly into our previously defined `run_simulation` function - What a lucky coincidence!

    As priors for our parameters, we can use everything that is defined as a `scipy` random distribution. As we do not have any prior knowledge here, we will assume uniform distributions. Note that life is easier if mean values are known or can be guessed. We should mainly take care that we do not accidentally sample implausible values (for example, negative values). In these cases, the model returns a warning. In order to avoid theses problems, priors should be selected carefully.
    """)
    return


@app.cell
def _(pyabc):
    prior = pyabc.Distribution(
            TimeExposed = pyabc.RV("uniform", 0.5, 4),
            TimeInfectedNoSymptoms = pyabc.RV("uniform", 0.5, 5),
            TimeInfectedSymptoms = pyabc.RV("uniform", 0.5, 5),
            TimeInfectedSevere = pyabc.RV("uniform", 1, 8),
            TimeInfectedCritical = pyabc.RV("uniform", 2, 10),
            RelativeTransmissionNoSymptoms = pyabc.RV("uniform", 0.01, 0.5),
            TransmissionProbabilityOnContact = pyabc.RV("uniform", 0.01, 0.5),
            RecoveredPerInfectedNoSymptoms = pyabc.RV("uniform", 0.01, 0.5),
            RiskOfInfectionFromSymptomatic = pyabc.RV("uniform", 0.01, 0.9),
            SeverePerInfectedSymptoms = pyabc.RV("uniform", 0.01, 0.5),
            CriticalPerSevere = pyabc.RV("uniform", 0.01, 0.5),
            DeathsPerCritical = pyabc.RV("uniform", 0.01, 0.8),
            InitiallyExposed = pyabc.RV("uniform", 1, 200)
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

    The last step before running the fitting is the defintion of an objective (or distance) function. Here, we are given data for the ICU cases and deaths per day. Thus an obvious choice for the distance function is to calculate the difference between the simulated and the observed numbers per day and adding them up.

    We need a function that takes a `data` dictionary provided by our `run_simulation` function and an `observation` dictionary, given by our input data. As in the plotting sections of the previous tutorials, we have to access the correct columns of our simulation results by indexing as there is no name provided.
    """)
    return


@app.cell
def _(np, osecir, pyabc):
    def distance_ICU(simulation, real_data):
        real_ICU = real_data['Critical']
        sim = simulation['data']
        sim_ICU = sim[1+ int(osecir.InfectionState.InfectedCritical), :]
        return np.sum(np.abs(real_ICU - sim_ICU))

    def distance_Deaths(simulation, real_data):
        real_Deaths = real_data['Deaths']
        sim = simulation['data']
        sim_Death = sim[1 + int(osecir.InfectionState.Dead), :]
        return np.sum(np.abs(real_Deaths - sim_Death))

    distance = pyabc.AdaptiveAggregatedDistance([distance_ICU, distance_Deaths], adaptive=False, scale_function=pyabc.distance.median)
    return (distance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data loading and testing

    Before starting the fitting process, we need to load our data and check once that everything works as intended.
    """)
    return


@app.cell
def _(pd):
    observation_data = pd.read_csv("data/cases_1.csv")
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
    distance(example_results, observation_data, t=-1)
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

    With all the previous work done, there are only four more lines of code needed to run the inference process. First, we need to create the fitting object. It is called `ABCSMC` because we perform the fitting using Approximate Bayesian Computation - Sequential Monte Carlo. The object is created by giving it our simulation function, the prior and the distance. We set the population size to the somewhat arbitrary value of 400. This will reduce the chance of numerical instabilites. There are more possible parameters for which we will just use the defaults. The full documentation is available [here](https://pyabc.readthedocs.io/en/latest/api/pyabc.inference.html#pyabc.inference.ABCSMC).

    Then we need to define a database path. `pyabc` stores all simulations in a database. This allows us to take a closer look at them after the inference. However, here we will use a temporary directory to store the database. Once we have found that folder, we need to create the database before finally running the inference.

    One important question is for how long to run the inference. Here, we will simply set a `minimum_epsilon` of 0.1. Then the inference is stopped once the value of the distance function is below 0.1. For other stopping criteria we refer to the [pyabc documentation](https://pyabc.readthedocs.io/en/latest/index.html).

    The inference is automatically parallelized to all available cores. Thus the runtime depends on your machine and may range from a minute up to 15 minutes. To test performance, you can start with a `population_size` of 100 or less. If you encounter issues about the population size being `nan`, just try to create the `abc` object again. If it still doesn't work, try it with a bigger population size.
    """)
    return


@app.cell
def _(distance, observation_data, os, prior, pyabc, run_simulation, tempfile):
    abc = pyabc.ABCSMC(run_simulation, prior, distance, population_size=400)
    db_path = "sqlite:///" + os.path.join(tempfile.gettempdir(), "tmp.db")
    abc.new(db_path, observation_data)
    return (abc,)


@app.cell
def _(abc):
    history = abc.run(minimum_epsilon=5e-01)
    return (history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Great, that worked out well. Let's take a look at the evaluation figures produced by `pyabc`. Here we can see the posterior distributions for the different parameters. We want to see sharp peaks and narrow tails, indicating good identifiability. Rendering the figure takes a few seconds.
    """)
    return


@app.cell
def _(history, plt, pyabc):
    df, w = history.get_distribution()
    plot = pyabc.visualization.plot_kde_matrix(df, w)
    plt.show()
    return df, w


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
def _(np, osecir):
    def plot_critical_data(sum_stat, weight, ax, **kwargs):
        ax.plot(range(0, 31), sum_stat['data'][1+ int(osecir.InfectionState.InfectedCritical), :], color = 'grey', alpha = 0.1)

    def plot_critical_mean(sum_stats, weights, ax, **kwargs):
        weights = np.array(weights)
        weights /= weights.sum()
        data = np.array([sum_stat['data'][1+ int(osecir.InfectionState.InfectedCritical), :] for sum_stat in sum_stats])
        mean = (data * weights.reshape((-1, 1))).sum(axis=0)
        ax.plot(range(0, 31), mean, color='C2', label = "Simulation mean")

    return plot_critical_data, plot_critical_mean


@app.cell
def _(
    history,
    observation_data,
    plot_critical_data,
    plot_critical_mean,
    plt,
    pyabc,
):
    fig, ax = plt.subplots()
    ax = pyabc.visualization.plot_data_callback(history, plot_critical_data, plot_critical_mean, ax=ax)

    plt.scatter(range(0, 31), observation_data["Critical"], color = "C1", label = "Data", zorder = 2)
    plt.xlabel("Time")
    plt.ylabel("# Cases")
    plt.title("Number of ICU patients")
    plt.legend()
    plt.show()
    return


@app.cell
def _(np, osecir):
    def plot_dead_data(sum_stat, weight, ax, **kwargs):
        ax.plot(range(0, 31), sum_stat['data'][1+ int(osecir.InfectionState.Dead), :], color = 'grey', alpha = 0.1)

    def plot_dead_mean(sum_stats, weights, ax, **kwargs):
        weights = np.array(weights)
        weights /= weights.sum()
        data = np.array([sum_stat['data'][1+ int(osecir.InfectionState.Dead), :] for sum_stat in sum_stats])
        mean = (data * weights.reshape((-1, 1))).sum(axis=0)
        ax.plot(range(0, 31), mean, color='C2', label = "Simulation mean")

    return plot_dead_data, plot_dead_mean


@app.cell
def _(history, observation_data, plot_dead_data, plot_dead_mean, plt, pyabc):
    fig_dead, ax_dead = plt.subplots()
    ax_dead = pyabc.visualization.plot_data_callback(history, plot_dead_data, plot_dead_mean, ax=ax_dead)

    plt.scatter(range(0, 31), observation_data["Deaths"], color = "C1", label = "Data", zorder = 2)
    plt.xlabel("Time")
    plt.ylabel("# Cases")
    plt.title("Cumulative number of dead patients")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This fit looks really good! Apparently, we can fit to the data really well, and our uncertainty is also very low. In our setting, this is not (too) surprising, as ODE models can be fitted very well und our underlying data was also produced by an ODE model (albeit a slightly different one).

    Let's take a look at how we can predict the future numbers in our epidemic. To do that, we return to the parameters of our last population. We already plotted them in the kde plot above; now we will use them for simulations.
    """)
    return


@app.function
def get_params(dataframerow):
    params = dict()
    for key in dataframerow.keys():
        params[key] = dataframerow[key]
    return params


@app.cell
def _(run_simulation):
    def run_simulation_with_params(df):
        params = get_params(df)
        return run_simulation(params, tmax= 60)

    return (run_simulation_with_params,)


@app.cell
def _(df, run_simulation_with_params):
    predictions = df.apply(run_simulation_with_params, axis = 1)
    return (predictions,)


@app.cell
def _(np, observation_data, osecir, plt, predictions, w):
    fig_prog, ax_prog = plt.subplots(1, 2, figsize = (12,4))
    ax_prog[0].scatter(range(0, 31), observation_data["Deaths"], color = "C1", label = "Data")
    for prediction in predictions:
        ax_prog[0].plot(range(61), prediction["data"][1+int(osecir.InfectionState.Dead)], color = "grey", alpha = 0.2)
    weights = np.array(w)
    weights /= weights.sum()
    data = np.array([prediction['data'][1+ int(osecir.InfectionState.Dead), :] for prediction in predictions])
    mean = (data * weights.reshape((-1, 1))).sum(axis=0)
    ax_prog[0].plot(range(0, 61), mean, color='C2', label = "Simulation mean")
    ax_prog[0].legend()
    ax_prog[0].set_title("Number of Deaths")
    ax_prog[0].set_ylabel("Number")
    ax_prog[0].set_xlabel("Time [d]")

    ax_prog[1].scatter(range(0, 31), observation_data["Critical"], color = "C1", label = "Data")
    for prediction in predictions:
        ax_prog[1].plot(range(61), prediction["data"][1+int(osecir.InfectionState.InfectedCritical)], color = "grey", alpha = 0.2)
    weights = np.array(w)
    weights /= weights.sum()
    data = np.array([prediction['data'][1+ int(osecir.InfectionState.InfectedCritical), :] for prediction in predictions])
    mean = (data * weights.reshape((-1, 1))).sum(axis=0)
    ax_prog[1].plot(range(0, 61), mean, color='C2', label = "Simulation mean")
    ax_prog[1].legend()
    ax_prog[1].set_title("Number of ICU cases")
    ax_prog[1].set_ylabel("Number")
    ax_prog[1].set_xlabel("Time [d]")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That does not look good! If the epidemic evolves in the same way that it did for the last 30 days, we will be in very big trouble soon! We should immediately start with interventions. But to make them as efficient as possible, we should take a look at how the disease affects different age groups. Maybe children do not get sick at all and only older people die? Maybe it's the other way round? To answer these questions, we need to fit an age-resolved model. That will be done in the next notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
