import marimo

__generated_with = "0.19.4"
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

    Today, the german federal health authority RKI announced that the recently observed sharp increase in ICU case numbers after acute respiratory infections is due to a new lineage of the influenza virus. It was first sequenced at the university hospital of cologne and thus called Influenza B/Colognia/314/2026. After it's detection, laboratories in the whole of Germany rushed to test stored samples of recent patients s.t. we now have a good overview on the death numbers and numbers of patients in ICUs.

    The oldest sample found was a 80 year old man from Cologne, who died on 2026-02-28. His family reported that he felt sick shortly after excessively celebrating carneval on Rose Monday, which was 2026-02-16 this year.

    The RKI today published all available data and asked for modellers around the world to estimate paramaters and make predictions on the course of the disease.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction

    In this tutorial we will try to infere model parameters based on "real" data. This will allow us to make predictions on the course of an epidemic and suggest appropriate interventions.

    This is the first in a series of three notebooks that will introduce the usage of Approximate Bayesian computation with MEmilio. We will start with a simple model and progressively improve our fits by adding different age groups and multiple regions.

    This is not a tutorial on Approximate Bayesian Computation. For that, we refer to the tutorials of the software we use and the literature. We will just show how to use these tools together with the MEmilio software framework.

    In the first two tutorials we will use the package [pyabc](https://pyabc.readthedocs.io/en/latest/). It is a package for likelihood-free inference. This is, of course, a bit of overkill for a differentiable ODE model. However, due to the extremely short runtime of the ODE models, they are very well fitted for a tutorial. If you want to use the same methods for our stochastic models, you only need to replace the model (and wait for a bit longer).
    For the last tutorial we will use [Bayesflow](https://bayesflow.org/main/index.html), a state of the art python library for Bayesian inference with deep learning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Setup

    We will reuse the model that was already introduced in the last tutorials. It already has all the compartments for which we have data available, so we don't need to implement anything from scratch. Let's first import the necessary functions:
    """)
    return


@app.cell
def _():
    import memilio.simulation.osecir as osecir
    from memilio.simulation import AgeGroup, Damping, LogLevel, set_log_level
    set_log_level(LogLevel.Off)
    return AgeGroup, osecir


@app.cell
def _(mo):
    mo.md(r"""
    Next, we need a lot of parameters. We will first (i.e. in this notebook) try to fit a simple model without age groups or any spatial resolution. This, luckily, reduces our parameter space to just a few unknowns. The parameters that we already know are the total population of Germany, day 0 (rose monday is a good guess here) and thus the length of the simulation (the last reported day in the data). We should also choose an initial time step, but the ODE solver will use adaptive time steps later on. Previous studys give us good values for the contact rates in Germany that we will just reuse here.
    """)
    return


@app.cell
def _():
    # known parameters:
    total_population = 83000000
    t0 = 0
    tmax = 30
    dt = 0.1
    contact_frequency = 7.95
    return contact_frequency, dt, t0, tmax, total_population


@app.cell
def _(mo):
    mo.md(r"""
    Our unknown parameters should only be set by other variables, so we can try out different combinations. To simplify this a little bit, we will use a dictionary that we call `parameters`. But first, we need to create the model. As mentioned before, we will only use one sociodemographic group here.
    """)
    return


@app.cell
def _(AgeGroup, osecir):
    model = osecir.Model(1)

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
    What should we do with the start population? Even though Influenza B(Colognia) has never been observed before, it is quite likely that there is some type of cross-immunity from other Influenza viruses. Nevertheless, as we don't have data on that, we will first of all assume that the whole population starts of susceptible. On day 0, we then have an unknown number of exposed people that got infected celebrating carneval. One of them is our first dead patient. Patient 0, however, likely did recover without being tested and will never be found.
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
    Now we will combine the code into one function. This is necessary for our fitting process, but it also simplifies testing everything a lot.
    """)
    return


@app.cell
def _(
    contact_frequency,
    dt,
    np,
    osecir,
    set_parameters,
    set_population,
    t0,
    tmax,
):
    def run_simulation(parameters, tmax = tmax):
        # Create model and set parameters
        local_model = osecir.Model(1)
        set_population(local_model, parameters)
        set_parameters(local_model, parameters)
        local_model.parameters.ContactPatterns.cont_freq_mat[0].baseline = np.ones((1,1)) * contact_frequency
        # Check that the parameters can not be impossible choices like, for example, negative dwelling times
        local_model.apply_constraints()
    
        result = osecir.simulate(t0, tmax, dt, local_model)
        return {"data": osecir.interpolate_simulation_result(result).as_ndarray()}
    return (run_simulation,)


@app.cell
def _(mo):
    mo.md(r"""
    Why did we choose to return a dictionary instead of just the time series? We will see that in the next chapter:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fitting Setup

    We decided to use [`pyabc`](https://pyabc.readthedocs.io/en/latest/) for the parameter estimation. It is a well-established tool and maintained by collegues of us, so we get good support😇. In combination with MEmilio, it has for example been used in [this publication](https://doi.org/10.1101/2025.09.25.25336633). We will just introduce its features as needed. For advanced setups, like distributed cluster usage, additional settings, visualizations and examples we refer to the [documentation](https://pyabc.readthedocs.io/en/latest/).

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

    Before we can run the inference process, we first of all need a prior. This is a function that provides us with parameter estimates which we then use for simulations which in turn are then evaluated using an objective function.

    `pyabc` has a function that creates a prior based on given prior distributions for all defined parameters. On sampling, this function outputs a dictionary with the parameter names and values. This fits neatly into our previously defined `run_simulation` function - What a lucky coincidence!

    As priors for our parameters we can use everything that is defined as a scipy random distribution. As we don't have any prior knowledge here (for example, life is easier if mean values are known or guessed), we will assume uniform distributions. We should mainly take care that we do not accidently sample implausible values (for example, negative values). This would be catched by our model and "corrected", but then we would simulate with different values than the optimization algorithm beliefs.
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

    We need a function that takes a `data` dictionary provided by our `run_simulation` function and an `observation` dictionary, given by our input data. As with plotting in the previous tutorials, we have to access the correct columns of our simulation results by indexing as there is no name provided.
    """)
    return


@app.cell
def _(np, osecir):
    def distance_function(simulation, real_data):
        real_ICU = real_data['Critical']
        real_Deaths = real_data['Deaths']
        sim = simulation['data']
        sim_ICU = sim[1+ int(osecir.InfectionState.InfectedCritical), :]
        sim_Death = sim[1 + int(osecir.InfectionState.Dead), :]
        return (np.sum(np.abs(real_ICU - sim_ICU)) + np.sum(np.abs(real_Deaths - sim_Death))) / 1000
    return (distance_function,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data loading and testing

    Before starting the fitting process, we need to load our data and we should once check that everything works as intended.
    """)
    return


@app.cell
def _(pd):
    observation_data = pd.read_csv("cases_1.csv")
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
def _(distance_function, example_results, observation_data):
    distance_function(example_results, observation_data)
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

    With all the previous work done, there are only four more lines of code needed to run the inference process. First, we need to create the fitting object. It is called `ABCSMC` because we perform the fitting using Approximate Bayesian Computation -Sequential Monte Carlo. The object is created by giving it our simulation function, the prior and the distance. We set the population size to 300. This will reduce the chance of numerical instabilites. There are more possible parameters for which we will just use the defaults. The full documentation is available [here](https://pyabc.readthedocs.io/en/latest/api/pyabc.inference.html#pyabc.inference.ABCSMC).

    Then we need to define a database path. `pyabc` stores all simulations in a database. This allows us to take a closer look at them after the inference. However, here we will use a temporary directory to store the database. Once we found that folder, we need to create the database before finally running the inference.

    The inference is automatically parallelized to all available cores. Thus the runtime depends on your machine and may range from a minute up to a long time. You can first of all try to use a `population_size` of 100 or less to check out the runtime. If you encounter issues about the population size being `nan`, just try to create the `abc` object again. If it still doesn't work, try it with a bigger population size.
    """)
    return


@app.cell
def _(
    distance_function,
    observation_data,
    os,
    prior,
    pyabc,
    run_simulation,
    tempfile,
):
    abc = pyabc.ABCSMC(run_simulation, prior, distance_function, population_size=400)
    db_path = "sqlite:///" + os.path.join(tempfile.gettempdir(), "tmp.db")
    abc.new(db_path, observation_data)
    return (abc,)


@app.cell
def _(abc):
    history = abc.run(max_nr_populations = 2, minimum_epsilon=0.05)
    return (history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Great, that worked out well. Let's take a look at the evaluation figures by `pyabc`. Here we can see the posterior distributions for the different parameters. We would love to see sharp peaks and narrow tails which would imply a very good identifiability. Rendering the figure takes a few seconds.
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

    plt.scatter(range(0, 31), observation_data["Critical"], color = "C1", label = "Data")
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

    plt.scatter(range(0, 31), observation_data["Deaths"], color = "C1", label = "Data")
    plt.xlabel("Time")
    plt.ylabel("# Cases")
    plt.title("Cumulative number of dead patients")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This fit looks really good! Apparently, we can fit to the data really well and also our uncertainty is very low. In our setting, this is not (too) surprising, as ODE models can be fitted very well und our underlying data was also produced by a (though slightly different) ODE model.

    Let's take a look at how we can predict the future numbers in our epidemic. To do that, we return to the parameters of our last population. We already plotted them in the kde plot above, now we will use them for simulations.
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
