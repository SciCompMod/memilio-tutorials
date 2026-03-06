import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Breaking news 📺

    Today, the German federal health authority RKI announced that the recently observed sharp increase in ICU case numbers following acute respiratory infections is caused by a new lineage of the influenza virus. It was first sequenced at the University Hospital of Cologne and has consequently been named *Influenza B/Colognia/314/2026*. Following its detection, laboratories across Germany rushed to test stored samples from recent patients, giving us a reasonably complete picture of ICU admissions and deaths to date.

    The oldest sample identified came from an 80-year-old man from Cologne who died on 2026-02-29. His family reported that he first felt ill shortly after excessively celebrating Carnival on Rose Monday, which fell on 2026-02-16 this year.

    The RKI has today published all available data and called on modellers worldwide to estimate disease parameters and forecast the further course of the outbreak.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction

    In this notebook we tackle a real-world modelling task: inferring epidemic parameters from observed case data in order to characterise the ongoing outbreak and support decision-making. Concretely, we want to estimate when and how strongly contact-reducing interventions took effect in each of five German regions, using only the reported ICU and death counts.

    This is the third in a series of three notebooks introducing Approximate Bayesian Computation with MEmilio. In the first two notebooks we calibrated simple single-region compartmental models. Here we extend the approach to a **metapopulation model** that resolves spatial heterogeneity across Germany.

    We use [BayesFlow](https://bayesflow.org/main/index.html), a Python library for simulation-based inference with deep learning, as our inference method. This notebook is not a general tutorial on Neural Parameter Estimation with BayesFlow — for that we refer to the BayesFlow documentation and the primary literature. Our focus is on showing how MEmilio and BayesFlow work together.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Setup

    ## Spatial structure: the Kleeblatt regions

    For the spatial resolution we adopt the structure of the *Kleeblatt-Mechanismus*, Germany's framework for balancing critical care capacity across regions during health emergencies. It divides Germany into five large regional units (*Kleeblätter*), each comprising one or more federal states. In our model each Kleeblatt is one spatial node. Inter-regional population movement is represented using commuter flow data from the German Federal Employment Agency (*Bundesagentur für Arbeit*), aggregated to the Kleeblatt level.

    ### Infection dynamics: OSECIR

    Within each region, infection dynamics follow the OSECIR model introduced in the previous notebooks. We begin by importing the required modules and silencing MEmilio's console output.
    """)
    return


@app.cell
def _():
    import memilio.simulation as mio
    import memilio.simulation.osecir as osecir
    from memilio.simulation.osecir import Model, interpolate_simulation_result

    # Suppress verbose MEmilio logs
    mio.set_log_level(mio.LogLevel.Error)
    return Model, mio, osecir


@app.cell
def _(mo):
    mo.md(r"""
    We simulate a period of 30 days, matching the window of available case data:
    """)
    return


@app.cell
def _():
    # Simulation time settings:
    t0 = 0
    tmax = 30
    return t0, tmax


@app.cell
def _(mo):
    mo.md(r"""
    ### Epidemic parameters

    Most model parameters are assumed to be **age-dependent but spatially uniform** — that is, the same across all five regions. These include the mean time spent in each disease compartment and the transition probabilities between them. They are treated as fixed and known, not inferred. We set them with the function below.
    """)
    return


@app.cell
def _(mio):
    def set_covid_parameters(model):
        """
        Set fixed epidemic parameters for all 6 age groups.
        These are NOT inferred — they are considered known.
        """
        # ----- Transition times (days) -----
        for ag, val in enumerate([2.5, 2.3, 2.1, 1.9, 1.7, 1.5]):
            model.parameters.TimeExposed[mio.AgeGroup(ag)] = val

        for ag, val in enumerate([1., 1., 1., 1.5, 1.5, 1.5]):
            model.parameters.TimeInfectedNoSymptoms[mio.AgeGroup(ag)] = val

        for ag, val in enumerate([3.4, 3.8, 4., 4.2, 4.6, 5.]):
            model.parameters.TimeInfectedSymptoms[mio.AgeGroup(ag)] = val

        for ag, val in enumerate([3.6, 3.8, 4., 4.2, 4.4, 4.4]):
            model.parameters.TimeInfectedSevere[mio.AgeGroup(ag)] = val

        for ag, val in enumerate([2., 2., 2.4, 2.4, 3., 3.5]):
            model.parameters.TimeInfectedCritical[mio.AgeGroup(ag)] = val

        # ----- Transition probabilities -----
        for ag, val in enumerate([0.40, 0.35, 0.30, 0.30, 0.25, 0.20]):
            model.parameters.RecoveredPerInfectedNoSymptoms[mio.AgeGroup(
                ag)] = val

        for ag, val in enumerate([0.025, 0.05, 0.10, 0.20, 0.24, 0.25]):
            model.parameters.SeverePerInfectedSymptoms[mio.AgeGroup(ag)] = val

        for ag, val in enumerate([0.40, 0.15, 0.05, 0.15, 0.25, 0.40]):
            model.parameters.CriticalPerSevere[mio.AgeGroup(ag)] = val

        for ag, val in enumerate([0.25, 0.20, 0.15, 0.20, 0.30, 0.40]):
            model.parameters.DeathsPerCritical[mio.AgeGroup(ag)] = val

        # ----- Transmission probabilities -----
        for ag, val in enumerate([0.10, 0.10, 0.10, 0.08, 0.08, 0.05]):
            model.parameters.TransmissionProbabilityOnContact[mio.AgeGroup(
                ag)] = val

        # Relative transmission for asymptomatic and symptomatic cases
        for ag in range(6):
            model.parameters.RelativeTransmissionNoSymptoms[mio.AgeGroup(
                ag)] = 1.
            model.parameters.RiskOfInfectionFromSymptomatic[mio.AgeGroup(
                ag)] = 1.

    return (set_covid_parameters,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Contact matrices and interventions

    Age-structured contact rates are encoded in **contact matrices**. We load baseline contacts and a minimum contact level from text files.

    **Dampings** represent non-pharmaceutical interventions such as stay-at-home orders or school closures. For simplicity we assume that a damping applies uniformly across all age groups, multiplying the contact matrix by a scalar $d \in [0, 1]$:

    - $d = 0$: no intervention, baseline contacts unchanged
    - $d = 1$: contacts reduced to the specified minimum (full lockdown)

    The damping is activated at time `damping_start` and held constant thereafter.

    These two quantities — `damping_start` and `damping_values` — are the **parameters we want to infer**, one value per region, giving $2 \times 5 = 10$ inference targets in total.
    """)
    return


@app.cell
def _(mio, np, os):
    def set_contact_matrices(model, damping_start, damping_value):
        """
        Load baseline contact matrices and apply a time-dependent
        damping starting at `damping_start`.

        Args:
            model:          MEmilio OSECIR model instance
            damping_start:  Day on which the damping begins
            damping_value:  Strength of the contact reduction in [0, 1]
        """
        contact_matrices = mio.ContactMatrixGroup(1, 6)
        baseline_file = os.path.join(
            "data/contact_matrix_baseline.txt")
        minimum_file = os.path.join(
            "data/contact_matrix_minimum.txt")

        # Build a ContactMatrix from baseline and minimum files
        contact_matrices[0] = mio.ContactMatrix(
            mio.read_mobility_plain(baseline_file),
            mio.read_mobility_plain(minimum_file),
        )

        # Add a single damping event: uniform across all age groups
        contact_matrices[0].add_damping(mio.Damping(
            coeffs=np.full((6, 6), damping_value),
            t=damping_start
        ))

        model.parameters.ContactPatterns.cont_freq_mat = contact_matrices

    return (set_contact_matrices,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Initial populations

    Each of the five Kleeblatt regions is initialised with its approximate age-stratified population, drawn from German census data. All individuals start as susceptible — except in Region 3 (North-Rhine Westphalia), where we seed the outbreak with **100 exposed individuals in age group 2 (15–34 years)**.

    Note that only the **initial conditions** differ between regions. The epidemic parameters and contact structure (except for region-dependent dampings) are otherwise identical.
    """)
    return


@app.cell
def _(mio, osecir, t0):
    def set_populations(graph, model):
        """
        Add 5 nodes to the mobility graph, each with region-specific
        initial susceptible populations. Region 3 seeds the outbreak
        with 100 exposed individuals in age group 2.
        """
        # Population data: [AG0, AG1, AG2, AG3, AG4, AG5] per region (approx. German states)
        susceptible_pops = [
            [660000,  1420000, 3360000, 5050000, 3240000, 1370000],  # Region 1
            [560000,  1350000, 3020000, 4880000, 3270000, 1440000],  # Region 2
            [820000,  1730000, 4140000, 6040000,
                3770000, 1530000],  # Region 3 (seeded)
            [1040000, 2170000, 5200000, 7600000, 4730000, 1930000],  # Region 4
            [620000,  1250000, 3040000, 4510000, 2710000, 1120000],  # Region 5
        ]

        for node_id, pops in enumerate(susceptible_pops, start=1):
            for ag, pop in enumerate(pops):
                model.populations[mio.AgeGroup(ag),
                                  osecir.InfectionState.Susceptible] = float(pop)
                model.populations[mio.AgeGroup(ag),
                                  osecir.InfectionState.Exposed] = 0.0

            # Seed outbreak: region 3 (index 2) has 100 exposed in age group 2
            if node_id == 3:
                model.populations[mio.AgeGroup(2),
                                  osecir.InfectionState.Susceptible].value -= 100
                model.populations[mio.AgeGroup(2),
                                  osecir.InfectionState.Exposed].value = 100

            graph.add_node(id=node_id, model=model, t0=t0)

        return graph

    return (set_populations,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Mobility between regions

    Inter-regional population movement is represented as a directed weighted graph. Each edge carries a `MobilityParameters` vector specifying the daily fraction of each compartment that commutes between the two connected regions. The coefficients are derived from a commuter flow matrix and normalised by the source region's total population. Deceased individuals are excluded from mobility.
    """)
    return


@app.cell
def _(mio, np, osecir):
    def set_mobility(graph):
        """
        Load a mobility matrix and add directed edges to the graph.
        Mobility is expressed as fraction of population commuting per day.
        Dead individuals are excluded from mobility.
        """
        mobility_matrix = np.loadtxt("data/mobility_matrix.txt")
        # Number of state variables per node (age groups × compartments)
        num_groups = graph.get_node(0).property.model.populations.numel()

        for i in range(graph.num_nodes):
            for j in range(i + 1, graph.num_nodes):
                total_i = graph.get_node(
                    i).property.model.populations.get_total()
                total_j = graph.get_node(
                    j).property.model.populations.get_total()

                coeff_ij = (mobility_matrix[i, j] /
                            total_i) * np.ones(num_groups)
                coeff_ji = (mobility_matrix[j, i] /
                            total_j) * np.ones(num_groups)

                # Dead compartment does not contribute to mobility
                coeff_ij[osecir.InfectionState.Dead] = 0.
                coeff_ji[osecir.InfectionState.Dead] = 0.

                graph.add_edge(i, j, mio.MobilityParameters(coeff_ij))
                graph.add_edge(j, i, mio.MobilityParameters(coeff_ji))

        return graph

    return (set_mobility,)


@app.cell
def _(mo):
    mo.md(r"""
    ### The simulator

    The function below is the **core forward model** that BayesFlow will call repeatedly during training. Given a set of intervention parameters it assembles the full metapopulation model, runs the ODE solver, and returns the simulated observables.

    The outputs are the **Critical (ICU)** and **Dead** compartment counts for each of the six age groups, extracted at daily resolution for all five regions. This gives, per region, an array of shape `(31, 12)` — 31 time points (days 0–30) and 12 compartment-age combinations (6 age groups × 2 outcomes).
    """)
    return


@app.cell
def _(
    Model,
    np,
    osecir,
    set_contact_matrices,
    set_covid_parameters,
    set_mobility,
    set_populations,
    t0,
    tmax,
):
    compartment_indices = np.sort(
        [10 * i - 3 for i in range(1, 7)] +   # Critical
        [10 * i - 1 for i in range(1, 7)]      # Dead
    )

    def run_simulation(damping_start, damping_values):
        """
        Run a multi-region OSECIR simulation and return Critical + Dead
        compartments for all age groups and regions.

        Args:
            damping_start:  array of shape (5,) — intervention start day per region
            damping_values: array of shape (5,) — intervention strength per region

        Returns:
            dict with keys 'region0' ... 'region4', each an array of shape
            (time_steps+1, 12):  6 age groups × [Critical, Dead]
        """
        graph = osecir.MobilityGraph()
        model = Model(6)  # 6 age groups

        set_covid_parameters(model)
        model.check_constraints()

        graph = set_populations(graph, model)
        graph = set_mobility(graph)

        # Apply per-region contact dampings (the parameters we infer)
        for node_idx in range(graph.num_nodes):
            node = graph.get_node(node_idx)
            set_contact_matrices(
                node.property.model,
                damping_start[node_idx],
                damping_values[node_idx],
            )

        # Run the ODE-based simulation
        sim = osecir.MobilitySimulation(graph, t0, dt=0.5)
        sim.advance(tmax)

        # Extract and return selected compartments per region
        results = {}
        for node_idx in range(sim.graph.num_nodes):
            node = sim.graph.get_node(node_idx)
            # interpolate_simulation_result resamples to integer time points
            full_result = np.array(
                osecir.interpolate_simulation_result(node.property.result)
            )
            results[f'region{node_idx}'] = np.round(
                full_result[:, compartment_indices]
            )

        return results

    return (run_simulation,)


@app.cell
def _(mo):
    mo.md(r"""
    # Simulation-Based Inference with BayesFlow

    Classical Bayesian inference requires an explicit likelihood $p(y \mid \theta)$. For mechanistic simulators like OSECIR this likelihood is analytically intractable. **Simulation-based inference (SBI)** sidesteps this problem: instead of evaluating the likelihood directly, we train a neural network to learn the posterior $p(\theta \mid y)$ from many simulator runs.

    BayesFlow implements a particularly efficient variant called **amortized inference**. The key idea is that training is done once, after which posterior samples for *any* observed dataset can be drawn in milliseconds — no MCMC required.

    The training loop is straightforward:
    1. Draw parameters $\theta \sim p(\theta)$ from a prior.
    2. Simulate data $y \sim p(y \mid \theta)$ using MEmilio.
    3. Train a neural network to approximate $p(\theta \mid y)$.

    Once trained, inference on the actual outbreak data is a single forward pass through the network.
    """)
    return


@app.cell
def _(mio):
    import os
    # Must be set before importing BayesFlow/Keras
    os.environ["KERAS_BACKEND"] = "tensorflow"

    import bayesflow as bf

    # Suppress verbose MEmilio logs
    mio.set_log_level(mio.LogLevel.Error)

    print("All imports successful!")
    print(f"BayesFlow version: {bf.__version__}")
    return bf, os


@app.cell
def _(mo):
    mo.md(r"""
    ## Prior distribution

    The prior encodes our beliefs about the parameters *before* seeing any data. BayesFlow expects the prior as a function that returns a dictionary of parameter arrays; each call produces one independent draw.

    We use **uniform priors** over the plausible ranges — a non-informative choice that lets the data speak for itself:

    - `damping_start ~ Uniform(0, 30)`: the intervention may have started on any day within our observation window.
    - `damping_values ~ Uniform(0, 1)`: the intervention strength is unconstrained between no effect and full lockdown.

    Both quantities are vectors of length 5, one entry per region.
    """)
    return


@app.cell
def _(np):
    bounds = {
        "damping_start":  (0., 30.),
        "damping_values": (0., 1.),
    }

    def prior():
        """
        Sample one set of parameters for the 5-region simulation.

        Returns:
            dict with:
                'damping_start':  np.ndarray, shape (5,)
                'damping_values': np.ndarray, shape (5,)
        """
        return {
            "damping_start":  np.random.uniform(*bounds["damping_start"],  size=5),
            "damping_values": np.random.uniform(*bounds["damping_values"], size=5),
        }

    # Quick sanity check
    sample = prior()
    print("Prior sample:")
    print(f"  damping_start  = {sample['damping_start'].round(2)}")
    print(f"  damping_values = {sample['damping_values'].round(2)}")
    return bounds, prior


@app.cell
def _(mo):
    mo.md(r"""
    ## Data adapter

    The raw simulator output needs to be transformed before it can be fed to the neural networks. BayesFlow's `Adapter` object defines this transformation pipeline as a sequence of named steps.
    """)
    return


@app.cell
def _(bf, bounds):
    adapter = (
        bf.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        # Constrain neural network predictions of a data variable to specified bounds
        .constrain(
            "damping_start",
            lower=bounds["damping_start"][0],
            upper=bounds["damping_start"][1],
        )
        .constrain(
            "damping_values",
            lower=bounds["damping_values"][0],
            upper=bounds["damping_values"][1],
        )
        # Concatenate the two parameter vectors into one inference target
        .concatenate(
            ["damping_values", "damping_start"],
            into="inference_variables",
            axis=-1,
        )
        # Concatenate all region outputs into one summary input tensor
        .concatenate(
            [f"region{i}" for i in range(5)],
            into="summary_variables",
            axis=-1,
        )
    )
    return (adapter,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Neural network architecture

    BayesFlow uses two networks in tandem.

    The **summary network** (`TimeSeriesNetwork`) acts as a learned data compressor. It takes the high-dimensional multi-region time series as input and produces a fixed-size embedding vector. This embedding plays the role of a sufficient statistic — but unlike hand-crafted summaries, it is optimised jointly with the inference network to compress the data in a way that is useful for posterior estimation.

    The **inference network** (`CouplingFlow`) is a normalising flow that learns the mapping from the summary embedding to samples from the approximate posterior $q(\theta \mid s(y))$. It consists of alternating affine coupling layers and can represent complex, multi-modal distributions.

    For more details we refer to the BayesFlow documentation.
    """)
    return


@app.cell
def _(bf):
    # Summary network: compress the multi-region time series
    summary_network = bf.networks.TimeSeriesNetwork(summary_dim=20)

    # Inference network: normalising flow over the posterior
    inference_network = bf.networks.CouplingFlow()
    return inference_network, summary_network


@app.cell
def _(mo):
    mo.md(r"""
    ## Assembling the workflow

    `bf.BasicWorkflow` ties the simulator, adapter, and networks into a single trainable object.

    We verify the pipeline with a small batch before committing to a full training run. The expected shapes are:
    - `summary_variables`: `(4, 31, 60)` — 4 samples, 31 daily time points, 60 compartment-region combinations
    - `inference_variables`: `(4, 10)` — 4 samples, 10 parameters (5 damping values + 5 damping starts)
    """)
    return


@app.cell
def _(adapter, bf, inference_network, prior, run_simulation, summary_network):
    simulator = bf.make_simulator([prior, run_simulation])

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_network,
        inference_network=inference_network,
        standardize="all",   # Normalise inputs and targets
    )

    # Verify shapes with a small batch
    test_batch = workflow.adapter(simulator.sample(batch_size=4))
    print("Shape check with batch_size=4:")
    print(f"  summary_variables  : {test_batch['summary_variables'].shape}")
    print(f"  inference_variables: {test_batch['inference_variables'].shape}")
    print()
    print("Expected:")
    print("  summary_variables  : (4, 31, 60)  — 31 time × 5 regions × 12 compartments")
    print("  inference_variables: (4, 10)       — 5 damping_values + 5 damping_starts")
    return (workflow,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Training

    We train online for 10 epochs with a batch size of 64. The trained model is saved so it can be reloaded without retraining.
    """)
    return


@app.cell
def _(mo, workflow):
    mo.md("Running training... (this may take a few minutes)")

    history = workflow.fit_online(
        epochs=15,
        batch_size=64,
        validation_data=50,  # Simulate 50 validation samples per epoch
    )

    # Save the trained approximator for later reuse
    workflow.approximator.save(filepath="memilio_bayesflow_model.keras")
    print("Training complete. Model saved to 'memilio_bayesflow_model.keras'.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Inference on the Outbreak Data

    With a trained and validated approximator in hand, we can now turn to the actual outbreak data. We load the observed ICU and death counts and pass them to the workflow. Posterior samples are drawn in a single forward pass — this is the payoff of amortized inference: no matter how long training took, each new inference is essentially free.

    The posterior samples are then fed back into the simulator to produce a **posterior predictive distribution**: an ensemble of epidemic trajectories consistent with both the model and the observed data. This allows us to assess model fit and quantify forecast uncertainty.
    """)
    return


@app.cell
def _(np, pd):
    def load_data():
        """
        Load observed case data and reshape into BayesFlow-compatible format.

        Expects 'cases_3.csv' with columns ordered as:
            [region0_comp0, region0_comp2, region1_comp0, ..., region4_comp2]

        Returns:
            conditions: dict mapping 'region{i}' → array of shape (1, T, 12)
                        (the '1' is a batch dimension for workflow.sample)
            data:       np.ndarray of shape (T, 12, 5)
        """
        df = pd.read_csv("cases_3.csv").to_numpy()  # shape: (T, 60)

        data = np.zeros((df.shape[0], df.shape[1] // 5, 5))
        conditions = {}

        for i in range(5):
            region_data = df[:, i * 12:(i + 1) * 12]  # (T, 12)
            data[:, :, i] = region_data
            # Add batch dimension: (1, T, 12)
            conditions[f"region{i}"] = region_data[np.newaxis, ...]

        return conditions, data

    return (load_data,)


@app.cell
def _(load_data, np, run_simulation, workflow):
    conditions, observed_data = load_data()

    num_samples = 50

    # Draw posterior samples — this is near-instantaneous after training!
    posterior_samples = workflow.sample(
        conditions=conditions,
        num_samples=num_samples,
    )

    print("Posterior samples drawn!")
    print(
        f"  damping_start  shape: {posterior_samples['damping_start'].shape}")
    print(
        f"  damping_values shape: {posterior_samples['damping_values'].shape}")
    print()
    print("Shape explanation: (1, num_samples, 5)")
    print("  1 = one observed dataset, num_samples = 50 posterior draws, 5 = regions")

    # Run the simulator forward for each posterior sample (posterior predictive check)
    predictive_results = []
    for _i in range(num_samples):
        _result = run_simulation(
            damping_start=posterior_samples["damping_start"][0, _i],
            damping_values=posterior_samples["damping_values"][0, _i],
        )
        # Add sample and singleton region axes
        for _k in _result:
            _result[_k] = _result[_k][np.newaxis, ..., np.newaxis]
        predictive_results.append(_result)

    # Merge all samples into a single dict
    predictive_combined = {}
    for _d in predictive_results:
        if predictive_combined:
            for _k in _d:
                predictive_combined[_k] = np.concatenate(
                    [predictive_combined[_k], _d[_k]], axis=0
                )
        else:
            predictive_combined = _d

    # Stack into (num_samples, T, 12, 5)
    simulations = np.zeros((num_samples, 31, 12, 5))
    for _i in range(num_samples):
        simulations[_i] = np.concatenate(
            [predictive_combined[f"region{_r}"][_i] for _r in range(5)], axis=-1
        )
    return observed_data, posterior_samples, simulations


@app.cell
def _(mo):
    mo.md(r"""
    # Results

    We visualise the posterior predictive distribution in two complementary ways.

    The first plot shows ICU and death counts **aggregated across all regions**, broken down by age group. Shaded bands represent 50%, 90%, and 95% credible intervals; observed data points are shown as black crosses.

    The second and third plots show a **per-region breakdown** — one panel per region (rows) and age group (columns) — for Critical cases and Deaths respectively. This finer view reveals whether the model fits each region individually and can highlight spatial heterogeneity in the outbreak dynamics.
    """)
    return


@app.cell
def _(np, plt):
    def plot_aggregated_over_regions(
        data,
        region_agg=np.sum,
        true_data=None,
        label=None,
        color="steelblue",
        ax=None
    ):
        """
        Plot posterior predictive distribution aggregated over all regions.

        Args:
            data:        np.ndarray, shape (samples, time_points, regions)
            region_agg:  aggregation function over regions (default: sum)
            true_data:   np.ndarray, shape (time_points, regions) — observed data
            label:       plot title / legend label
            color:       line/fill color
        """
        if data.ndim != 3:
            raise ValueError(
                "data must have shape (samples, time_points, regions)")

        agg = region_agg(data, axis=-1)  # (samples, time_points)

        qs_50 = np.quantile(agg, q=[0.25, 0.75], axis=0)
        qs_90 = np.quantile(agg, q=[0.05, 0.95], axis=0)
        qs_95 = np.quantile(agg, q=[0.025, 0.975], axis=0)
        median = np.median(agg, axis=0)

        x = np.arange(median.shape[0])

        ax.plot(x, median, lw=2.5, label=label or "Posterior median", color=color)
        ax.fill_between(x, qs_50[0], qs_50[1], alpha=0.6, linewidth=0,
                        color=color, label="50% CI")
        ax.fill_between(x, qs_90[0], qs_90[1], alpha=0.35, linewidth=0,
                        color=color, label="90% CI")
        ax.fill_between(x, qs_95[0], qs_95[1], alpha=0.15, linewidth=0,
                        color=color, label="95% CI")

        if true_data is not None:
            true_vals = region_agg(true_data, axis=-1)
            ax.scatter(x, true_vals, color="black", label="Observed data",
                       marker="x", zorder=3, s=40)

        ax.set_xlabel("Day", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(label or "Aggregated Posterior Predictive", fontsize=13)
        ax.legend(fontsize=10)

    def plot_region_fit(data, region, true_data=None, ax=None, label=None, color="steelblue"):
        """
        Plot posterior predictive for a single region.

        Args:
            data:     np.ndarray, shape (samples, time_points, regions)
            region:   integer region index
            true_data: np.ndarray, shape (time_points, regions)
            ax:       matplotlib Axes (creates new figure if None)
        """
        if data.ndim != 3:
            raise ValueError(
                "data must have shape (samples, time_points, regions)")

        vals = data[:, :, region]  # (samples, time_points)
        qs_80 = np.quantile(vals, q=[0.10, 0.90], axis=0)
        qs_95 = np.quantile(vals, q=[0.025, 0.975], axis=0)
        med = np.median(vals, axis=0)

        x = np.arange(med.shape[0])
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(x, med, lw=1.5, color=color, label=label)
        ax.fill_between(x, qs_80[0], qs_80[1], alpha=0.5,
                        color=color, label="80% CI")
        ax.fill_between(x, qs_95[0], qs_95[1], alpha=0.2,
                        color=color, label="95% CI")

        if true_data is not None:
            ax.plot(x, true_data[:, region], lw=1.5,
                    color="black", label="Observed")

        return ax

    return plot_aggregated_over_regions, plot_region_fit


@app.cell
def _(observed_data, plot_aggregated_over_regions, plt, simulations):
    # --- Aggregated plots: Critical and Dead per age group (6x2 grid) ---
    fig_agg, axes = plt.subplots(6, 2, figsize=(16, 18), sharex=True)
    fig_agg.suptitle("Posterior Predictive — Aggregated Over Regions\nRows: Age Groups, Columns: Critical vs Deaths",
                     fontsize=14, y=0.995)

    for _age in range(6):
        # Critical cases
        plot_aggregated_over_regions(
            simulations[:, :, _age * 2, :],
            true_data=observed_data[:, _age * 2, :],
            ax=axes[_age, 0],
            label=f"Age {_age} — Critical",
            color="#1a6fa3",
        )

        # Deaths
        plot_aggregated_over_regions(
            simulations[:, :, _age * 2 + 1, :],
            true_data=observed_data[:, _age * 2 + 1, :],
            ax=axes[_age, 1],
            label=f"Age {_age} — Deaths",
            color="#a3381a",
        )

        axes[_age, 0].set_ylabel(f"Age {_age}", fontsize=10)
        axes[_age, 1].set_ylabel(f"Age {_age}", fontsize=10)

    axes[0, 0].set_title("Critical (ICU)", fontsize=11)
    axes[0, 1].set_title("Deaths", fontsize=11)

    for ax in axes[-1, :]:
        ax.set_xlabel("Day", fontsize=11)

    fig_agg.tight_layout()
    fig_agg
    return


@app.cell
def _(observed_data, plot_region_fit, plt, simulations):
    # --- Per-region grid: Critical cases ---
    fig_grid_crit, ax_grid_crit = plt.subplots(
        5, 6, figsize=(20, 16), sharex=True)
    fig_grid_crit.suptitle("Posterior Predictive — Critical (ICU) Cases\nRows: Regions, Columns: Age Groups",
                           fontsize=14, y=1.01)

    for _region in range(5):
        for _ag in range(6):
            plot_region_fit(
                simulations[:, :, _ag * 2, :],
                region=_region,
                true_data=observed_data[:, _ag * 2, :],
                ax=ax_grid_crit[_region, _ag],
                color="#1a6fa3",
            )
            if _region == 0:
                ax_grid_crit[_region, _ag].set_title(f"Age {_ag}", fontsize=10)
            if _ag == 0:
                ax_grid_crit[_region, _ag].set_ylabel(
                    f"Region {_region + 1}", fontsize=9)

    fig_grid_crit
    return


@app.cell
def _(observed_data, plot_region_fit, plt, simulations):
    # --- Per-region grid: Deaths ---
    fig_grid_dead, ax_grid_dead = plt.subplots(
        5, 6, figsize=(20, 16), sharex=True)
    fig_grid_dead.suptitle("Posterior Predictive — Deaths\nRows: Regions, Columns: Age Groups",
                           fontsize=14, y=1.01)

    for _region in range(5):
        for _ag in range(6):
            plot_region_fit(
                simulations[:, :, _ag * 2 + 1, :],
                region=_region,
                true_data=observed_data[:, _ag * 2 + 1, :],
                ax=ax_grid_dead[_region, _ag],
                color="#a3381a",
            )
            if _region == 0:
                ax_grid_dead[_region, _ag].set_title(f"Age {_ag}", fontsize=10)
            if _ag == 0:
                ax_grid_dead[_region, _ag].set_ylabel(
                    f"Region {_region + 1}", fontsize=9)

    fig_grid_dead
    return


@app.cell
def _(mo, np, posterior_samples):
    _ds = posterior_samples["damping_start"][0]   # (num_samples, 5)
    _dv = posterior_samples["damping_values"][0]  # (num_samples, 5)

    _header = "| Region | Damping Start (median) | Damping Start (90% CI) | Damping Value (median) | Damping Value (90% CI) |\n|---|---|---|---|---|\n"
    _body = ""
    for _r in range(5):
        _ds_med = f"{np.median(_ds[:, _r]):.1f}"
        _ds_ci = f"[{np.percentile(_ds[:, _r], 5):.1f}, {np.percentile(_ds[:, _r], 95):.1f}]"
        _dv_med = f"{np.median(_dv[:, _r]):.3f}"
        _dv_ci = f"[{np.percentile(_dv[:, _r], 5):.3f}, {np.percentile(_dv[:, _r], 95):.3f}]"
        _body += f"| Region {_r + 1} | {_ds_med} | {_ds_ci} | {_dv_med} | {_dv_ci} |\n"

    mo.md("## Posterior Summary\n\n" + _header + _body)
    return


if __name__ == "__main__":
    app.run()
