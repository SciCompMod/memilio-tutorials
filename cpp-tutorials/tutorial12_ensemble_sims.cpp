#include "memilio/compartments/parameter_studies.h"
#include "memilio/config.h"
#include "memilio/data/analyze_result.h"
#include "memilio/io/result_io.h"
#include "memilio/utils/base_dir.h"
#include "memilio/utils/miompi.h"
#include "memilio/utils/stl_util.h"

// These are model specific includes, replace them if you want to use another model
#include "ode_secir/model.h"
#include "ode_secir/parameters_io.h"
#include "ode_secir/parameter_space.h"
#include <algorithm>

// This method sets up the ODE model used below. For details on this setup, check out the ODE tutorial.
// It is not important for this tutorial, so it can be skipped.
auto set_up_model(ScalarType t0, ScalarType tmax, ScalarType dt)
{
    ScalarType cont_freq = 10, num_total_t0 = 10000, num_exp_t0 = 100, num_inf_t0 = 50, num_car_t0 = 50,
               num_hosp_t0 = 20, num_icu_t0 = 10, num_rec_t0 = 10, num_dead_t0 = 0;

    mio::osecir::Model<ScalarType> model(1);
    mio::AgeGroup num_groups = model.parameters.get_num_groups();
    ScalarType fact          = 1.0 / (ScalarType)(size_t)num_groups;

    auto& params = model.parameters;
    params.set<mio::osecir::ICUCapacity<ScalarType>>(std::numeric_limits<ScalarType>::max());
    params.set<mio::osecir::StartDay<ScalarType>>(0);
    params.set<mio::osecir::Seasonality<ScalarType>>(0);

    for (auto i = mio::AgeGroup(0); i < num_groups; i++) {
        params.get<mio::osecir::TimeExposed<ScalarType>>()[i]            = 3.2;
        params.get<mio::osecir::TimeInfectedNoSymptoms<ScalarType>>()[i] = 2.;
        params.get<mio::osecir::TimeInfectedSymptoms<ScalarType>>()[i]   = 6.;
        params.get<mio::osecir::TimeInfectedSevere<ScalarType>>()[i]     = 12;
        params.get<mio::osecir::TimeInfectedCritical<ScalarType>>()[i]   = 8;

        model.populations[{i, mio::osecir::InfectionState::Exposed}]            = fact * num_exp_t0;
        model.populations[{i, mio::osecir::InfectionState::InfectedNoSymptoms}] = fact * num_car_t0;
        model.populations[{i, mio::osecir::InfectionState::InfectedSymptoms}]   = fact * num_inf_t0;
        model.populations[{i, mio::osecir::InfectionState::InfectedSevere}]     = fact * num_hosp_t0;
        model.populations[{i, mio::osecir::InfectionState::InfectedCritical}]   = fact * num_icu_t0;
        model.populations[{i, mio::osecir::InfectionState::Recovered}]          = fact * num_rec_t0;
        model.populations[{i, mio::osecir::InfectionState::Dead}]               = fact * num_dead_t0;
        model.populations.set_difference_from_group_total<mio::AgeGroup>({i, mio::osecir::InfectionState::Susceptible},
                                                                         fact * num_total_t0);

        params.get<mio::osecir::TransmissionProbabilityOnContact<ScalarType>>()[i] = 0.05;
        params.get<mio::osecir::RelativeTransmissionNoSymptoms<ScalarType>>()[i]   = 0.67;
        params.get<mio::osecir::RecoveredPerInfectedNoSymptoms<ScalarType>>()[i]   = 0.09;
        params.get<mio::osecir::RiskOfInfectionFromSymptomatic<ScalarType>>()[i]   = 0.25;
        params.get<mio::osecir::SeverePerInfectedSymptoms<ScalarType>>()[i]        = 0.2;
        params.get<mio::osecir::CriticalPerSevere<ScalarType>>()[i]                = 0.25;
        params.get<mio::osecir::DeathsPerCritical<ScalarType>>()[i]                = 0.3;
    }
    params.apply_constraints();

    mio::ContactMatrixGroup<ScalarType>& contact_matrix = params.get<mio::osecir::ContactPatterns<ScalarType>>();
    contact_matrix[0]                                   = mio::ContactMatrix<ScalarType>(
        Eigen::MatrixX<ScalarType>::Constant((size_t)num_groups, (size_t)num_groups, fact * cont_freq));

    mio::osecir::set_params_distributions_normal<ScalarType>(model, t0, tmax, dt);

    return model;
}

// *** Tutorial 12 - Ensemble Simulations ***
// We provide several models in MEmilio, each with different features and/or modelling granularity. In either case, we
// often want to run several simulations of the same model type, so we can study the effect of stochasticity or varying
// parameter values.
//
int main()
{
    mio::set_log_level(mio::LogLevel::err);
    // It often makes sense to run simulations in parallel. For that, we use MPI, which can be enabled with the cmake
    // flag `-DMEMILIO_ENABLE_MPI=On`. This function here simply wraps MPI_Init. You will find a similar call for
    // MPI_Finalize at the end of main.
    mio::mpi::init();

    // Here, we set up the simulation scope. That is, we define the time interval [t0, tmax] and
    // the initial time step dt. It is called "initial", because the default integrator for ODE models is adaptive,
    // modifying the step size over time.
    ScalarType t0   = 0;
    ScalarType tmax = 50;
    ScalarType dt   = 0.1;
    // Set the total number of simulations to perform, i.e., the number of "runs".
    size_t num_runs = 3;
    // Create the model that we want to simulate.
    auto model = set_up_model(t0, tmax, dt);

    // Create a ParameterStudy object. It takes the model as first argument, followed by the simulation scope, and
    // number of runs.
    mio::ParameterStudy simulation_study(model, t0, tmax, dt, num_runs);
    // In the ParameterStudy, that first argument is treated as a constant parameter, that is used to
    // create a new simulation for each run. Since the steps to create this simulation varies across models and study
    // setups (e.g., we may want to draw new parameters), we must tell the study how to create it.
    // This is done by providing a function to the ParameterStudy::run function, which executes the entire study.

    // To that end, we define a function object (also called lambda expression) that describes how the ParameterStudy
    // should create a new Simulation. This is called once per run, and expects that the returned object has an
    // `advance(ScalarType tmax)` member. In our case, this type is:
    using SimulationType = mio::osecir::Simulation<ScalarType>;
    // The argument names in this lambda contain an underscore to differentiate them from other variables. The run id,
    // is a unique number from 0 to num_runs-1, which is unused in this setup.
    const auto create_simulation = [](const auto& model_, ScalarType t0_, ScalarType dt_, size_t run_id_) {
        mio::unused(run_id_);
        auto copy = model_;
        draw_sample(copy);
        return SimulationType(std::move(copy), t0_, dt_);
    };
    // This function defines how to treat the simulation result. Without it, ParameterStudy::run will simply return a
    // vector containing all simulations. Depending on the specifics of the setup, this can potentially fill up the
    // entire computer memory rather quickly, so be careful what to store, and consider writing larger data sets
    // directly to files. In this example, this should not be an issue, but, nevertheless, we only store the
    // interpolated results.
    const auto handle_result = [](auto&& sim_, size_t run_id_) {
        mio::unused(run_id_);
        return std::vector{mio::interpolate_simulation_result(sim_.get_result())};
    };

    // Optional: Set seeds to get reproducible results.
    //simulation_study.get_rng().seed({1456, 157456, 521346, 35345, 6875, 6435});

    // Finally, we run the study using its "run" function and the lambdas defined above.
    auto ensemble_result = simulation_study.run(create_simulation, handle_result);

    // Now, we can evaluate the results, for example, by checking the median values.
    auto median_result = ensemble_percentile(ensemble_result, 0.5).front();
    median_result.print_table({}, 10, 2);

    // Save the result to CSV, so we can plot it using the "plot_secir_results.py" script.
    (void)median_result.export_csv("../results_ensemble_sims.csv");

    // Close the MPI context and exit.
    mio::mpi::finalize();
    return 0;
}
