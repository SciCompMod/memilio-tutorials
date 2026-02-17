#include "lct_secir/model.h"
#include "lct_secir/infection_state.h"
#include "lct_secir/initializer_flows.h"
#include "memilio/config.h"
#include "memilio/utils/time_series.h"
#include "memilio/epidemiology/uncertain_matrix.h"
#include "memilio/epidemiology/lct_infection_state.h"
#include "memilio/math/eigen.h"
#include "memilio/utils/logging.h"
#include "memilio/compartments/simulation.h"
#include "memilio/data/analyze_result.h"

#include <vector>

int main()
{
    // MEmilio implements a SECIR-type model utilizing the Linear Chain Trick (LCT). This is a generalization of simple
    // ODE-based models and allows for Erlang distributed stay times in the compartments by introducing subcompartments.
    // Note that the resulting system is still described by ODEs.

    // The following example shows how to set up and run a simple LCT-SECIR model without any further stratification.

    // *** Set up model. ***
    // Define number of age groups.
    size_t num_agegroups = 1;

    // We start by defining the number of subcompartments per InfectionState. These are passed to an LctInfectionState
    // object that is then passed to the Model object. Note that the number of subcompartments in the Susceptible,
    // Recovered and Dead compartments are always one as individuals are either only leaving or entering the
    // respective compartments.
    constexpr size_t NumExposed = 2, NumInfectedNoSymptoms = 3, NumInfectedSymptoms = 1, NumInfectedSevere = 1,
                     NumInfectedCritical = 5;
    using InfState                       = mio::lsecir::InfectionState;
    using LctState = mio::LctInfectionState<ScalarType, InfState, 1, NumExposed, NumInfectedNoSymptoms,
                                            NumInfectedSymptoms, NumInfectedSevere, NumInfectedCritical, 1, 1>;

    // One single AgeGroup/Category member is used here. This is set implicitly by the number of LctState objects that
    // are passed to the model.
    using Model = mio::lsecir::Model<ScalarType, LctState>;
    Model model;

    // Define the initial values of the population per subcompartment.
    std::vector<std::vector<ScalarType>> initial_populations = {{750}, {30, 20},          {20, 10, 10}, {50},
                                                                {50},  {10, 10, 5, 3, 2}, {20},         {10}};

    // Assert that initial_populations has the right shape.
    if (initial_populations.size() != (size_t)InfState::Count) {
        mio::log_error("The number of vectors in initial_populations does not match the number of InfectionStates.");
        return 1;
    }
    if ((initial_populations[(size_t)InfState::Susceptible].size() !=
         LctState::get_num_subcompartments<InfState::Susceptible>()) ||
        (initial_populations[(size_t)InfState::Exposed].size() != NumExposed) ||
        (initial_populations[(size_t)InfState::InfectedNoSymptoms].size() != NumInfectedNoSymptoms) ||
        (initial_populations[(size_t)InfState::InfectedSymptoms].size() != NumInfectedSymptoms) ||
        (initial_populations[(size_t)InfState::InfectedSevere].size() != NumInfectedSevere) ||
        (initial_populations[(size_t)InfState::InfectedCritical].size() != NumInfectedCritical) ||
        (initial_populations[(size_t)InfState::Recovered].size() !=
         LctState::get_num_subcompartments<InfState::Recovered>()) ||
        (initial_populations[(size_t)InfState::Dead].size() != LctState::get_num_subcompartments<InfState::Dead>())) {
        mio::log_error("The length of at least one vector in initial_populations does not match the related number of "
                       "subcompartments.");
        return 1;
    }

    // Transfer the initial values in initial_populations to the model.
    std::vector<ScalarType> flat_initial_populations;
    for (auto&& vec : initial_populations) {
        flat_initial_populations.insert(flat_initial_populations.end(), vec.begin(), vec.end());
    }
    for (size_t i = 0; i < LctState::Count; i++) {
        model.populations[i] = flat_initial_populations[i];
    }

    // *** Set parameters. ***
    // The following parameters define the times individuals spend on average in the respective InfectionStates.
    model.parameters.get<mio::lsecir::TimeExposed<ScalarType>>()[0]            = 3.2;
    model.parameters.get<mio::lsecir::TimeInfectedNoSymptoms<ScalarType>>()[0] = 2.;
    model.parameters.get<mio::lsecir::TimeInfectedSymptoms<ScalarType>>()[0]   = 5.8;
    model.parameters.get<mio::lsecir::TimeInfectedSevere<ScalarType>>()[0]     = 9.5;
    model.parameters.get<mio::lsecir::TimeInfectedCritical<ScalarType>>()[0]   = 7.1;

    // The following parameters define the relevant transition probabilities between InfectionStates.
    model.parameters.get<mio::lsecir::RecoveredPerInfectedNoSymptoms<ScalarType>>()[0] = 0.09;
    model.parameters.get<mio::lsecir::SeverePerInfectedSymptoms<ScalarType>>()[0]      = 0.2;
    model.parameters.get<mio::lsecir::CriticalPerSevere<ScalarType>>()[0]              = 0.25;
    model.parameters.get<mio::lsecir::DeathsPerCritical<ScalarType>>()[0]              = 0.3;

    // Further epidemiological parameters that define the transmission probability of Susceptibles on Contact with
    // infectious individulas, the risk of transmission from individuals that are infectious but not symptomatic, and
    // the risk of infection from indidivuals that are infectious and symptomatic.
    model.parameters.get<mio::lsecir::TransmissionProbabilityOnContact<ScalarType>>()[0] = 0.05;
    model.parameters.get<mio::lsecir::RelativeTransmissionNoSymptoms<ScalarType>>()[0]   = 0.7;
    model.parameters.get<mio::lsecir::RiskOfInfectionFromSymptomatic<ScalarType>>()[0]   = 0.25;

    // Set factor for seasonality and start day of simulation to include seasonal variation of infection dynamics
    // within a year.
    model.parameters.get<mio::lsecir::Seasonality<ScalarType>>() = 0.1;
    model.parameters.get<mio::lsecir::StartDay<ScalarType>>() =
        40.; // Start the simulation on the 40th day of a year (i.e. in February).

    // Define contact matrix that defines the average number of contacts between individuals.
    mio::ContactMatrixGroup<ScalarType>& contact_matrix =
        model.parameters.get<mio::lsecir::ContactPatterns<ScalarType>>();
    contact_matrix[0] =
        mio::ContactMatrix<ScalarType>(Eigen::MatrixX<ScalarType>::Constant(num_agegroups, num_agegroups, 10));

    // *** Simulate. ***
    // Define simulation parameters.
    ScalarType t0      = 0.;
    ScalarType tmax    = 10.;
    ScalarType init_dt = 0.5; // May change throughout simulation as we are using an adaptive solver.

    // Perform a simulation.
    mio::TimeSeries<ScalarType> result = mio::simulate<ScalarType, Model>(t0, tmax, init_dt, model);
    // The simulation result is divided by subcompartments.
    // We call the function calculate_compartments to get a result according to the InfectionStates.
    mio::TimeSeries<ScalarType> population_no_subcompartments = model.calculate_compartments(result);

    // We interpolate the simulation results to days and print the results.
    auto interpolated_results = mio::interpolate_simulation_result(population_no_subcompartments);
    interpolated_results.print_table({"S", "E", "C", "I", "H", "U", "R", "D "}, 12, 4);
}