#include "lct_secir/model.h"
#include "lct_secir/infection_state.h"
#include "memilio/config.h"
#include "memilio/utils/time_series.h"
#include "memilio/epidemiology/uncertain_matrix.h"
#include "memilio/epidemiology/lct_infection_state.h"
#include "memilio/math/eigen.h"
#include "memilio/utils/logging.h"
#include "memilio/compartments/simulation.h"
#include "memilio/data/analyze_result.h"
#include "tutorial.h"

#include <vector>

int main()
{
    // MEmilio implements a SECIR-type model utilizing the Linear Chain Trick (LCT). This is a generalization of simple
    // ODE-based models and allows for Erlang distributed stay times in the compartments by introducing subcompartments.
    // In contrast to integral formulations (see tutorial_ide.cpp), the resulting system is still described by ODEs.

    // The following example shows how to set up and run a simple LCT-SECIR model without any further stratification.

    /*** Model setup ***/
    // First, we define the number of age groups used in the model.
    const size_t num_agegroups = 1;

    // We then define the number of subcompartments per InfectionState. The model-specific InfectionStates
    // can be found in `infection_state.h` in the model folder. The numbers for subdivision are passed to an
    // LctInfectionState object that is then passed to the Model object. Note that the number of subcompartments in the
    // Susceptible, Recovered and Dead compartments are always one as individuals are either only leaving or
    // entering the respective compartments. The `ScalarType` type below by default represents computation in double
    // precision.
    constexpr size_t NumExposed = 2, NumInfectedNoSymptoms = 3, NumInfectedSymptoms = 1, NumInfectedSevere = 1,
                     NumInfectedCritical = 5;
    using InfState                       = mio::lsecir::InfectionState;
    using LctState = mio::LctInfectionState<ScalarType, InfState, 1, NumExposed, NumInfectedNoSymptoms,
                                            NumInfectedSymptoms, NumInfectedSevere, NumInfectedCritical, 1, 1>;

    // For a single age group, the following call is sufficient.
    // For age-stratified models, we need to supply one LctState per age group.
    using Model = mio::lsecir::Model<ScalarType, LctState>;
    Model model;

    // Next, we define the initial values of the population per subcompartment with 750 susceptible individuals,
    // 30 individuals in the first exposed and 20 individuals in the second exposed state et cetera
    std::vector<ScalarType> initial_susceptible              = {750};
    std::vector<ScalarType> initial_exposed                  = {30, 20};
    std::vector<ScalarType> initial_infectednosymptoms       = {20, 10, 10};
    std::vector<ScalarType> initial_infectedsymptoms         = {50};
    std::vector<ScalarType> initial_infectedsevere           = {50};
    std::vector<ScalarType> initial_infectedcritical         = {10, 10, 5, 3, 2};
    std::vector<ScalarType> initial_recovered                = {20};
    std::vector<ScalarType> initial_dead                     = {10};
    std::vector<std::vector<ScalarType>> initial_populations = {
        initial_susceptible,    initial_exposed,          initial_infectednosymptoms, initial_infectedsymptoms,
        initial_infectedsevere, initial_infectedcritical, initial_recovered,          initial_dead};
    // A shorter initialization is given as follows.
    // std::vector<std::vector<ScalarType>> initial_populations = {{750}, {30, 20},          {20, 10, 10}, {50},
    //                                                            {50},  {10, 10, 5, 3, 2}, {20},         {10}};

    // We now validate that the initial_population vector has the right shape. For this we use
    // check_initial_population_per_group, see tutorial.h Since we are considering only one age group in this example,
    // we can apply the function directly to initial_populations.
    check_initial_population_per_group<LctState>(initial_populations);

    // After validation, we transfer the initial values in initial_populations to the model.
    std::vector<ScalarType> flat_initial_populations;
    for (auto&& vec : initial_populations) {
        flat_initial_populations.insert(flat_initial_populations.end(), vec.begin(), vec.end());
    }
    for (size_t i = 0; i < LctState::Count; i++) {
        model.populations[i] = flat_initial_populations[i];
    }

    // After having defined the populations, we now set the epidemiological parameters
    // that define the times individuals spend on average in the respective InfectionStates.
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

    // Further epidemiological parameters define the transmission probability of Susceptibles on contact with
    // infectious individuals, the relative risk of transmission from individuals that are infectious but not
    // symptomatic, and the risk of infection from indidivuals that are infectious and symptomatic.
    model.parameters.get<mio::lsecir::TransmissionProbabilityOnContact<ScalarType>>()[0] = 0.05;
    model.parameters.get<mio::lsecir::RelativeTransmissionNoSymptoms<ScalarType>>()[0]   = 0.7;
    model.parameters.get<mio::lsecir::RiskOfInfectionFromSymptomatic<ScalarType>>()[0]   = 0.25;

    // In order to include seasonality, we can set the seasonality's impact and the start day of
    // simulation. Seasonality is modeled by a sinoidal function. With a Seasonality value of 0.2, the risk of
    // transmission on January 1st is 50 % higher than on July 1st.
    model.parameters.get<mio::lsecir::Seasonality<ScalarType>>() = 0.2;
    model.parameters.get<mio::lsecir::StartDay<ScalarType>>() =
        40.; // Start the simulation on the 40th day of a year (i.e. February 9).

    // Transmission is driven by the risk of transmission per contact and the contact matrix that defines the
    // average daily number of contacts between individuals. For a model with a single age group, the contact matrix
    // reduces to a simple scalar value (here, set to 10).
    mio::ContactMatrixGroup<ScalarType>& contact_matrix =
        model.parameters.get<mio::lsecir::ContactPatterns<ScalarType>>();
    contact_matrix[0] =
        mio::ContactMatrix<ScalarType>(Eigen::MatrixX<ScalarType>::Constant(num_agegroups, num_agegroups, 10));

    // *** Model simulation ***
    // For model simulation, we first define simulation parameters.
    ScalarType t0      = 0.;
    ScalarType tmax    = 10.;
    ScalarType init_dt = 0.5; // May change throughout simulation as we are using an adaptive solver.

    // We then perform a simulation.
    mio::TimeSeries<ScalarType> result = mio::simulate<ScalarType, Model>(t0, tmax, init_dt, model);
    // The simulation result is divided by the subcompartments defined above.
    // We call the function calculate_compartments to get a result according to the InfectionStates.
    mio::TimeSeries<ScalarType> population_no_subcompartments = model.calculate_compartments(result);

    // We interpolate the simulation results to days and print the results.
    auto interpolated_results = mio::interpolate_simulation_result(population_no_subcompartments);
    interpolated_results.print_table({"S", "E", "C", "I", "H", "U", "R", "D "}, 12, 4);
}