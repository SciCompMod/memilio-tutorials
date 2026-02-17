#include "ide_secir/model.h"
#include "ide_secir/infection_state.h"
#include "ide_secir/simulation.h"
#include "memilio/config.h"
#include "memilio/epidemiology/age_group.h"
#include "memilio/math/eigen.h"
#include "memilio/utils/custom_index_array.h"
#include "memilio/utils/time_series.h"
#include "memilio/epidemiology/uncertain_matrix.h"
#include "memilio/epidemiology/state_age_function.h"
#include "memilio/data/analyze_result.h"

int main()
{
    // MEmilio implements two models based on integro-differential equations (IDEs) with different infection states.
    // IDE-based models are a generalization of ODE-based models. Whereas ODE-based models assume an exponential
    // distribution regarding the time spent in an infection state, IDE-based models allow for arbitrary distributions.

    // The following example shows how to set up and run a simple IDE-SECIR model without any further stratification.

    using Vec = Eigen::VectorX<ScalarType>;

    // Define simulation parameters.
    ScalarType t0   = 0.;
    ScalarType tmax = 5.;
    ScalarType dt   = 0.01; // The step size will stay constant throughout the simulation.

    // *** Set up model. ***
    // Define number of age groups.
    size_t num_agegroups = 1;

    // Define initial values for the total population and number of deaths per age group.
    mio::CustomIndexArray<ScalarType, mio::AgeGroup> total_population_init =
        mio::CustomIndexArray<ScalarType, mio::AgeGroup>(mio::AgeGroup(num_agegroups), 1000.);
    mio::CustomIndexArray<ScalarType, mio::AgeGroup> deaths_init =
        mio::CustomIndexArray<ScalarType, mio::AgeGroup>(mio::AgeGroup(num_agegroups), 6.);

    // Create TimeSeries with num_transitions * num_agegroups elements where initial transitions needed for simulation
    // will be stored. We require values for the transitions for a sufficient number of time points before the start of
    // the simulation to initialize our model.
    size_t num_transitions = (size_t)mio::isecir::InfectionTransition::Count;
    mio::TimeSeries<ScalarType> transitions_init(num_transitions);

    // Define vector of transitions that will be added as values to the time points of the TimeSeries transitions_init.
    Vec vec_init(num_transitions);
    vec_init[(size_t)mio::isecir::InfectionTransition::SusceptibleToExposed]                 = 25.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::ExposedToInfectedNoSymptoms]          = 15.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedNoSymptomsToInfectedSymptoms] = 8.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedNoSymptomsToRecovered]        = 4.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedSymptomsToInfectedSevere]     = 1.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedSymptomsToRecovered]          = 4.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedSevereToInfectedCritical]     = 1.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedSevereToRecovered]            = 1.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedCriticalToDead]               = 1.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedCriticalToRecovered]          = 1.0;

    // Multiply vec_init with dt so that within a time interval of length 1, always the above number of
    // individuals are transitioning from one compartment to another, irrespective of the chosen time step size.
    vec_init = vec_init * dt;

    // In this example, we will set the TransitionDistributions below. For these distributions, setting the initial time
    // point of the TimeSeries transitions_init at time -10 will give us a sufficient number of time points before t0.
    // For more information on this, we refer to the documentation of TransitionDistributions in
    // models/ide_secir/parameters.h.
    transitions_init.add_time_point(-10, vec_init);
    // Add further time points with distance dt until time t0. Here, we already define the start time of our simulation
    // as we will start the simulation at the last time point in transitions_init.
    while (transitions_init.get_last_time() < t0 - dt / 2) {
        transitions_init.add_time_point(transitions_init.get_last_time() + dt, vec_init);
    }

    // Initialize model.
    mio::isecir::Model model(std::move(transitions_init), total_population_init, deaths_init, num_agegroups);

    // *** Set parameters. ***
    // In the following we set the transition distributions between the InfectionStates.
    // We start by defining a SmootherCosine object that is defined by an initial distribtion parameter.
    mio::SmootherCosine<ScalarType> smoothcos(3.0);
    // This is passed to a StateAgeFunctionWrapper object which is the type of oject used within the model to allow
    // for variable transition distributions.
    mio::StateAgeFunctionWrapper<ScalarType> transition_distribution(smoothcos);
    // We define a vector of StateAgeFunctionsWrappers where we set each transition distribution of our model to the
    // above defined transition_distribution.
    std::vector<mio::StateAgeFunctionWrapper<ScalarType>> vec_transition_distribution(num_transitions,
                                                                                      transition_distribution);
    // Each transition can be set individually and is not necessary that all transition follow the same distribution
    // with the same parameter. For this, MEmilio already implements several possible distributions such as exponential,
    // gamma and lognormal distributions but arbitrary distributions can be implemented, see
    // cpp/memilio/epidemiology/state_age_function.h. Below, we demonstrate how to set the distribution from
    // InfectedNoSymptoms to InfectedSymptoms with a lognormal distibution.
    mio::LognormSurvivalFunction lognormal(0.3);
    mio::StateAgeFunctionWrapper<ScalarType> transition_distribution_INStISy(lognormal);
    vec_transition_distribution[(size_t)mio::isecir::InfectionTransition::InfectedNoSymptomsToInfectedSymptoms] =
        transition_distribution_INStISy;
    // Finally, the TransitionDistributions of the IDE-SECIR model are set according to vec_transition_distribution.
    model.parameters.get<mio::isecir::TransitionDistributions>()[mio::AgeGroup(0)] = vec_transition_distribution;

    // The following parameters define the relevant transition probabilities between InfectionStates.
    std::vector<ScalarType> vec_prob(num_transitions, 0.5);
    // The following probabilities must be 1, as there is no other way to go.
    vec_prob[(size_t)mio::isecir::InfectionTransition::SusceptibleToExposed]        = 1;
    vec_prob[(size_t)mio::isecir::InfectionTransition::ExposedToInfectedNoSymptoms] = 1;
    // The TransitionProbabilities of the IDE-SECIR model are set according to vec_prob.
    model.parameters.get<mio::isecir::TransitionProbabilities>()[mio::AgeGroup(0)] = vec_prob;

    // Further epidemiological parameters that define the transmission probability of Susceptibles on Contact with
    // infectious individulas, the risk of transmission from individuals that are infectious but not symptomatic, and
    // the risk of infection from indidivuals that are infectious and symptomatic. These parameters can be set as
    // functions of the time since infection and are set as the survival function of the exponential distribution here.
    mio::ExponentialSurvivalFunction<ScalarType> exponential(0.5);
    mio::StateAgeFunctionWrapper<ScalarType> prob(exponential);
    model.parameters.get<mio::isecir::TransmissionProbabilityOnContact>()[mio::AgeGroup(0)] = prob;
    model.parameters.get<mio::isecir::RelativeTransmissionNoSymptoms>()[mio::AgeGroup(0)]   = prob;
    model.parameters.get<mio::isecir::RiskOfInfectionFromSymptomatic>()[mio::AgeGroup(0)]   = prob;

    // Set factor for seasonality and start day of simulation to include seasonal variation of infection dynamics
    // within a year.
    model.parameters.get<mio::isecir::Seasonality>() = 0.1;
    model.parameters.get<mio::isecir::StartDay>() =
        40.; // Start the simulation on the 40th day of a year (i.e. in February).

    // Define contact matrix that defines the average number of contacts between individuals.
    mio::ContactMatrixGroup<ScalarType>& contact_matrix = model.parameters.get<mio::isecir::ContactPatterns>();
    contact_matrix[0] =
        mio::ContactMatrix<ScalarType>(Eigen::MatrixX<ScalarType>::Constant(num_agegroups, num_agegroups, 10.));

    std::cout << model.get_global_support_max(dt) << std::endl;
    // *** Simulate. ***
    // Check if all model constraints regarding initial values and parameters are satisfied before simulating.
    model.check_constraints(dt);

    // Carry out simulation.
    mio::isecir::Simulation sim(model, dt);
    sim.advance(tmax);

    // Interpolate results for compartments and flows to days.
    auto interpolated_results       = mio::interpolate_simulation_result(sim.get_result());
    auto interpolated_results_flows = mio::interpolate_simulation_result(sim.get_transitions());

    // Print results for compartments and flows.
    interpolated_results.print_table({"S", "E", "C", "I", "H", "U", "R", "D"}, 12, 4);
    interpolated_results_flows.print_table(
        {"S->E", "E->C", "C->I", "C->R", "I->H", "I->R", "H->U", "H->R", "U->D", "U->R"}, 12, 4);
}