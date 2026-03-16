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

    /*** Model setup ***/
    // Define simulation parameters.
    ScalarType t0   = 0.;
    ScalarType tmax = 20.;
    ScalarType dt   = 0.01; // The step size will stay constant throughout the simulation.

    // We define the number of age groups used in the model.
    size_t num_agegroups = 1;

    // Next, we define initial values for the total population and number of deaths per age group.
    mio::CustomIndexArray<ScalarType, mio::AgeGroup> total_population_init =
        mio::CustomIndexArray<ScalarType, mio::AgeGroup>(mio::AgeGroup(num_agegroups), 1000.);
    mio::CustomIndexArray<ScalarType, mio::AgeGroup> deaths_init =
        mio::CustomIndexArray<ScalarType, mio::AgeGroup>(mio::AgeGroup(num_agegroups), 0.);

    // Now, we create a TimeSeries object with num_transitions * num_agegroups elements where initial transitions needed for simulation
    // will be stored. We require values for the transitions for a sufficient number of time points before the start of
    // the simulation to initialize our model.
    size_t num_transitions = (size_t)mio::isecir::InfectionTransition::Count;
    mio::TimeSeries<ScalarType> transitions_init(num_transitions);

    // Then, we define vector of transitions that will be added as values to the time points of the TimeSeries transitions_init.
    Vec vec_init(num_transitions);
    vec_init[(size_t)mio::isecir::InfectionTransition::SusceptibleToExposed]                 = 3.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::ExposedToInfectedNoSymptoms]          = 0.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedNoSymptomsToInfectedSymptoms] = 0.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedNoSymptomsToRecovered]        = 0.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedSymptomsToInfectedSevere]     = 0.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedSymptomsToRecovered]          = 0.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedSevereToInfectedCritical]     = 0.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedSevereToRecovered]            = 0.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedCriticalToDead]               = 0.0;
    vec_init[(size_t)mio::isecir::InfectionTransition::InfectedCriticalToRecovered]          = 0.0;

    // We multiply vec_init with dt so that within a time interval of length 1, always the above number of
    // individuals are transitioning from one compartment to another, irrespective of the chosen time step size.
    vec_init = vec_init * dt;

    // In this example, we will set the TransitionDistributions below. For these distributions, setting the initial time
    // point of the TimeSeries transitions_init at time -10 will give us a sufficient number of time points before t0.
    // For more information on this, we refer to the documentation of TransitionDistributions in
    // models/ide_secir/parameters.h.

    // TODO: Set the initial time point of the TimeSeries transitions_init where the initial time is -10 and the value
    // is given by vec_init.

    // We add further time points with distance dt until time t0. Here, we already define the start time of our simulation
    // as we will start the simulation at the last time point in transitions_init.
    while (transitions_init.get_last_time() < t0 - dt / 2) {
        transitions_init.add_time_point(transitions_init.get_last_time() + dt, vec_init);
    }

    // With this, we can initialize the model.
    mio::isecir::Model model(std::move(transitions_init), total_population_init, deaths_init, num_agegroups);

    // We set the number of initially Recovered individuals to 10.
    model.populations[0][(size_t)mio::isecir::InfectionState::Recovered] = 10.;

    // After having initialized the model, we now set the epidemiological parameters.
    // We start with setting the transition distributions between the InfectionStates. With this, we define the times
    // individuals spend on average in the respective InfectionStates.
    // We start by defining a SmootherCosine object that is defined by an initial distribution parameter.
    mio::SmootherCosine<ScalarType> smoothcos(4.0);
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
    // cpp/memilio/epidemiology/state_age_function.h.
    //
    // TODO: Set the distribution for the transition from InfectedNoSymptoms to InfectedSymptoms with a lognormal
    // distribution with an initial distribution parameter of 0.3.

    // Finally, the TransitionDistributions of the IDE-SECIR model are set according to vec_transition_distribution.
    model.parameters.get<mio::isecir::TransitionDistributions>()[mio::AgeGroup(0)] = vec_transition_distribution;

    // The following parameters define the relevant transition probabilities between InfectionStates.
    std::vector<ScalarType> vec_prob(num_transitions, 0.5);
    // The following probabilities must be 1, as there is no other way to go.
    vec_prob[(size_t)mio::isecir::InfectionTransition::SusceptibleToExposed]        = 1;
    vec_prob[(size_t)mio::isecir::InfectionTransition::ExposedToInfectedNoSymptoms] = 1;
    // The TransitionProbabilities of the IDE-SECIR model are set according to vec_prob.
    model.parameters.get<mio::isecir::TransitionProbabilities>()[mio::AgeGroup(0)] = vec_prob;

    // Further epidemiological parameters define the transmission probability of Susceptibles on contact with
    // infectious individuals, the relative risk of transmission from individuals that are infectious but not
    // symptomatic, and the risk of infection from indidivuals that are infectious and symptomatic. These parameters
    // can be set as functions of the time since infection and are set as the survival function of the exponential distribution here.
    mio::ExponentialSurvivalFunction<ScalarType> exponential(0.3);
    mio::StateAgeFunctionWrapper<ScalarType> prob(exponential);
    model.parameters.get<mio::isecir::TransmissionProbabilityOnContact>()[mio::AgeGroup(0)] = prob;
    model.parameters.get<mio::isecir::RelativeTransmissionNoSymptoms>()[mio::AgeGroup(0)]   = prob;
    model.parameters.get<mio::isecir::RiskOfInfectionFromSymptomatic>()[mio::AgeGroup(0)]   = prob;

    // In order to include seasonality, we can set the seasonality's impact and the start day of
    // simulation. Seasonality is modeled by a sinoidal function. With a Seasonality value of 0.2, the risk of
    // transmission on January 1st is 50 % higher than on July 1st.
    model.parameters.get<mio::isecir::Seasonality>() = 0.1;
    model.parameters.get<mio::isecir::StartDay>() =
        40.; // Start the simulation on the 40th day of a year (i.e. February 9).

    // Transmission is driven by the risk of transmission per contact and the contact matrix that defines the
    // average daily number of contacts between individuals. For a model with a single age group, the contact matrix
    // reduces to a simple scalar value (here, set to 10).
    mio::ContactMatrixGroup<ScalarType>& contact_matrix = model.parameters.get<mio::isecir::ContactPatterns>();
    contact_matrix[0] =
        mio::ContactMatrix<ScalarType>(Eigen::MatrixX<ScalarType>::Constant(num_agegroups, num_agegroups, 10.));

    // We call the global_support_max method to get the maximum support_max over all TransitionDistributions. This
    // determines how many historic time points we need when initializing the model.
    std::cout << "Global support max: " << model.get_global_support_max(dt) << std::endl;

    // *** Model simulation ***
    // We check if all model constraints regarding initial values and parameters are satisfied before simulating.
    // Note: MEmilio's check_constraints() returns True if a constraint is violated, and False if everything is fine.
    model.check_constraints(dt);

    // We then perform a simulation.
    mio::isecir::Simulation sim(model, dt);
    sim.advance(tmax);

    // We interpolate the simulation results for compartments and flows to days and print the results.
    auto interpolated_results       = mio::interpolate_simulation_result(sim.get_result());
    auto interpolated_results_flows = mio::interpolate_simulation_result(sim.get_transitions());

    interpolated_results.print_table({"S", "E", "C", "I", "H", "U", "R", "D"}, 12, 4);
    interpolated_results_flows.print_table(
        {"S->E", "E->C", "C->I", "C->R", "I->H", "I->R", "H->U", "H->R", "U->D", "U->R"}, 12, 4);

    // We export the results as csv which is saved in the current folder. Then we can plot the results using plot_secir_results.py.
    auto export_status = sim.get_result().export_csv("../../cpp-tutorials/exercises/results_ide.csv");
}