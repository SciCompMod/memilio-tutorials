#include "memilio/config.h"
#include "ode_secir/model.h"
#include "ode_secir/infection_state.h"
#include "ode_secir/parameters.h"
#include "memilio/mobility/metapopulation_mobility_instant.h"
#include "memilio/compartments/simulation.h"
#include "memilio/data/analyze_result.h"

int main()
{
    // In the previous tutorials, we saw how to set up and run an age-resolved ODE-based SECIR-type model. However, one limiting assumption of simple ODE-based models is the assumption of homogenous mixing within the population. To overcome this limitation and incorporate spatial heterogeneity, in this example we show how to use MEmilio's graph-based metapopulation model. This model realizes mobility between regions via graph edges, while every region is represented by a graph node containing it's own ODE-based model.

    // *** Set up model. ***
    // We set the simulation start time `t0`, the end time `tmax` and the initial step size `dt` as:
    ScalarType t0   = 0;
    ScalarType tmax = 100;
    ScalarType dt   = 0.1;

    // Next, we need to specify the parameters. We will initialize a metapopulation model with two regions. The total population as well as the epidemiological parameters will be the same for both regions.
    ScalarType total_population_per_region = 100000;

    // We use a model with three age groups for both regions:
    size_t num_agegroups = 3;
    // Create model with three age groups
    mio::osecir::Model<ScalarType> model(num_agegroups);

    // Now, we have to set the epidemiological model parameters which are dependent on age group. A list of all parameters can be found at https://memilio.readthedocs.io/en/latest/cpp/models/osecir.html.
    // We choose an increasing risk of severe and critical infections for age group 2 and 3 compared to age group 1. The other parameters are equal for all age groups.

    for (size_t i = 0; i < num_agegroups; i++) {
        // Set infection state stay times (in days)
        model.parameters.get<mio::osecir::TimeExposed<ScalarType>>()[mio::AgeGroup(i)]            = 3.2;
        model.parameters.get<mio::osecir::TimeInfectedNoSymptoms<ScalarType>>()[mio::AgeGroup(i)] = 2.;
        model.parameters.get<mio::osecir::TimeInfectedSymptoms<ScalarType>>()[mio::AgeGroup(i)]   = 6.;
        model.parameters.get<mio::osecir::TimeInfectedSevere<ScalarType>>()[mio::AgeGroup(i)]     = 12.;
        model.parameters.get<mio::osecir::TimeInfectedCritical<ScalarType>>()[mio::AgeGroup(i)]   = 9.;
        // Set infection state transition probabilities
        model.parameters.get<mio::osecir::TransmissionProbabilityOnContact<ScalarType>>()[mio::AgeGroup(i)] = 0.1;
        model.parameters.get<mio::osecir::RelativeTransmissionNoSymptoms<ScalarType>>()[mio::AgeGroup(i)]   = 0.67;
        model.parameters.get<mio::osecir::RecoveredPerInfectedNoSymptoms<ScalarType>>()[mio::AgeGroup(i)]   = 0.2;
        model.parameters.get<mio::osecir::RiskOfInfectionFromSymptomatic<ScalarType>>()[mio::AgeGroup(i)]   = 0.25;
        model.parameters.get<mio::osecir::DeathsPerCritical<ScalarType>>()[mio::AgeGroup(i)]                = 0.3;
    }

    // The groups have an increasing risk of severe and critical infections
    model.parameters.get<mio::osecir::SeverePerInfectedSymptoms<ScalarType>>()[mio::AgeGroup(0)] = 0.2;
    model.parameters.get<mio::osecir::SeverePerInfectedSymptoms<ScalarType>>()[mio::AgeGroup(1)] = 0.2 * 1.5;
    model.parameters.get<mio::osecir::SeverePerInfectedSymptoms<ScalarType>>()[mio::AgeGroup(2)] = 0.2 * 2;
    model.parameters.get<mio::osecir::CriticalPerSevere<ScalarType>>()[mio::AgeGroup(0)]         = 0.25;
    model.parameters.get<mio::osecir::CriticalPerSevere<ScalarType>>()[mio::AgeGroup(1)]         = 0.25 * 1.5;
    model.parameters.get<mio::osecir::CriticalPerSevere<ScalarType>>()[mio::AgeGroup(2)]         = 0.25 * 2;

    // Set contact frequency
    ScalarType contact_frequency = 10;
    mio::ContactMatrixGroup<ScalarType>& contact_matrix =
        model.parameters.get<mio::osecir::ContactPatterns<ScalarType>>();
    contact_matrix[0] = mio::ContactMatrix<ScalarType>(
        Eigen::MatrixX<ScalarType>::Constant(num_agegroups, num_agegroups, contact_frequency));

    // Next, we create the graph via:
    mio::Graph<mio::SimulationNode<ScalarType, mio::osecir::Simulation<ScalarType>>, mio::MobilityEdge<ScalarType>>
        graph;

    // we want to add two regions (nodes) to the graph, therefore we need two copies of the model
    auto model_region1 = model;
    auto model_region2 = model;

    // In the graph-based metapopulation model, every graph node gets it's own ODE-based model which is copied when adding a graph node and handing the model to it as parameter. Therefore we can choose different initial conditions (as well as differing parameters) for different graph nodes. In our example, we simulate two regions with only one region having initially infected individuals. We choose 1% initially infected for that region while the other region starts with a totally susceptible population.
    // The model compartments for the first node are initialized via:
    for (size_t i = 0; i < num_agegroups; i++) {
        model_region1.populations[{mio::AgeGroup(i), mio::osecir::InfectionState::Exposed}] =
            0.005 * total_population_per_region / num_agegroups;
        model_region1.populations[{mio::AgeGroup(i), mio::osecir::InfectionState::InfectedNoSymptoms}] =
            0.005 * total_population_per_region / num_agegroups;
        model_region1.populations.set_difference_from_group_total<mio::AgeGroup>(
            {mio::AgeGroup(i), mio::osecir::InfectionState::Susceptible}, total_population_per_region / num_agegroups);
    }

    // We set the second model's initial populations to totally susceptible
    for (size_t i = 0; i < num_agegroups; i++) {
        model_region2.populations.set_difference_from_group_total<mio::AgeGroup>(
            {mio::AgeGroup(i), mio::osecir::InfectionState::Susceptible}, total_population_per_region / num_agegroups);
    }

    // After having initialized the models, we add two nodes (regions) to the graph
    graph.add_node(0, model_region1, t0, dt);

    //EXERCISE: Please add the second node to the graph.

    // If we would simulate the graph-based metapopulation model now, we would just have two independent ODE-based SECIR-type models running with different initial conditions. In reality, there is usually exchange between regions through individuals travelling or commuting from one region to another. This can be realized via graph edges.
    // We here use a symmetric mobility i.e. we have the same number of individuals that travel from node 0 to node 1 as vice versa. We let 10% of the population commute via the edges twice a day.
    graph.add_edge(
        0, 1, Eigen::VectorX<ScalarType>::Constant((size_t)mio::osecir::InfectionState::Count * num_agegroups, 0.1));

    // EXERCISE: Please add an edge from node 1 to node 0.

    // Exchange commuters twice a day
    double dt_exchange = 0.5;

    // *** Simulate model. ***
    // We now have finished initializing the metapopulation model. The graph-based simulation is created and advanced until `tmax`
    auto sim = mio::make_mobility_sim<ScalarType>(t0, dt_exchange, std::move(graph));

    // EXERCISE: Please advance the simulation until `tmax`.

    // As every graph node has its own model, we get one result time series per node. Those can be accessed as follows
    auto result_region0 = sim.get_graph().nodes()[0].property.get_result();
    auto result_region1 = sim.get_graph().nodes()[1].property.get_result();
    // Interpolate time series to full days.
    auto interpolated_result_r0 = mio::interpolate_simulation_result(result_region0);

    // *** Print results. ***
    interpolated_result_r0.print_table();
}
