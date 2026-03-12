#include "ode_secir/model.h"
#include "memilio/compartments/simulation.h"
#include "memilio/data/analyze_result.h"

int main()
{
    // In Tutorial 1, we created, initialized and simulated MEmilio's ODE-based SECIR-type model without any sociodemographic resolution. All ODE-based models have the possibility to add an arbitrary number of sociodemographic groups which can represent certain certain risk groups, like vaccination or age groups. Adding those groups can have a relevant impact on the simulation outcome. If for example older people have a higher risk of severe and critical infections, that can have an impact on ICU occupancy.
    // In the following, we initialize and simulate an ODE-based SECIR-type model with three age groups.

    // *** Set up model. ***
    // First we create and initialize a SECIR-type model with three age groups. For a detailed description on that, see Tutorial 1.
    size_t num_agegroups        = 3;
    ScalarType total_population = 100000;
    ScalarType t0               = 0;
    ScalarType tmax             = 100;
    ScalarType dt               = 0.1;

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

    //EXERCISE: Please increase the percentage of severe cases per symptomatic cases in age group 2 by 10%.

    //Set contact frequency
    ScalarType contact_frequency = 10;
    mio::ContactMatrixGroup<ScalarType>& contact_matrix =
        model.parameters.get<mio::osecir::ContactPatterns<ScalarType>>();
    contact_matrix[0] = mio::ContactMatrix<ScalarType>(
        Eigen::MatrixX<ScalarType>::Constant(num_agegroups, num_agegroups, contact_frequency));

    // In this example, 0.5% of the population is initially in `Exposed` and 0.5% is initially in `InfectedNoSymptoms` while the remaining 99% of the population is `Susceptible`. The fractions are equally distributed to all age groups by:
    for (size_t i = 0; i < num_agegroups; i++) {
        model.populations[{mio::AgeGroup(i), mio::osecir::InfectionState::Exposed}] =
            0.005 * total_population / num_agegroups;
        model.populations[{mio::AgeGroup(i), mio::osecir::InfectionState::InfectedNoSymptoms}] =
            0.005 * total_population / num_agegroups;
        model.populations.set_difference_from_group_total<mio::AgeGroup>(
            {mio::AgeGroup(i), mio::osecir::InfectionState::Susceptible}, total_population / num_agegroups);
    }

    // *** Simulate model. ***
    // After having initialized the model, dynamics can be simulated. The simulation output is a time series containing the evolution of all compartments per age group over time. In the following we simulate the model from `t0` to `tmax` with initial step size `dt` and subsequently print the time series result:
    mio::TimeSeries<ScalarType> result = mio::osecir::simulate<ScalarType>(t0, tmax, dt, model);
    // Interpolate time series to full days.
    auto interpolated_result = mio::interpolate_simulation_result(result);

    // *** Print results. ***
    interpolated_result.print_table();

    // We export the results as csv which is saved in the current folder. Then we can plot the results using plot_secir_results.py.
    auto export_status = result.export_csv("../../cpp-tutorials/exercises/results_ode_ageres.csv");
}
