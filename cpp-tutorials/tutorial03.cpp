#include "ode_secir/model.h"
#include "memilio/compartments/simulation.h"
#include "memilio/data/analyze_result.h"

int main()
{
    // In the previous tutorial, we created, initialized and simulated MEmilio's ODE-based SECIR-type model with one (age) group. In this tutorial, we will show how to incorporate non-pharmaceutical interventions (NPIs) through the use of `Dampings` in the ODE-based SECIR-type model.

    // *** Set up model. ***
    // First we create and initialize a SECIR-type model with one age group. For a detailed description on taht, see Tutorial 1.
    size_t num_agegroups        = 1;
    ScalarType total_population = 100000;
    ScalarType t0               = 0;
    ScalarType tmax             = 100;
    ScalarType dt               = 0.1;

    // Create model
    mio::osecir::Model<ScalarType> model(num_agegroups);

    // Set infection state stay times (in days)
    model.parameters.get<mio::osecir::TimeExposed<ScalarType>>()            = 3.2;
    model.parameters.get<mio::osecir::TimeInfectedNoSymptoms<ScalarType>>() = 2.;
    model.parameters.get<mio::osecir::TimeInfectedSymptoms<ScalarType>>()   = 6.;
    model.parameters.get<mio::osecir::TimeInfectedSevere<ScalarType>>()     = 12.;
    model.parameters.get<mio::osecir::TimeInfectedCritical<ScalarType>>()   = 8.;
    // Set infection state transition probabilities
    model.parameters.get<mio::osecir::RelativeTransmissionNoSymptoms<ScalarType>>()   = 0.67;
    model.parameters.get<mio::osecir::TransmissionProbabilityOnContact<ScalarType>>() = 0.1;
    model.parameters.get<mio::osecir::RecoveredPerInfectedNoSymptoms<ScalarType>>()   = 0.2;
    model.parameters.get<mio::osecir::RiskOfInfectionFromSymptomatic<ScalarType>>()   = 0.25;
    model.parameters.get<mio::osecir::SeverePerInfectedSymptoms<ScalarType>>()        = 0.2;
    model.parameters.get<mio::osecir::CriticalPerSevere<ScalarType>>()                = 0.25;
    model.parameters.get<mio::osecir::DeathsPerCritical<ScalarType>>()                = 0.3;
    //Set contact frequency
    ScalarType contact_frequency = 10;
    mio::ContactMatrixGroup<ScalarType>& contact_matrix =
        model.parameters.get<mio::osecir::ContactPatterns<ScalarType>>();
    contact_matrix[0] = mio::ContactMatrix<ScalarType>(Eigen::MatrixX<ScalarType>::Constant(1, 1, contact_frequency));

    // After the model initialization, we add a contact reduction (`Damping`) that represents an NPI like e.g. mask wearing or social distancing. Dampings are a factor applied to the contact frequency and can be added to the model at fixed simulation time points before simulating. They have a *Level* and a *Type*. A damping with a given level and type replaces the previously active one with the same level and type, while all currently active dampings of one level and different types are summed up. If two dampings have different levels (independent of the type) they are combined multiplicatively. In the following we apply a `Damping` of 0.9 after 10 days and another damping of 0.6 after 20 days which means that the contacts are reduced by 10% and 40%, respectively. To always retain a minimum level of contacts, a minimum contact frequency can be set that is never deceeded. In our example we set this minimum contact rate to 0.
    contact_matrix[0].add_damping(0.9, mio::SimulationTime<ScalarType>(10.));
    contact_matrix[0].add_damping(0.6, mio::SimulationTime<ScalarType>(20.));

    // Again, we start with 0.5% of the population initially in `Exposed` and 0.5% initially in `InfectedNoSymptoms` while the remaining 99% is `Susceptible`.
    model.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::Exposed}]            = 0.005 * total_population;
    model.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::InfectedNoSymptoms}] = 0.005 * total_population;
    model.populations.set_difference_from_total({mio::AgeGroup(0), mio::osecir::InfectionState::Susceptible},
                                                total_population);

    // *** Simulate model. ***
    mio::TimeSeries<ScalarType> result = mio::osecir::simulate<ScalarType>(t0, tmax, dt, model);
    // Interpolate time series to full days.
    auto interpolated_result = mio::interpolate_simulation_result(result);

    // *** Print results. ***
    interpolated_result.print_table({"S", "E", "C", "C_confirmed", "I", "I_confirmed", "H", "U", "R", "D"}, 12, 4);

    // We export the results as csv which is saved in the current folder. Then we can plot the results using plot_secir_results.py.
    auto export_status = result.export_csv("../../cpp-tutorials/results_ode_npis.csv");
}
