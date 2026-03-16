#include "ode_secir/model.h"
#include "memilio/compartments/simulation.h"
#include "memilio/data/analyze_result.h"

int main()
{
    // MEmilio implements various models based on ordinary differential equations (ODEs). ODE-based models are a
    // subclass of compartmental models in which individuals are grouped into subpopulations called compartments.

    // In this tutorial we will setup and run MEmilio's ODE-based SECIR-type model. This model is particularly
    // suited for pathogens with pre- or asymptomatic infection states and when severe or critical symptoms are possible.
    // The model assumes perfect immunity after recovery. The used infection states or compartments are Susceptible (S),
    // Exposed(E), Non-symptomatically Infected (Ins), Symptomatically Infected (Isy), Severely Infected (Isev),
    // Critically Infected (Icri), Dead (D) and Recovered (R). The transitions are depicted in the following figure.

    // *** Set up model. ***
    // We need to specify basic parameters. In this tutorial, we use a simple model without spatial resolution and
    // with only one age group.
    size_t num_agegroups = 1;
    // We first define the `total_population` size and the simulation horizong through the start day `t0`, and the
    // simulation's end point `tmax`. By default, the ODE is solved with adaptive time stepping and the initial time step is `dt`.
    ScalarType total_population = 100000;
    ScalarType t0               = 0;
    ScalarType tmax             = 100;
    ScalarType dt               = 0.1;

    // Create model
    mio::osecir::Model<ScalarType> model(num_agegroups);

    // Next, we have to set the epidemiological model parameters which include the average stay times per infection
    // state, the state transition probabilities, and the contact frequency. A list of all parameters can be found
    // at https://memilio.readthedocs.io/en/latest/cpp/models/osecir.html. The parameters can be set as follows:
    // Set infection state stay times (in days)
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

    // EXERCISE: Please set the average time individuals spend in the exposed state (TimeExposed) to 4 days

    // In addition to the parameters, the initial number of individuals in each compartment has to be set. If a
    // compartment is not set, its initial value is zero by default. In this example, we start our simulation with 1 %
    // of the population initially infected, distributing them equally to the `Exposed` and the `InfectedNoSymptoms`
    // state, where the latter contains pre- and asymptomatic infectious individuals. With the last line,
    // we set the remaining part of the population (99%) to be susceptible.
    model.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::Exposed}]            = 0.005 * total_population;
    model.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::InfectedNoSymptoms}] = 0.005 * total_population;
    model.populations.set_difference_from_total({mio::AgeGroup(0), mio::osecir::InfectionState::Susceptible},
                                                total_population);

    // EXERCISE: Please set 2% of the population initially infected. The initially infected should be distributed to the compartpartments as follows: 0.5% is initially exposed (state `Exposed`), 0.5% is non-symptomatically infected (state `InfectedNosymptoms`) and 1% is symptomatically infected (state `InfectedSymptoms`).

    // To check that all initial parameter and compartmental values are in a meaningful range, MEmilio provides the
    // `check_constraints` function. If a value exceeds its meaningful range, a warning is printed and the function
    // returns `True`, otherwise it returns `False`.
    model.check_constraints();

    // *** Simulate model. ***
    mio::TimeSeries<ScalarType> result = mio::osecir::simulate<ScalarType>(t0, tmax, dt, model);
    // Interpolate time series to full days.
    auto interpolated_result = mio::interpolate_simulation_result(result);

    // *** Print results. ***
    interpolated_result.print_table({"S", "E", "C", "C_confirmed", "I", "I_confirmed", "H", "U", "R", "D"}, 12, 4);

    // We export the results as csv which is saved in the current folder. Then we can plot the results using plot_secir_results.py.
    auto export_status = result.export_csv("../../cpp-tutorials/exercises/results_ode.csv");

    // Try yourself
    // You have seen how to set up and run MEmilio's ODE-SECIR model. You can now explore the model yourself. Here are some suggestions what you can do:
    // - **Controlling the transmission process**: What happens if you modify the transmission probability or the contact frequency in the same way?
    // - **Diseases with severe courses**: Increase the proportion of severe or critical cases. What happens to the number of deaths?
    // - **Asymptomatic courses**: How does disease dynamics behave if we only have pre-symptomatic and no asymptomatic cases? What happens if we increase the average time in the non-symptomatic state?
}
