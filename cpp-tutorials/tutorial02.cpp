#include "ode_secir/model.h"
#include "memilio/compartments/flow_simulation.h"
#include "memilio/data/analyze_result.h"

int main()
{
    //  In Tutorial 1, we learned how to set up and simulate an ODE-based SECIR-type model. The output of the standard
    //  simulation gave us the exact number of individuals in each compartment (e.g., Susceptible, InfectedSymptoms,
    //  Recovered) at any given time. However, public health data usually reports incidence - such as daily new cases
    //  or daily new hospitalizations. If we simply look at the difference in the `InfectedSymptoms` compartment
    //  between today and yesterday, we do not get the true number of new cases. This is because, during that same day,
    //  other individuals may have recovered or progressed to a more severe state, leaving the compartment.

    // To obtain the exact number of new transitions (flows) between compartments, MEmilio provides a
    // flow-based formulation. The computational overhead of this flow-based formulation is minimal. The transition
    // rates between compartments have to be evaluated at every step in the standard compartmental formulation anyway.
    //  Therefore, calculating the cumulative flows simultaneously only adds a small amount of overhead due to the
    //  mapping of these variables and a slightly higher memory usage to store the additional values.

    // In this tutorial, we will show how to use `simulate_flows` to obtain these transition counts.

    // *** Set up model. ***
    // Identical setup to Tutorial 1 (one age group, no spatial resolution).
    size_t num_agegroups        = 1;
    ScalarType total_population = 100000;
    ScalarType t0               = 0;
    ScalarType tmax             = 100;
    ScalarType dt               = 0.1;

    mio::osecir::Model<ScalarType> model(num_agegroups);

    // Set infection state stay times (in days).
    model.parameters.get<mio::osecir::TimeExposed<ScalarType>>()            = 3.2;
    model.parameters.get<mio::osecir::TimeInfectedNoSymptoms<ScalarType>>() = 2.;
    model.parameters.get<mio::osecir::TimeInfectedSymptoms<ScalarType>>()   = 6.;
    model.parameters.get<mio::osecir::TimeInfectedSevere<ScalarType>>()     = 12.;
    model.parameters.get<mio::osecir::TimeInfectedCritical<ScalarType>>()   = 8.;

    // Set infection state transition probabilities.
    model.parameters.get<mio::osecir::RelativeTransmissionNoSymptoms<ScalarType>>()   = 0.67;
    model.parameters.get<mio::osecir::TransmissionProbabilityOnContact<ScalarType>>() = 0.1;
    model.parameters.get<mio::osecir::RecoveredPerInfectedNoSymptoms<ScalarType>>()   = 0.2;
    model.parameters.get<mio::osecir::RiskOfInfectionFromSymptomatic<ScalarType>>()   = 0.25;
    model.parameters.get<mio::osecir::SeverePerInfectedSymptoms<ScalarType>>()        = 0.2;
    model.parameters.get<mio::osecir::CriticalPerSevere<ScalarType>>()                = 0.25;
    model.parameters.get<mio::osecir::DeathsPerCritical<ScalarType>>()                = 0.3;

    // Set contact frequency.
    mio::ContactMatrixGroup<ScalarType>& contact_matrix =
        model.parameters.get<mio::osecir::ContactPatterns<ScalarType>>();
    contact_matrix[0] = mio::ContactMatrix<ScalarType>(Eigen::MatrixX<ScalarType>::Constant(1, 1, ScalarType(10)));

    // Initialize compartments: 1% of the population infected, split between Exposed and InfectedNoSymptoms.
    model.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::Exposed}]            = 0.005 * total_population;
    model.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::InfectedNoSymptoms}] = 0.005 * total_population;
    model.populations.set_difference_from_total({mio::AgeGroup(0), mio::osecir::InfectionState::Susceptible},
                                                total_population);

    // Before running the simulation, we check if all initial values and parameters are within a valid range.
    model.check_constraints();

    // *** Simulate flows. ***
    // Instead of using `simulate`, we now use `simulate_flows`.
    // This function integrates the transition rates between compartments. The result is a vector which contains
    // the compartment states as first element, and the cumulative flows between compartments as the second element.
    // The cumulative flows are the total number of transitions that have occurred between compartments up to that
    //point in time. The compartment states are the same as the output of `simulate`. Note that the entries of the
    // vectors are `TimeSeries` objects.
    std::vector<mio::TimeSeries<ScalarType>> results = mio::osecir::simulate_flows<ScalarType>(t0, tmax, dt, model);

    const auto& compartments     = results[0]; // compartments
    const auto& cumulative_flows = results[1]; // cumulative flows

    // Interpolate both results to full integer days for easier analysis.
    auto interp_comps = mio::interpolate_simulation_result(compartments);
    auto interp_flows = mio::interpolate_simulation_result(cumulative_flows);

    // *** Identify flow indices. ***
    // The SECIR model defines 15 flows per age group.  The flat index for a specific
    // flow transition in age group g is:  g * num_flows_per_group + position_in_Flows_TypeList.
    // Use get_flat_flow_index<Source, Target>({AgeGroup}) to obtain the specific index for a given transition.
    //
    // Flow list:
    //   0: Susceptible          -> Exposed                       (new infections)
    //   1: Exposed              -> InfectedNoSymptoms
    //   2: InfectedNoSymptoms   -> InfectedSymptoms              (new symptomatic cases)
    //   3: InfectedNoSymptoms   -> Recovered
    //   4: InfectedNoSymptomsConfirmed -> InfectedSymptomsConfirmed
    //   5: InfectedNoSymptomsConfirmed -> Recovered
    //   6: InfectedSymptoms     -> InfectedSevere                (new hospitalizations)
    //   7: InfectedSymptoms     -> Recovered
    //   ...  (full list: https://memilio.readthedocs.io/en/latest/cpp/models/osecir.html#flows)

    using IS = mio::osecir::InfectionState;
    const size_t idx_new_symptomatic =
        model.get_flat_flow_index<IS::InfectedNoSymptoms, IS::InfectedSymptoms>({mio::AgeGroup(0)});
    const size_t idx_new_hospitalized =
        model.get_flat_flow_index<IS::InfectedSymptoms, IS::InfectedSevere>({mio::AgeGroup(0)});

    // *** Compute daily incidences. ***
    // The interpolated flow TimeSeries stores cumulative values.  Subtracting consecutive days
    // returns the count of new transitions on each day (like np.diff in Python).
    // We build a new TimeSeries with two columns: daily new symptomatic cases and daily new hospitalizations.
    mio::TimeSeries<ScalarType> daily_incidences(2);
    const auto num_days = interp_flows.get_num_time_points();
    for (Eigen::Index i = 1; i < num_days; ++i) {
        const ScalarType t = interp_flows.get_time(i);
        const ScalarType daily_symptomatic =
            interp_flows.get_value(i)[idx_new_symptomatic] - interp_flows.get_value(i - 1)[idx_new_symptomatic];
        const ScalarType daily_hospitalized =
            interp_flows.get_value(i)[idx_new_hospitalized] - interp_flows.get_value(i - 1)[idx_new_hospitalized];
        Eigen::Vector2<ScalarType> vals;
        vals << daily_symptomatic, daily_hospitalized;
        daily_incidences.add_time_point(t, vals);
    }

    // *** Print results. ***
    // Compartment trajectories (same layout as Tutorial 1).
    // interp_comps.print_table({"S", "E", "C", "C_confirmed", "I", "I_confirmed", "H", "U", "R", "D"}, 16, 4);

    // Daily incidence table (rows = days 1..tmax, columns = new symptomatic / new hospitalized).
    // Note: the adaptive integrator may make time steps which are larger than 1 day. Following, interpolate_simulation_result
    // then fills missing days by linear interpolation of the cumulative flows, producing constant values in the
    // daily differences.
    daily_incidences.print_table({"New_Symptomatic", "New_Hospitalized"}, 20, 4);

    // *** Optional: export to CSV for plotting in Python (see tutorial2.py for the corresponding visualization). ***
    // interp_comps.export_csv("compartments.csv",
    //     {"S", "E", "C", "C_confirmed", "I", "I_confirmed", "H", "U", "R", "D"});
    // interp_flows.export_csv("flows.csv");
    // daily_incidences.export_csv("daily_incidences.csv", {"New_Symptomatic", "New_Hospitalized"});

    return 0;
}
