#include "ode_secir/model.h"
#include "memilio/compartments/simulation.h"
#include "memilio/data/analyze_result.h"

#include <iostream>

// *** Introduction ***
// In Tutorial 3, we applied contact reductions (Dampings) uniformly to a single contact matrix. In reality,
// contacts between individuals occur in different locations: at home, at school, at work, and in other places
// such as transport. Different non-pharmaceutical interventions (NPIs) target different settings, for instance
// a school closure reduces school contacts, while a home-office mandate mainly reduces work contacts.
//
// Large-scale studies, such as the POLYMOD project, have measured social contact patterns across different
// locations (like home, school, work, and others) in various countries. For instance, see
//   Mossong J, Hens N, Jit M, Beutels P, Auranen K, et al. (2008) Social Contacts and Mixing Patterns Relevant to
//               the Spread of Infectious Diseases. PLoS Med 5(3): e74. https://doi.org/10.1371/journal.pmed.0050074
//   Prem K, Cook AR, Jit M (2017) Projecting social contact matrices in 152 countries using contact surveys and
//               demographic data. PLoS Comput Biol 13(9): e1005697. https://doi.org/10.1371/journal.pcbi.1005697
//
// In this tutorial, we extend the approach from Tutorial 3 by splitting the contact matrix into
// location-specific contact matrices using ContactMatrixGroup. This allows us to apply NPIs to individual
// locations, allowing a more realistic and detailed representation of intervention effects.

// Location indices matching the four contact contexts.
enum class Location : size_t
{
    Home   = 0,
    School = 1,
    Work   = 2,
    Other  = 3
};

// *** Location-specific Contact Matrices ***
// Instead of a single contact matrix, we now use a ContactMatrixGroup, containing one ContactMatrix per
// location. Each ContactMatrix has two components:
//   Baseline: the typical number of daily contacts at the location under regular conditions.
//   Minimum:  the minimal contact rate even under the strictest restrictions, as some contacts, especially
//             at home, cannot be fully avoided.
//
// The effective contact rate at simulation time t for a location l is:
//   C_l(t) = C_l_baseline - D_l(t) * (C_l_baseline - C_l_minimum)
// where D_l(t) <= 1 is the damping coefficient. For D > 0 contacts are reduced; for D = 1 they reach the
// minimum; and for D < 0 contacts are increased above the baseline. The upper bound D <= 1 is enforced by
// MEmilio's C++ core. The ODE model uses the sum of all location-specific matrices as the total contact rate.
//
// The total baseline contact rate (sum over all locations) equals 4+3+2+1 = 10 contacts/day, matching Tutorial 3.

ScalarType total_population = 100000;
ScalarType t0               = 0;
ScalarType tmax             = 100;
ScalarType dt               = 0.1;

// Create an OSECIR model with location-specific contact matrices.
mio::osecir::Model<ScalarType> create_model()
{
    mio::osecir::Model<ScalarType> m(1); // one age group

    // Set infection state stay times (in days)
    m.parameters.get<mio::osecir::TimeExposed<ScalarType>>()            = 3.2;
    m.parameters.get<mio::osecir::TimeInfectedNoSymptoms<ScalarType>>() = 2.;
    m.parameters.get<mio::osecir::TimeInfectedSymptoms<ScalarType>>()   = 6.;
    m.parameters.get<mio::osecir::TimeInfectedSevere<ScalarType>>()     = 12.;
    m.parameters.get<mio::osecir::TimeInfectedCritical<ScalarType>>()   = 8.;

    // Set infection state transition probabilities
    m.parameters.get<mio::osecir::RelativeTransmissionNoSymptoms<ScalarType>>()   = 0.67;
    m.parameters.get<mio::osecir::TransmissionProbabilityOnContact<ScalarType>>() = 0.1;
    m.parameters.get<mio::osecir::RecoveredPerInfectedNoSymptoms<ScalarType>>()   = 0.2;
    m.parameters.get<mio::osecir::RiskOfInfectionFromSymptomatic<ScalarType>>()   = 0.25;
    m.parameters.get<mio::osecir::SeverePerInfectedSymptoms<ScalarType>>()        = 0.2;
    m.parameters.get<mio::osecir::CriticalPerSevere<ScalarType>>()                = 0.25;
    m.parameters.get<mio::osecir::DeathsPerCritical<ScalarType>>()                = 0.3;

    // Create ContactMatrixGroup: 4 location matrices, each of size 1x1 (one age group).
    mio::ContactMatrixGroup<ScalarType> contacts(4, 1);
    // Home contacts: baseline 4/day, minimum 1/day (irreducible household contacts)
    contacts[(size_t)Location::Home] = mio::ContactMatrix<ScalarType>(Eigen::MatrixX<ScalarType>::Constant(1, 1, 4.0),
                                                                      Eigen::MatrixX<ScalarType>::Constant(1, 1, 1.0));
    // School contacts: baseline 3/day
    contacts[(size_t)Location::School] =
        mio::ContactMatrix<ScalarType>(Eigen::MatrixX<ScalarType>::Constant(1, 1, 3.0));
    // Work contacts: baseline 2/day
    contacts[(size_t)Location::Work] = mio::ContactMatrix<ScalarType>(Eigen::MatrixX<ScalarType>::Constant(1, 1, 2.0));
    // Other contacts: baseline 1/day
    contacts[(size_t)Location::Other] = mio::ContactMatrix<ScalarType>(Eigen::MatrixX<ScalarType>::Constant(1, 1, 1.0));

    m.parameters.get<mio::osecir::ContactPatterns<ScalarType>>() = contacts;

    // Initial populations: 0.5% Exposed, 0.5% InfectedNoSymptoms, rest Susceptible
    m.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::Exposed}]            = 0.005 * total_population;
    m.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::InfectedNoSymptoms}] = 0.005 * total_population;
    m.populations.set_difference_from_total({mio::AgeGroup(0), mio::osecir::InfectionState::Susceptible},
                                            total_population);
    return m;
}

int main()
{
    // *** Baseline Simulation Without NPIs ***
    // We first create and simulate the model without any interventions to obtain a baseline.
    auto model_no_npi  = create_model();
    auto result_no_npi = mio::osecir::simulate<ScalarType>(t0, tmax, dt, model_no_npi);

    // *** Adding Location-specific NPIs ***
    // We now model a lockdown scenario in which three NPIs are activated simultaneously on day 20:
    //
    //   Intervention               | Location | Damping D | Effect
    //   School closure             | School   | 1.0       | All school contacts eliminated
    //   Home-office mandate        | Work     | 0.5       | Work contacts reduced by 50%
    //   Public transport restriction| Other   | 0.8       | Other contacts reduced by 80%
    //
    // The key difference from Tutorial 3 is that each Damping is applied to a specific location matrix by
    // indexing cont_freq_mat[location_index] before calling add_damping. In Tutorial 3, add_damping was called
    // on the whole group, reducing all contact matrices simultaneously. Home contacts are left unrestricted
    // here, but they cannot fall below the minimum of 1 contact/day regardless.

    ScalarType t_npi_start = 20.0; // day at which interventions start

    auto model_with_npi = create_model();
    auto& cm            = model_with_npi.parameters.get<mio::osecir::ContactPatterns<ScalarType>>().get_cont_freq_mat();

    // School closure: damping of 1.0 -> effective school contacts reach the minimum (0)
    cm[(size_t)Location::School].add_damping(1.0, mio::SimulationTime<ScalarType>(t_npi_start));
    // Home-office mandate: damping of 0.5 -> work contacts halved
    cm[(size_t)Location::Work].add_damping(0.5, mio::SimulationTime<ScalarType>(t_npi_start));
    // Restrictions in public transport: damping of 0.8 -> other contacts reduced by 80%
    cm[(size_t)Location::Other].add_damping(0.8, mio::SimulationTime<ScalarType>(t_npi_start));

    // We can verify the effective total contact rates before and after the NPIs by evaluating the contact
    // matrix group at a specific point in time.
    auto contacts_before = cm.get_matrix_at(mio::SimulationTime<ScalarType>(t_npi_start - 1));
    auto contacts_after  = cm.get_matrix_at(mio::SimulationTime<ScalarType>(t_npi_start + 1));
    std::cout << "Total contacts before NPIs (day " << (int)(t_npi_start - 1) << "): " << contacts_before(0, 0)
              << " / day\n";
    std::cout << "Total contacts after  NPIs (day " << (int)(t_npi_start + 1) << "): " << contacts_after(0, 0)
              << " / day\n";

    // We simulate the NPI scenario from t0 to tmax.
    auto result_with_npi = mio::osecir::simulate<ScalarType>(t0, tmax, dt, model_with_npi);

    // *** Print results ***
    // Interpolate both results to full integer days and print the InfectedSymptoms column for comparison.
    auto interp_no_npi   = mio::interpolate_simulation_result(result_no_npi);
    auto interp_with_npi = mio::interpolate_simulation_result(result_with_npi);

    std::cout << "\n--- Without NPIs ---\n";
    interp_no_npi.print_table({"S", "E", "C", "C_confirmed", "I", "I_confirmed", "H", "U", "R", "D"}, 12, 4);
    std::cout << "\n--- With location-specific NPIs ---\n";
    interp_with_npi.print_table({"S", "E", "C", "C_confirmed", "I", "I_confirmed", "H", "U", "R", "D"}, 12, 4);

    // Optional: export to CSV for plotting in Python (see tutorial10.py for the corresponding visualization).
    // interp_no_npi.export_csv("result_no_npi.csv",
    //     {"S", "E", "C", "C_confirmed", "I", "I_confirmed", "H", "U", "R", "D"});
    // interp_with_npi.export_csv("result_with_npi.csv",
    //     {"S", "E", "C", "C_confirmed", "I", "I_confirmed", "H", "U", "R", "D"});

    // *** Summary ***
    // In this tutorial, we have introduced location-specific contact patterns via ContactMatrixGroup and
    // applied targeted NPIs to individual location matrices. Key takeaways:
    //   - A ContactMatrixGroup collects one ContactMatrix per location (Home, School, Work, Other).
    //   - Each ContactMatrix has a baseline (normal contacts) and a minimum (irreducible lower bound).
    //   - Location-specific NPIs are applied with cont_freq_mat[location_index].add_damping(...).
    //     In contrast, Tutorial 3 used add_damping on the whole group (all matrices simultaneously).
    //   - The effective contact rate per location is C_eff = C_baseline - D*(C_baseline - C_minimum),
    //     and the ODE model receives the sum over all locations.

    return 0;
}
