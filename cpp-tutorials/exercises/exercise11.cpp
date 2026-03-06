#include "ode_secir/model.h"
#include "memilio/compartments/simulation.h"
#include "memilio/data/analyze_result.h"

// *** Introduction ***
// In Tutorial 10, we applied NPIs at a predefined fixed time (day 20). In practice, interventions are often
// activated reactively and triggered when the number of infections exceeds a critical threshold, such as a
// specific incidence per 100,000 individuals.
//
// MEmilio supports this pattern through dynamic NPIs: a set of contact dampings that are automatically
// activated whenever a specified infection threshold is exceeded, remain active for a defined duration,
// and are then automatically lifted if the incidence is below the threshold again.
//
// Key parameters of a dynamic NPI:
//   threshold   -- incidence limit that triggers the NPI
//   base_value  -- reference population size for the incidence calculation (typically 100,000)
//   duration    -- minimum number of days the NPI stays active once triggered
//   interval    -- how often (in days) the incidence is re-evaluated

// Location indices (reused from Tutorial 10).
enum class Location : size_t
{
    Home   = 0,
    School = 1,
    Work   = 2,
    Other  = 3
};

ScalarType total_population = 100000;
ScalarType t0               = 0;
ScalarType tmax             = 100;
ScalarType dt               = 0.1;

// We reuse the create_model() helper from Tutorial 10, which already sets up location-specific contact
// matrices (baseline sum = 10 contacts/day).
mio::osecir::Model<ScalarType> create_model()
{
    mio::osecir::Model<ScalarType> m(1);

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
    contacts[(size_t)Location::Home] = mio::ContactMatrix<ScalarType>(Eigen::MatrixX<ScalarType>::Constant(1, 1, 4.0),
                                                                      Eigen::MatrixX<ScalarType>::Constant(1, 1, 1.0));
    contacts[(size_t)Location::School] =
        mio::ContactMatrix<ScalarType>(Eigen::MatrixX<ScalarType>::Constant(1, 1, 3.0));
    contacts[(size_t)Location::Work]  = mio::ContactMatrix<ScalarType>(Eigen::MatrixX<ScalarType>::Constant(1, 1, 2.0));
    contacts[(size_t)Location::Other] = mio::ContactMatrix<ScalarType>(Eigen::MatrixX<ScalarType>::Constant(1, 1, 1.0));
    m.parameters.get<mio::osecir::ContactPatterns<ScalarType>>() = contacts;

    // Initial populations: 0.5% Exposed, 0.5% InfectedNoSymptoms, rest Susceptible
    m.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::Exposed}]            = 0.005 * total_population;
    m.populations[{mio::AgeGroup(0), mio::osecir::InfectionState::InfectedNoSymptoms}] = 0.005 * total_population;
    m.populations.set_difference_from_total({mio::AgeGroup(0), mio::osecir::InfectionState::Susceptible},
                                            total_population);
    return m;
}

// Helper: create a DampingSampling for one location.
// Each DampingSampling describes one location-specific contact reduction:
//   value          -- damping coefficient
//   level          -- damping level (for combining multiple dampings)
//   type           -- damping type  (for combining multiple dampings)
//   time           -- time offset within the NPI duration
//   matrix_indices -- which location matrix to damp
//   group_weights  -- one entry per age group
mio::DampingSampling<ScalarType> loc_damping(ScalarType coefficient, Location location)
{
    return mio::DampingSampling<ScalarType>(mio::UncertainValue<ScalarType>(coefficient), mio::DampingLevel(0),
                                            mio::DampingType(0), mio::SimulationTime<ScalarType>(0.0),
                                            {(size_t)location}, Eigen::VectorX<ScalarType>::Ones(1));
}

int main()
{
    // *** Baseline Simulation Without NPIs ***
    // We first run the model without any interventions.
    auto model_baseline  = create_model();
    auto result_baseline = mio::osecir::simulate<ScalarType>(t0, tmax, dt, model_baseline);

    // *** Defining Dynamic NPIs ***
    // Dynamic NPIs are given to the model via DynamicNPIsInfectedSymptoms. The simulator checks every
    // interval days whether the current number of InfectedSymptoms relative to base_value exceeds a
    // threshold. If so, the corresponding set of DampingSampling objects is applied for duration days.
    //
    // We define two escalation levels:
    //   Level | Threshold (per 100k) | School | Work | Other
    //   Mild  | 500                  | 0.3    | 0.3  | 0.3
    //   Strict| 5000                 | 1.0    | 0.6  | 0.8

    // Mild restrictions (threshold: 500 per 100k)
    std::vector<mio::DampingSampling<ScalarType>> mild_npis = {
        loc_damping(0.3, Location::School), // school contacts reduced by 30%
        loc_damping(0.3, Location::Work), // work contacts reduced by 30%
        loc_damping(0.3, Location::Other), // other contacts reduced by 30%
    };

    // EXERCISE: Define strict_npis for the high-incidence threshold (5000 per 100k):
    //   School closure:              D = 1.0  (schools fully closed)
    //   Home-office mandate:         D = 0.6  (work contacts reduced by 60%)
    //   Public transport restriction: D = 0.8  (other contacts reduced by 80%)
    // Hint: follow the same pattern as mild_npis above using loc_damping().
    // std::vector<mio::DampingSampling<ScalarType>> strict_npis = { ??? };

    // *** Setting Up the Model with Dynamic NPIs ***
    // We create a new model and set the two dynamic NPIs. The simulator will automatically select the
    // highest exceeded threshold at each check interval.
    auto model_dynamic = create_model();
    auto& dyn_npis     = model_dynamic.parameters.get<mio::osecir::DynamicNPIsInfectedSymptoms<ScalarType>>();

    // EXERCISE: Configure the dynamic NPI mechanism and register both thresholds.
    //   interval   = 3 days   (how often the incidence is checked)
    //   duration   = 14 days  (minimum time an NPI stays active once triggered)
    //   base_value = 100000   (reference population for the incidence calculation)
    //   threshold 1: 500  per 100k -> mild_npis
    //   threshold 2: 5000 per 100k -> strict_npis
    // Hint: use dyn_npis.set_interval / set_duration / set_base_value / set_threshold.
    // ???

    // *** Simulation with Dynamic NPIs ***
    // To use the dynamic NPI checking, we must use the specific osecir::Simulation directly. The Simulation class
    // overrides advance() and includes a check for dynamic NPIs in regular intervals. We create the
    // simulation object and advance it to tmax.
    mio::osecir::Simulation<ScalarType> sim(model_dynamic, t0, dt);
    sim.advance(tmax);
    auto result_dynamic = sim.get_result();

    // *** Print results ***
    auto interp_baseline = mio::interpolate_simulation_result(result_baseline);
    auto interp_dynamic  = mio::interpolate_simulation_result(result_dynamic);

    std::cout << "\n--- Without NPIs ---\n";
    interp_baseline.print_table({"S", "E", "C", "C_confirmed", "I", "I_confirmed", "H", "U", "R", "D"}, 12, 4);
    std::cout << "\n--- With dynamic NPIs ---\n";
    interp_dynamic.print_table({"S", "E", "C", "C_confirmed", "I", "I_confirmed", "H", "U", "R", "D"}, 12, 4);

    // Optional: export to CSV for plotting in Python (see tutorial11.py for the corresponding visualization).
    // interp_baseline.export_csv("result_baseline.csv",
    //     {"S", "E", "C", "C_confirmed", "I", "I_confirmed", "H", "U", "R", "D"});
    // interp_dynamic.export_csv("result_dynamic.csv",
    //     {"S", "E", "C", "C_confirmed", "I", "I_confirmed", "H", "U", "R", "D"});

    // *** Summary ***
    // In this tutorial, we introduced dynamic NPIs: contact reductions which are triggered automatically
    // when an incidence threshold is exceeded. Key takeaways:
    //   - Dynamic NPIs are configured via model.parameters.get<DynamicNPIsInfectedSymptoms<FP>>().
    //   - Three control parameters determine the mechanism: interval (check frequency), duration (minimum
    //     active time), and base_value (reference population).
    //   - Each threshold is paired with a vector of DampingSampling objects that specify which location
    //     and how much to damp.
    //   - If multiple thresholds are defined, MEmilio automatically selects the highest exceeded threshold
    //     at each check.
    //   - Dynamic NPIs require using osecir::Simulation<FP> and sim.advance(tmax), since the threshold
    //     check is embedded in advance().

    return 0;
}
