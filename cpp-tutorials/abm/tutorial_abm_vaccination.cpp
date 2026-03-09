/*
* Copyright (C) 2020-2026 MEmilio
*
* Authors: Sascha Korf
*
* Contact: Martin J. Kuehn <Martin.Kuehn@DLR.de>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/**
 * @file tutorial_abm_vacc3.cpp
 * @brief ABM Tutorial 3: Adding vaccinations to the simulation.
 *
 * This tutorial builds on Tutorials 1 and 2 and introduces vaccinations.
 * Each person can receive one or more vaccinations via add_new_vaccination().
 * A vaccination records a ProtectionType (e.g. GenericVaccine) and the
 * TimePoint it was administered. Protection factors -- piecewise-linear
 * functions of days since vaccination -- reduce the probability of infection,
 * severe disease, and high viral load.
 *
 * This tutorial demonstrates how to:
 *  1. Set vaccination protection parameters (InfectionProtectionFactor,
 *     SeverityProtectionFactor) per age group and virus variant.
 *  2. Run an age-prioritised vaccination campaign (elderly first).
 *  3. Read real-world vaccination data from a JSON file
 *     (vacc_county_ageinf_ma7.json) and apply it day-by-day.
 *  4. Run the simulation and write results.
 */

#include "abm/household.h"
#include "abm/lockdown_rules.h"
#include "abm/model.h"
#include "abm/common_abm_loggers.h"
#include "abm/protection_event.h"
#include "memilio/io/epi_data.h"
#include "parameter_setter.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <vector>

// ============================================================================
// Helper: Map RKI age group index to our 6 age groups (identity mapping here).
// ============================================================================
size_t rki_age_to_index(mio::AgeGroup age)
{
    return static_cast<size_t>(age.get());
}

// ============================================================================
// Helper: Read the vaccination JSON and build a map
//   date -> vector<uint32_t> (daily new first-dose vaccinations per age group).
// Cumulative counts in the file are converted to daily increments.
// Only first-dose ("Vacc_partially") data is used; boosters are ignored.
// This mirrors prepare_vaccination_state() from the paper code.
// ============================================================================
std::map<mio::Date, std::vector<uint32_t>>
prepare_vaccination_data(const std::string& filename, size_t num_age_groups,
                         int county_id)
{
    // Previous day's cumulative first-dose count per age group.
    // Initialised to UINT32_MAX so we can detect the very first entry and
    // use it as the baseline (its "delta" would otherwise be the entire
    // cumulative total, which is far too large).
    std::vector<uint32_t> prev(num_age_groups, std::numeric_limits<uint32_t>::max());
    std::map<mio::Date, std::vector<uint32_t>> vacc_map;

    auto vacc_data = mio::read_vaccination_data(filename);
    if (!vacc_data) {
        std::cerr << "Warning: could not read vaccination file " << filename
                  << " – skipping data-driven vaccination.\n";
        return vacc_map;
    }

    for (auto& entry : vacc_data.value()) {
        // Filter by county.
        if (!entry.county_id || entry.county_id.value() != mio::regions::CountyId(county_id))
            continue;

        size_t age_idx = rki_age_to_index(entry.age_group);
        if (age_idx >= num_age_groups)
            continue;

        // Create date slot if needed.
        if (vacc_map.find(entry.date) == vacc_map.end())
            vacc_map[entry.date] = std::vector<uint32_t>(num_age_groups, 0);

        // First entry for this age group: use as baseline, record 0.
        if (prev[age_idx] == std::numeric_limits<uint32_t>::max()) {
            prev[age_idx] = static_cast<uint32_t>(entry.num_vaccinations_partial);
            continue;
        }

        // Daily new first-dose = cumulative today – cumulative yesterday.
        auto partial_new = static_cast<int>(entry.num_vaccinations_partial) -
                           static_cast<int>(prev[age_idx]);

        vacc_map[entry.date][age_idx] = static_cast<uint32_t>(std::max(0, partial_new));

        prev[age_idx] = static_cast<uint32_t>(entry.num_vaccinations_partial);
    }

    return vacc_map;
}

// ============================================================================
// Helper: Apply the data-driven vaccination map to the model's persons.
// For each day in the map, we vaccinate the specified number of new first-dose
// recipients per age group (no booster / second dose).
// This mirrors assign_vaccination_state() from the paper code.
// ============================================================================
void apply_vaccination_data(mio::abm::Model& model, size_t num_age_groups,
                            const mio::Date& sim_start_date,
                            const std::map<mio::Date, std::vector<uint32_t>>& vacc_map)
{
    // Build per-age-group lists of person indices (unvaccinated pool).
    std::vector<std::vector<size_t>> persons_by_age(num_age_groups);
    {
        size_t idx = 0;
        for (auto& person : model.get_persons()) {
            persons_by_age[person.get_age().get()].push_back(idx);
            ++idx;
        }
    }

    std::mt19937 gen{std::random_device{}()};
    size_t total     = 0;
    size_t requested = 0;

    for (auto& [date, age_counts] : vacc_map) {
        // Convert calendar date to simulation TimePoint.
        int day_offset = mio::get_offset_in_days(date, sim_start_date);
        if (day_offset < 0)
            continue; // date is before simulation start
        auto tp = mio::abm::TimePoint(day_offset * 24 * 60 * 60) - mio::abm::days(92);

        for (size_t age = 0; age < age_counts.size(); ++age) {
            uint32_t n_first_dose = age_counts[age];
            if (n_first_dose == 0)
                continue;

            auto& pool = persons_by_age[age];
            std::shuffle(pool.begin(), pool.end(), gen);

            uint32_t done = 0;
            size_t i      = 0;
            while (i < pool.size() && done < n_first_dose) {
                auto persons_range = model.get_persons();
                auto it            = persons_range.begin();
                std::advance(it, pool[i]);
                auto& person = *it;
                    person.add_new_vaccination(
                        mio::abm::ProtectionType::GenericVaccine, tp);
                    // Remove vaccinated person from pool to avoid double-vaccination.
                    pool.erase(pool.begin() + i);
                    ++done;
            }
            total += done;
        }
    }
    size_t population = model.get_persons().size();
    std::cout << "Data-driven vaccination: " << total << " out of " << population
              << " persons vaccinated\n";
}

int main(int argc, char* argv[])
{
    // Usage: tutorial_abm_vaccination [vaccination_rate] [n_households] [protection_peak] [use_data_vacc]
    //   vaccination_rate  : fraction of each age group vaccinated pre-sim (default: 0.0)
    //   n_households      : number of each household type               (default: 1000)
    //   protection_peak   : peak infection protection factor            (default: 0.67)
    //   use_data_vacc     : 1 = apply JSON data-driven vaccination, 0 = skip (default: 1)
    double arg_vacc_rate       = (argc > 1) ? std::atof(argv[1]) : 0.0;
    int    arg_n_households    = (argc > 2) ? std::atoi(argv[2]) : 1000;
    double arg_protection_peak = (argc > 3) ? std::atof(argv[3]) : 0.67;
    int    arg_use_data_vacc   = (argc > 4) ? std::atoi(argv[4]) : 1;

    // Suppress verbose log output; only warnings and errors are shown.
    mio::set_log_level(mio::LogLevel::warn);

    // *** Define age groups. ***
    size_t num_age_groups         = 6;
    const auto age_group_0_to_4   = mio::AgeGroup(0); // toddlers / kindergarten
    const auto age_group_5_to_14  = mio::AgeGroup(1); // school children
    const auto age_group_15_to_34 = mio::AgeGroup(2); // young adults
    const auto age_group_35_to_59 = mio::AgeGroup(3); // middle-aged adults
    const auto age_group_60_to_79 = mio::AgeGroup(4); // seniors
    const auto age_group_80_plus  = mio::AgeGroup(5); // elderly

    // *** Create the model and set infection parameters (same as Tutorial 1). ***
    auto model = mio::abm::Model(num_age_groups);
    set_local_parameters(model);
    set_world_parameters(model.parameters);

    // Define which age groups are eligible to go to school and to work.
    // The AgeGroupGotoSchool / AgeGroupGotoWork arrays default to false for
    // every group; we enable only the relevant ones.
    model.parameters.get<mio::abm::AgeGroupGotoSchool>()                    = false;
    model.parameters.get<mio::abm::AgeGroupGotoSchool>()[age_group_5_to_14] = true;

    model.parameters.get<mio::abm::AgeGroupGotoWork>().set_multiple(
        {age_group_15_to_34, age_group_35_to_59}, true);

    // *** Set vaccination protection parameters. ***
    // Protection factors are piecewise-linear functions of days since
    // vaccination, stored per {ProtectionType, AgeGroup, VirusVariant}.
    // Values range from 0 (no protection) to 1 (full protection).
    // We set them for GenericVaccine x every age group x Wildtype variant.
    //
    // CLI parameters (see usage at top of main):
    //   argv[1] = vaccination_rate       (0.0, 0.3, 0.7)
    //   argv[2] = n_households           (125, 500, 1000)
    //   argv[3] = protection_peak        (0.3, 0.67, 0.95)

    for (auto age = mio::AgeGroup(0); age < mio::AgeGroup(num_age_groups); ++age) {
        // Infection protection: ramps up to 0.67 at day 14, wanes to 0.45 by day 180.
        model.parameters.get<mio::abm::InfectionProtectionFactor>()[{
            mio::abm::ProtectionType::GenericVaccine, age,
            mio::abm::VirusVariant::Wildtype}] =
            mio::TimeSeriesFunctor<ScalarType>{
                mio::TimeSeriesFunctorType::LinearInterpolation,
                {{0, 0.0}, {14, arg_protection_peak}, {180, arg_protection_peak * 0.67}}};

        // Severity protection: ramps up to 0.85 at day 14, wanes to 0.70 by day 180.
        model.parameters.get<mio::abm::SeverityProtectionFactor>()[{
            mio::abm::ProtectionType::GenericVaccine, age,
            mio::abm::VirusVariant::Wildtype}] =
            mio::TimeSeriesFunctor<ScalarType>{
                mio::TimeSeriesFunctorType::LinearInterpolation,
                {{0, 0.0}, {1, 0.85}, {180, 0.70}}};
    }

    std::cout << "Vaccination protection factors configured.\n";

    // *** Define HouseholdMember types (same as Tutorial 1). ***

    // child: equally likely to be 0-4 or 5-14 years old (weights 1 and 1).
    auto child = mio::abm::HouseholdMember(num_age_groups);
    child.set_age_weight(age_group_0_to_4,  1);
    child.set_age_weight(age_group_5_to_14, 1);

    // parent: equally likely to be 15-34 or 35-59 years old.
    auto parent = mio::abm::HouseholdMember(num_age_groups);
    parent.set_age_weight(age_group_15_to_34, 1);
    parent.set_age_weight(age_group_35_to_59, 1);

    // single adult: exclusively in the 35-59 age group (weight 1 only there).
    // This illustrates how to pin a member slot to one specific age group.
    auto single_adult = mio::abm::HouseholdMember(num_age_groups);
    single_adult.set_age_weight(age_group_35_to_59, 1);

    auto senior_adult = mio::abm::HouseholdMember(num_age_groups);
    senior_adult.set_age_weight(age_group_60_to_79, 1);
    senior_adult.set_age_weight(age_group_80_plus, 1);

    // *** Compose households and add them to the model. ***
    int n_households = arg_n_households;

    // --- Type A: two-person household (1 parent + 1 child) -------------------
    auto twoPersonHousehold = mio::abm::Household();
    twoPersonHousehold.add_members(child,  1);
    twoPersonHousehold.add_members(parent, 1);

    auto twoPersonGroup = mio::abm::HouseholdGroup();
    twoPersonGroup.add_households(twoPersonHousehold, n_households);
    add_household_group_to_model(model, twoPersonGroup);

    // --- Type B: three-person household (2 parents + 1 child) ----------------
    auto threePersonHousehold = mio::abm::Household();
    threePersonHousehold.add_members(child,  1);
    threePersonHousehold.add_members(parent, 2);

    auto threePersonGroup = mio::abm::HouseholdGroup();
    threePersonGroup.add_households(threePersonHousehold, n_households);
    add_household_group_to_model(model, threePersonGroup);

    // --- Type C: single-adult household (1 adult, no children) ---------------
    // This group is entirely in the 35-59 age group because single_adult has
    // weight 1 only for age_group_35_to_59.
    auto singleAdultHousehold = mio::abm::Household();
    singleAdultHousehold.add_members(single_adult, 1);

    // --- Type D: Senior & ELderly household (2 adult, no children) ---------------
    auto seniorAdultHousehold = mio::abm::Household();
    seniorAdultHousehold.add_members(senior_adult, 2);

    auto seniorAdultGroup = mio::abm::HouseholdGroup();
    seniorAdultGroup.add_households(seniorAdultHousehold, n_households);
    add_household_group_to_model(model, seniorAdultGroup);

    // *** Add locations (same as Tutorial 1). ***


    // One hospital and one ICU shared by all persons.
    auto hospital = model.add_location(mio::abm::LocationType::Hospital);
    auto icu = model.add_location(mio::abm::LocationType::ICU);

    // One social event venue (e.g. a community centre).
    auto event = model.add_location(mio::abm::LocationType::SocialEvent);

    // One supermarket.  Groceries shops allow up to 20 simultaneous contacts.
    auto shop = model.add_location(mio::abm::LocationType::BasicsShop);

    // One school for all school-age children.
    auto school = model.add_location(mio::abm::LocationType::School);

    // One workplace for all working adults.
    auto work = model.add_location(mio::abm::LocationType::Work);


    // *** Assign initial infection states. ***
    //
    //  Index | InfectionState          | Probability
    //  ------|-------------------------|------------
    //    0   | Susceptible             | 0.80
    //    1   | Exposed                 | 0.10
    //    2   | InfectedNoSymptoms      | 0.01
    //    3   | InfectedSymptoms        | 0.01
    //    4   | InfectedSevere          | 0.01
    //    5   | InfectedCritical        | 0.01
    //    6   | Recovered               | 0.00
    //    7   | Dead                    | 0.06
    auto start_date = mio::abm::TimePoint(0); // t = 0 s from the simulation epoch
    
    std::vector<ScalarType> infection_distribution{0.8, 0.05, 0.1, 0.05, 0.00, 0.00, 0.00, 0.00};

    for (auto& person : model.get_persons()) {
        // Draw an infection state from the distribution above.
        mio::abm::InfectionState infection_state = mio::abm::InfectionState(
            mio::DiscreteDistribution<size_t>::get_instance()(
                mio::thread_local_rng(), infection_distribution));
        
        auto rng = mio::abm::PersonalRandomNumberGenerator(person);
        if (infection_state != mio::abm::InfectionState::Susceptible) {
            // Infect an agent with the drawn state
            person.add_new_infection(
                mio::abm::Infection(rng, mio::abm::VirusVariant::Wildtype,
                                    person.get_age(), model.parameters,
                                    start_date, infection_state));
        }
    }

    // *** Assign locations to persons. ***
    for (auto& person : model.get_persons()) {
        const auto id = person.get_id();

        // Shared locations – everyone has access to these.
        model.assign_location(id, event);
        model.assign_location(id, shop);
        model.assign_location(id, hospital);
        model.assign_location(id, icu);

        // Age-specific locations.
        if (person.get_age() == age_group_5_to_14) {
            model.assign_location(id, school);
        }
        if (person.get_age() == age_group_15_to_34 ||
            person.get_age() == age_group_35_to_59) {
            model.assign_location(id, work);
        }
    }

    // *** Vaccination: either age-prioritised campaign OR data-driven. ***
    // argv[4] selects the mode:
    //   0 = campaign only  (use vaccination_rate to vaccinate before sim start)
    //   1 = data-driven    (read real vaccination counts from JSON file)

    if (!arg_use_data_vacc) {
        // ── Campaign mode ────────────────────────────────────────────
        // Vaccinate a fraction of each eligible age group at t = -20 days,
        // prioritising elderly first.
        const double vaccination_rate = arg_vacc_rate;
        const auto   vaccination_time = start_date - mio::abm::days(20);
        auto population = 0;
        std::vector<std::vector<size_t>> persons_by_age(num_age_groups);
        {
            size_t idx = 0;
            for (auto& person : model.get_persons()) {
                persons_by_age[person.get_age().get()].push_back(idx);
                ++idx;
            }
            population =0;
        }
       
        std::vector<mio::AgeGroup> vaccination_priority = {
            age_group_80_plus, age_group_60_to_79, age_group_35_to_59, age_group_15_to_34};

        size_t total_vaccinated = 0;
        for (auto age : vaccination_priority) {
            auto& indices = persons_by_age[age.get()];
            std::shuffle(indices.begin(), indices.end(),
                         std::mt19937{std::random_device{}()});
            size_t n_to_vaccinate = static_cast<size_t>(
                std::round(vaccination_rate * indices.size()));

            size_t count = 0;
            for (size_t i = 0; i < n_to_vaccinate; ++i) {
                auto persons_range = model.get_persons();
                auto it            = persons_range.begin();
                std::advance(it, indices[i]);
                auto& person = *it;
                if (person.get_infection_state(vaccination_time) ==
                    mio::abm::InfectionState::Susceptible) {
                    person.add_new_vaccination(
                        mio::abm::ProtectionType::GenericVaccine, vaccination_time);
                    ++count;
                }
            }
            total_vaccinated += count;
            std::cout << "Vaccinated " << count << " persons in age group "
                      << age.get() << "\n";
        }
        std::cout << "Campaign total vaccinated: " << total_vaccinated << "out of " << population <<"\n";
    }
    else {
        // ── Data-driven mode ─────────────────────────────────────────
        // Read real-world vaccination counts from a JSON file and apply
        // them day-by-day to the model's persons.
        const std::string vacc_json_path = "/Users/saschakorf/Documents/Promotion/memilio-tutorials/cpp-tutorials/abm/vacc_county_ageinf_ma7.json";
        const int         county_id      = 1002;
        const mio::Date   sim_start      = mio::Date(2020, 10, 1);

        auto vacc_map = prepare_vaccination_data(vacc_json_path, num_age_groups, county_id);

        if (!vacc_map.empty()) {
            std::cout << "Read " << vacc_map.size()
                      << " days of vaccination data from " << vacc_json_path << "\n";
            apply_vaccination_data(model, num_age_groups, sim_start, vacc_map);
        }
        else {
            std::cout << "No vaccination data loaded (file missing or empty).\n";
        }
    }

    // *** Run the simulation. ***
    auto t0   = mio::abm::TimePoint(0);
    auto tmax = t0 + mio::abm::days(30);
    auto sim  = mio::abm::Simulation(t0, std::move(model));

    mio::History<mio::abm::TimeSeriesWriter, mio::abm::LogInfectionState> historyTimeSeries{
        Eigen::Index(mio::abm::InfectionState::Count)};

    sim.advance(tmax, historyTimeSeries);

    // *** Write results to file. ***
    std::ofstream outfile("abm_vaccination.txt");
    std::get<0>(historyTimeSeries.get_log())
        .print_table(outfile,
                     {"S", "E", "I_NS", "I_Sy", "I_Sev", "I_Crit", "R", "D"},
                     7, 4);
    std::cout << "Results written to abm_vaccination.txt\n";

    return 0;
}