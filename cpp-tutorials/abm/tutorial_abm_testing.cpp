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
 * @file tutorial_abm_testing.cpp
 * @brief ABM Tutorial 2: Adding testing strategies as non-pharmaceutical interventions.
 *
 * This tutorial builds on Tutorial 1 (households) and adds a testing strategy
 * that acts as a non-pharmaceutical intervention (NPI). When persons enter
 * certain location types, they are tested with a given probability. Positive
 * tests cause the person to isolate, reducing transmission.
 *
 * This tutorial demonstrates how to:
 *  1. Set up the same household-based model from Tutorial 1.
 *  2. Create multiple social-event venues (spreading contacts).
 *  3. Define a TestingScheme with PCR test parameters.
 *  4. Attach the scheme to location types via a TestingStrategy.
 *  5. Run the simulation and compare results with the no-testing baseline.
 *
 * Key concept -- Testing in MEmilio-ABM:
 *   An element of TestingCriteria specifies which age groups and infection states trigger a test. 
 *   A TestingScheme wraps the criteria together with test parameters (sensitivity, specificity via TestData), 
 *   validity period, active time window, and testing probability. Schemes are added to location types
 *   through the model's TestingStrategy.
 */

#include "abm/household.h"
#include "abm/lockdown_rules.h"
#include "abm/model.h"
#include "abm/common_abm_loggers.h"
#include "parameter_setter.h"

#include <cstdlib>
#include <fstream>
#include <iostream>

// Age group constants used by both the testing helper and main().
const auto age_group_0_to_4   = mio::AgeGroup(0);
const auto age_group_5_to_14  = mio::AgeGroup(1);
const auto age_group_15_to_34 = mio::AgeGroup(2);
const auto age_group_35_to_59 = mio::AgeGroup(3);
const auto age_group_60_to_79 = mio::AgeGroup(4);
const auto age_group_80_plus  = mio::AgeGroup(5);

// *** Set up a PCR testing strategy and attach it to public locations. ***
// This function creates a single TestingScheme that:
//   - Is active for the full 30-day simulation window.
//   - Uses a PCR test (sensitivity/specificity come from TestData defaults).
//   - Tests are valid for `validity_days`; a negative result exempts retesting for validity_days days (Default: 3).
//   - Targets persons in any infected state (Exposed through Critical).
//   - Applies to all age groups except school children (5-14).
//   - Tests with `testing_probability` probability upon entry (Default: 100%).
//   - Applies to all public locations (SocialEvent, School, Work, BasicsShop)
void add_npi_testing_strategies_to_world(mio::abm::Model& model, double testing_probability, int validity_days)
{
    auto start_date_test = mio::abm::TimePoint(mio::abm::days(0).seconds());
    auto end_date_test   = mio::abm::TimePoint(mio::abm::days(30).seconds());

    // Retrieve built-in PCR test parameters (sensitivity, specificity, etc.).
    auto pcr_test            = mio::abm::TestType::PCR;
    auto pcr_test_parameters = model.parameters.get<mio::abm::TestData>()[pcr_test];
    auto validity            = mio::abm::days(validity_days);

    // Infection states that should trigger a test.
    auto states_to_test = std::vector<mio::abm::InfectionState>{
        mio::abm::InfectionState::Exposed, mio::abm::InfectionState::InfectedNoSymptoms,
        mio::abm::InfectionState::InfectedSymptoms, mio::abm::InfectionState::InfectedSevere,
        mio::abm::InfectionState::InfectedCritical};

    // Age groups subject to testing (all except school children 5-14).
    auto ages_to_test = std::vector<mio::AgeGroup>{age_group_0_to_4, age_group_15_to_34, age_group_35_to_59,
                                                   age_group_60_to_79, age_group_80_plus};

    auto testing_criteria = mio::abm::TestingCriteria(ages_to_test, states_to_test);
    auto testing_scheme   = mio::abm::TestingScheme(testing_criteria, validity, start_date_test, end_date_test,
                                                  pcr_test_parameters, testing_probability);

    // Attach the scheme to all public location types.
    model.get_testing_strategy().add_scheme(mio::abm::LocationType::SocialEvent, testing_scheme);
    model.get_testing_strategy().add_scheme(mio::abm::LocationType::School, testing_scheme);
    model.get_testing_strategy().add_scheme(mio::abm::LocationType::Work, testing_scheme);
    model.get_testing_strategy().add_scheme(mio::abm::LocationType::BasicsShop, testing_scheme);
}

int main(int argc, char* argv[])
{
    // Usage: tutorial_abm_tests [testing_prob] [validity_days] [n_households]
    //   testing_prob  : probability of testing at location entry (default: 1.0)
    //   validity_days : days a negative test stays valid         (default: 3)
    //   n_households  : number of each household type            (default: 125)
    double arg_testing_prob = (argc > 1) ? std::atof(argv[1]) : 1.0;
    int arg_validity_days   = (argc > 2) ? std::atoi(argv[2]) : 3;
    int arg_n_households    = (argc > 3) ? std::atoi(argv[3]) : 125;

    // Suppress verbose log output; only warnings and errors are shown.
    mio::set_log_level(mio::LogLevel::warn);

    // *** Define age groups. ***
    size_t num_age_groups = 6;

    // *** Create the model and set infection parameters. ***
    auto model = mio::abm::Model(num_age_groups);
    set_local_parameters(model);
    set_world_parameters(model.parameters);

    // Define which age groups are eligible to go to school and to work.
    // The AgeGroupGotoSchool / AgeGroupGotoWork arrays default to false for
    // every group; we enable only the relevant ones.
    model.parameters.get<mio::abm::AgeGroupGotoSchool>()                    = false;
    model.parameters.get<mio::abm::AgeGroupGotoSchool>()[age_group_5_to_14] = true;

    model.parameters.get<mio::abm::AgeGroupGotoWork>().set_multiple({age_group_15_to_34, age_group_35_to_59}, true);

    // *** Define HouseholdMember types (same as Tutorial 1). ***
    //
    // CLI parameters (see usage at top of main):
    //   argv[1] = testing_prob       (0.0 = no testing, 1.0 = always test)
    //   argv[2] = validity_days      (test validity period: 1, 3, 7)
    //   argv[3] = n_households        (population size: 50, 125, 500)

    // child: equally likely to be 0-4 or 5-14 years old (weights 1 and 1).
    auto child = mio::abm::HouseholdMember(num_age_groups);
    child.set_age_weight(age_group_0_to_4, 1);
    child.set_age_weight(age_group_5_to_14, 1);

    // parent: equally likely to be 15-34 or 35-59 years old.
    auto parent = mio::abm::HouseholdMember(num_age_groups);
    parent.set_age_weight(age_group_15_to_34, 1);
    parent.set_age_weight(age_group_35_to_59, 1);

    // single adult: exclusively in the 35-59 age group (weight 1 only there).
    // This illustrates how to pin a member slot to one specific age group.
    auto single_adult = mio::abm::HouseholdMember(num_age_groups);
    single_adult.set_age_weight(age_group_35_to_59, 1);

    // senior: equally likely to be 60-79 or 80+ years old.
    auto senior_adult = mio::abm::HouseholdMember(num_age_groups);
    senior_adult.set_age_weight(age_group_60_to_79, 1);
    senior_adult.set_age_weight(age_group_80_plus, 1);

    // *** Compose households and add them to the model. ***
    int n_households = arg_n_households;

    // --- Type A: two-person household (1 parent + 1 child) -------------------
    auto twoPersonHousehold = mio::abm::Household();
    twoPersonHousehold.add_members(child, 1);
    twoPersonHousehold.add_members(parent, 1);

    auto twoPersonGroup = mio::abm::HouseholdGroup();
    twoPersonGroup.add_households(twoPersonHousehold, n_households);
    add_household_group_to_model(model, twoPersonGroup);

    // --- Type B: three-person household (2 parents + 1 child) ----------------
    auto threePersonHousehold = mio::abm::Household();
    threePersonHousehold.add_members(child, 1);
    threePersonHousehold.add_members(parent, 2);

    auto threePersonGroup = mio::abm::HouseholdGroup();
    threePersonGroup.add_households(threePersonHousehold, n_households);
    add_household_group_to_model(model, threePersonGroup);

    // --- Type C: single-adult household (1 adult, no children) ---------------
    // This group is entirely in the 35-59 age group because single_adult has
    // weight 1 only for age_group_35_to_59.
    auto singleAdultHousehold = mio::abm::Household();
    singleAdultHousehold.add_members(single_adult, 1);

    auto singleAdultGroup = mio::abm::HouseholdGroup();
    singleAdultGroup.add_households(singleAdultHousehold, n_households);
    add_household_group_to_model(model, singleAdultGroup);

    // --- Type D: Senior & ELderly household (2 adult, no children) ---------------
    auto seniorAdultHousehold = mio::abm::Household();
    seniorAdultHousehold.add_members(senior_adult, 2);

    auto seniorAdultGroup = mio::abm::HouseholdGroup();
    seniorAdultGroup.add_households(seniorAdultHousehold, n_households);
    add_household_group_to_model(model, seniorAdultGroup);

    // *** Add locations. ***
    // Compared to Tutorial 1, we now create 10 social-event venues instead of
    // one. This distributes contacts across venues and makes the testing
    // strategy more realistic.

    // One hospital and one ICU shared by all persons.
    auto hospital = model.add_location(mio::abm::LocationType::Hospital);
    auto icu      = model.add_location(mio::abm::LocationType::ICU);

    // 10 social event venues (e.g. a community centre).
    auto social_event_venues = std::vector<mio::abm::LocationId>();
    for (int i = 0; i < 10; ++i) {
        social_event_venues.push_back(model.add_location(mio::abm::LocationType::SocialEvent));
    }

    // One supermarket.
    auto shop = model.add_location(mio::abm::LocationType::BasicsShop);

    // One school for all school-age children.
    auto school = model.add_location(mio::abm::LocationType::School);

    // One workplace for all working adults.
    auto work = model.add_location(mio::abm::LocationType::Work);

    // *** Assign initial infection states. ***
    //
    //  Index | InfectionState          | Probability
    //  ------|-------------------------|------------
    //    0   | Susceptible             | 0.8
    //    1   | Exposed                 | 0.05
    //    2   | InfectedNoSymptoms      | 0.1
    //    3   | InfectedSymptoms        | 0.05
    //    4   | InfectedSevere          | 0.0
    //    5   | InfectedCritical        | 0.0
    //    6   | Recovered               | 0.0
    //    7   | Dead                    | 0.0
    auto start_date = mio::abm::TimePoint(0); // t = 0 s from the simulation epoch

    std::vector<ScalarType> infection_distribution{0.8, 0.05, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0};

    for (auto& person : model.get_persons()) {
        // Draw an infection state from the distribution above.
        mio::abm::InfectionState infection_state = mio::abm::InfectionState(
            mio::DiscreteDistribution<size_t>::get_instance()(mio::thread_local_rng(), infection_distribution));

        auto rng = mio::abm::PersonalRandomNumberGenerator(person);
        if (infection_state != mio::abm::InfectionState::Susceptible) {
            // Infect an agent with the drawn state
            person.add_new_infection(mio::abm::Infection(rng, mio::abm::VirusVariant::Wildtype, person.get_age(),
                                                         model.parameters, start_date, infection_state));
        }
    }

    // *** Assign locations to persons. ***
    for (auto& person : model.get_persons()) {
        const auto id = person.get_id();

        // Shared locations.
        model.assign_location(id, shop);
        model.assign_location(id, hospital);
        model.assign_location(id, icu);

        // Each person is assigned to one of the 10 social event venues.
        model.assign_location(id, social_event_venues[person.get_id().get() % social_event_venues.size()]);

        // Age-specific locations.
        if (person.get_age() == age_group_5_to_14) {
            model.assign_location(id, school);
        }
        if (person.get_age() == age_group_15_to_34 || person.get_age() == age_group_35_to_59) {
            model.assign_location(id, work);
        }
    }

    // *** Add testing strategies. ***
    // This is the key addition compared to Tutorial 1. The helper function
    // defined above attaches a PCR testing scheme to social events, school,
    // work, and shops. Persons who test positive will isolate at home.
    add_npi_testing_strategies_to_world(model, arg_testing_prob, arg_validity_days);

    // *** Run the simulation. ***
    auto t0   = mio::abm::TimePoint(0);
    auto tmax = t0 + mio::abm::days(30);
    auto sim  = mio::abm::Simulation(t0, std::move(model));

    mio::History<mio::abm::TimeSeriesWriter, mio::abm::LogInfectionState> historyTimeSeries{
        Eigen::Index(mio::abm::InfectionState::Count)};

    sim.advance(tmax, historyTimeSeries);

    // *** Write results to file. ***
    std::ofstream outfile("abm_tests.txt");
    std::get<0>(historyTimeSeries.get_log())
        .print_table(outfile, {"S", "E", "I_NS", "I_Sy", "I_Sev", "I_Crit", "R", "D"}, 7, 4);
    std::cout << "Results written to abm_tests.txt\n";

    return 0;
}