/*
* Copyright (C) 2020-2026 MEmilio
*
* Authors: Khoa Nguyen
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
 * @file tutorial_abm_household1.cpp
 * @brief Tutorial 1: Setting up an ABM model with household data.
 *
 * This tutorial demonstrates how to:
 *  1. Define age groups and model parameters.
 *  2. Create HouseholdMember types with weighted age distributions.
 *  3. Compose Household templates from HouseholdMember types.
 *  4. Group Household templates into HouseholdGroups and add them to the model.
 *  5. Add and configure locations (school, work, shop, etc.).
 *  6. Assign initial infection states and locations to persons.
 *  7. Run a short simulation and write the results to a file.
 *
 * Key concept – HouseholdMember age weights:
 *   Each HouseholdMember carries an integer weight for every AgeGroup.
 *   When a person is created for that member slot, the model draws its age
 *   from a discrete distribution proportional to these weights.
 *   Example: weights {1, 1, 0, 0} give a 50 / 50 chance of AgeGroup 0 or 1.
 *   Weights {0, 0, 1, 2} give a 1/3 chance of AgeGroup 2 and 2/3 of AgeGroup 3.
 */

#include "abm/household.h"
#include "abm/lockdown_rules.h"
#include "abm/model.h"
#include "abm/common_abm_loggers.h"
#include "parameter_setter.h"

#include <fstream>
#include <iostream>
const auto age_group_0_to_4   = mio::AgeGroup(0); // toddlers / kindergarten
const auto age_group_5_to_14  = mio::AgeGroup(1); // school children
const auto age_group_15_to_34 = mio::AgeGroup(2); // young adults
const auto age_group_35_to_59 = mio::AgeGroup(3); // middle-aged adults
const auto age_group_60_to_79 = mio::AgeGroup(4); // seniors
const auto age_group_80_plus  = mio::AgeGroup(5); // elderly

void add_npi_testing_strategies_to_world(mio::abm::Model& model)
{
    double testing_probability        = 0.0; // 50% of eligible persons will be tested at each time step
    auto start_date_test              = mio::abm::TimePoint(mio::abm::days(0).seconds());
    auto end_date_test                =  mio::abm::TimePoint(mio::abm::days(99999).seconds());

    auto antigen_test = mio::abm::TestType::PCR;
    auto antigen_test_parameters =
        model.parameters.get<mio::abm::TestData>()[antigen_test]; // Test parameters
    auto validity     = mio::abm::days(1);
    auto states_to_test  = std::vector<mio::abm::InfectionState>{mio::abm::InfectionState::InfectedSymptoms,
                                                                        mio::abm::InfectionState::InfectedNoSymptoms,   
                                                                      mio::abm::InfectionState::InfectedSevere,
                                                                      mio::abm::InfectionState::InfectedCritical};

    auto ages_to_test = std::vector<mio::AgeGroup>{age_group_0_to_4, age_group_15_to_34, age_group_35_to_59, age_group_60_to_79, age_group_80_plus};
    auto testing_criteria = mio::abm::TestingCriteria(ages_to_test, states_to_test);
    auto testing_scheme = mio::abm::TestingScheme(testing_criteria, validity, start_date_test, end_date_test, antigen_test_parameters, testing_probability);

    // social events
    model.get_testing_strategy().add_scheme(mio::abm::LocationType::SocialEvent, testing_scheme);
    model.get_testing_strategy().add_scheme(mio::abm::LocationType::School, testing_scheme);
    model.get_testing_strategy().add_scheme(mio::abm::LocationType::Work, testing_scheme);
    model.get_testing_strategy().add_scheme(mio::abm::LocationType::BasicsShop, testing_scheme);
}

int main()
{
    mio::set_log_level(mio::LogLevel::warn);

    // -------------------------------------------------------------------------
    // 1. Age groups
    // -------------------------------------------------------------------------
    // We use six age groups covering children and adults younger than 60.
    // The index passed to mio::AgeGroup must match the order in which the model
    // was constructed (0-based).
    size_t num_age_groups         = 6;


    // -------------------------------------------------------------------------
    // 2. Model and global infection parameters
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // 3. HouseholdMember types
    // -------------------------------------------------------------------------
    // A HouseholdMember is not a person itself; it is a *template* that
    // describes the age distribution of whoever fills that role in a household.
    // The weights are integers; the probability of each age group is
    //   P(AgeGroup i) = weight_i / sum_of_all_weights.

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

    // -------------------------------------------------------------------------
    // 4. Household templates and HouseholdGroups
    // -------------------------------------------------------------------------
    // A Household collects one or more (member_type, count) pairs.
    // set_space_per_member sets the volume per person in m³, which affects
    // aerosol transmission when UseLocationCapacityForTransmissions is true.
    int n_households = 125; // number of each household type to add to the model

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

    // -------------------------------------------------------------------------
    // 5. Locations
    // -------------------------------------------------------------------------
    // Locations are added one at a time.  The returned LocationId is used later
    // to assign persons and to read infection parameters.
    // MaximumContacts caps the total contact rate at a location: if the sum of
    // all ContactRates entries exceeds this value, every rate is scaled down
    // proportionally (see adjust_contact_rates in model_functions.cpp).


    // One hospital and one ICU shared by all persons.
    auto hospital = model.add_location(mio::abm::LocationType::Hospital);
    auto icu = model.add_location(mio::abm::LocationType::ICU);

    // 10 social event venues (e.g. a community centre).
    auto social_event_venues = std::vector<mio::abm::LocationId>();
    for (int i = 0; i < 10; ++i) {
        social_event_venues.push_back(model.add_location(mio::abm::LocationType::SocialEvent));
    }
    
    // // One supermarket.  Groceries shops allow up to 20 simultaneous contacts.
    auto shop = model.add_location(mio::abm::LocationType::BasicsShop);

    // One school for all school-age children.
    auto school = model.add_location(mio::abm::LocationType::School);

    // // One workplace for all working adults.
    auto work = model.add_location(mio::abm::LocationType::Work);


    // -------------------------------------------------------------------------
    // 6. Initial infection states
    // -------------------------------------------------------------------------
    // We assign each person a random infection state drawn from the discrete
    // distribution below.  Persons who are not Susceptible receive a full
    // Infection object so their viral-load course and state transitions are
    // properly initialised.
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

    std::vector<ScalarType> infection_distribution{0.8, 0.05, 0.1, 0.05, 0.00, 0.00, 0.0, 0.00};

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

    // -------------------------------------------------------------------------
    // 7. Location assignment
    // -------------------------------------------------------------------------
    // Every person must be assigned to at least: their home (done automatically
    // by add_household_to_model), a shop, a social-event venue, a hospital, and
    // an ICU.  School-age children also need a school; working adults need a
    // workplace.  The model's mobility rules then move persons between their
    // assigned locations according to the time of day.
    for (auto& person : model.get_persons()) {
        const auto id = person.get_id();

        // Shared locations – everyone has access to these.
        model.assign_location(id, shop);
        model.assign_location(id, hospital);
        model.assign_location(id, icu);

        // // Each person is assigned to one of the 10 social event venues at random.
        model.assign_location(id, social_event_venues[person.get_id().get() % social_event_venues.size()]);


        // Age-specific locations.
        if (person.get_age() == age_group_5_to_14) {
            model.assign_location(id, school);
        }
        if (person.get_age() == age_group_15_to_34 ||
            person.get_age() == age_group_35_to_59) {
            model.assign_location(id, work);
        }
    }

    // -------------------------------------------------------------------------
    // 7.5 Add teting strategies
    // -------------------------------------------------------------------------
    add_npi_testing_strategies_to_world(model); 

    // -------------------------------------------------------------------------
    // 8. Run the simulation
    // -------------------------------------------------------------------------
    auto t0   = mio::abm::TimePoint(0);
    auto tmax = t0 + mio::abm::days(10);
    auto sim  = mio::abm::Simulation(t0, std::move(model));

    // The History object accumulates one time-series entry per simulation step.
    // LogInfectionState counts persons in each InfectionState across all
    // locations at every logged time point.
    mio::History<mio::abm::TimeSeriesWriter, mio::abm::LogInfectionState> historyTimeSeries{
        Eigen::Index(mio::abm::InfectionState::Count)};

    sim.advance(tmax, historyTimeSeries);
    // sim.advance(tmax);

    // -------------------------------------------------------------------------
    // 9. Write results
    // -------------------------------------------------------------------------
    // The output file contains a table with 9 columns:
    //   Time  S  E  I_NS  I_Sy  I_Sev  I_Crit  R  D
    // where Time is in days and the remaining columns are person counts.
    std::ofstream outfile("abm_household.txt");
    std::get<0>(historyTimeSeries.get_log())
        .print_table(outfile,
                     {"S", "E", "I_NS", "I_Sy", "I_Sev", "I_Crit", "R", "D"},
                     7, 4);
    std::cout << "Results written to abm_household.txt\n";

    return 0;
}