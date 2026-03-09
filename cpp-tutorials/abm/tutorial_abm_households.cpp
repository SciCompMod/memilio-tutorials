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
 * @file tutorial_abm_households.cpp
 * @brief ABM Tutorial 1: Setting up an agent-based model with households.
 *
 * MEmilio provides an agent-based model (ABM) where each individual ("agent")
 * has an age, a home, and assigned locations they visit during the day.
 * Agents transition between infection states (Susceptible, Exposed, ..., Dead)
 * based on stochastic contact events at their current location.
 *
 * This tutorial demonstrates how to:
 *  1. Define age groups and configure model parameters.
 *  2. Create HouseholdMember types with weighted age distributions.
 *  3. Compose Household templates and add them to the model.
 *  4. Add locations (school, work, shop, hospital, etc.).
 *  5. Assign initial infection states and locations to persons.
 *  6. Run a 30-day simulation and write the results.
 *
 * Key concept -- HouseholdMember age weights:
 *   Each HouseholdMember carries an integer weight for every AgeGroup.
 *   When a person is created for that member slot, the model draws its age
 *   from a discrete distribution proportional to these weights.
 *   Example: weights {1, 1, 0, 0} give a 50/50 chance of AgeGroup 0 or 1.
 */

#include "abm/household.h"
#include "abm/lockdown_rules.h"
#include "abm/model.h"
#include "abm/common_abm_loggers.h"
#include "parameter_setter.h"

#include <cstdlib>
#include <fstream>
#include <iostream>

int main(int argc, char* argv[])
{
    // Usage: tutorial_abm_household [n_households] [infected_frac] [sim_days]
    //   n_households  : number of each household type          (default: 125)
    //   infected_frac : fraction initially infected             (default: 0.2)
    //   sim_days      : simulation duration in days             (default: 30)
    int    arg_n_households  = (argc > 1) ? std::atoi(argv[1]) : 125;
    double arg_infected_frac = (argc > 2) ? std::atof(argv[2]) : 0.2;
    int    arg_sim_days      = (argc > 3) ? std::atoi(argv[3]) : 30;

    // Suppress verbose log output; only warnings and errors are shown.
    mio::set_log_level(mio::LogLevel::warn);

    // *** Define age groups. ***
    // The ABM supports multiple age groups. Each person is assigned to exactly
    // one group when created. The number and meaning of groups must be
    // consistent throughout the entire setup.
    size_t num_age_groups         = 6;
    const auto age_group_0_to_4   = mio::AgeGroup(0); // toddlers / kindergarten
    const auto age_group_5_to_14  = mio::AgeGroup(1); // school children
    const auto age_group_15_to_34 = mio::AgeGroup(2); // young adults
    const auto age_group_35_to_59 = mio::AgeGroup(3); // middle-aged adults
    const auto age_group_60_to_79 = mio::AgeGroup(4); // seniors
    const auto age_group_80_plus  = mio::AgeGroup(5); // elderly

    // *** Create the model and set infection parameters. ***
    // The Model holds all persons, locations, and parameters. We pass in the
    // number of age groups so that all parameter arrays are sized correctly.
    // `set_local_parameters` and `set_world_parameters` fill in realistic
    // epidemiological values (see parameter_setter.h).
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


    // *** Define HouseholdMember types. ***
    // A HouseholdMember is not a person itself; it is a *template* that
    // describes the age distribution of whoever fills that role.
    // The probability of each age group is P(i) = weight_i / sum_of_weights.

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

    // senior: equally likely to be 60-79 or 80+ years old.
    auto senior_adult = mio::abm::HouseholdMember(num_age_groups);
    senior_adult.set_age_weight(age_group_60_to_79, 1);
    senior_adult.set_age_weight(age_group_80_plus, 1);

    // *** Compose households and add them to the model. ***
    // A Household collects (member_type, count) pairs. A HouseholdGroup bundles
    // many copies of a Household template. `add_household_group_to_model`
    // creates the actual persons and their home locations.
    //
    // CLI parameters (see usage at top of main):
    //   argv[1] = n_households       (population size: 50, 125, 500)
    //   argv[2] = infected_frac      (initial infected fraction: 0.05, 0.2, 0.5)
    //   argv[3] = sim_days           (simulation duration: 15, 30, 90)
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

    // *** Add locations. ***
    // Besides their home (created automatically above), persons need places to
    // visit: a hospital, an ICU, a social event venue, a shop, a school, and
    // a workplace. The returned LocationId is used to assign persons later.


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
    // Each person draws a random infection state from the distribution below.
    // Persons who are not Susceptible receive a full Infection object so their
    // viral-load course and state transitions are properly initialised.
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

    // Build infection distribution from the infected fraction.
    // The non-susceptible portion is split: 25% Exposed, 50% I_NS, 25% I_Sy.
    const double f = arg_infected_frac;
    std::vector<ScalarType> infection_distribution{
        1.0 - f, f * 0.25, f * 0.50, f * 0.25, 0.0, 0.0, 0.0, 0.0};

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
    // Every person must be assigned to at least a shop, social event venue,
    // hospital, and ICU. School-age children get a school; working-age adults
    // get a workplace. The mobility rules then move persons between their
    // assigned locations according to the time of day.
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

    // *** Run the simulation. ***
    // We simulate 30 days. The Simulation object takes ownership of the model.
    // A History logger records the number of persons in each InfectionState
    // at every time step.
    auto t0   = mio::abm::TimePoint(0);
    auto tmax = t0 + mio::abm::days(arg_sim_days);
    auto sim  = mio::abm::Simulation(t0, std::move(model));

    mio::History<mio::abm::TimeSeriesWriter, mio::abm::LogInfectionState> historyTimeSeries{
        Eigen::Index(mio::abm::InfectionState::Count)};

    sim.advance(tmax, historyTimeSeries);

    // *** Write results to file. ***
    std::ofstream outfile("abm_household.txt");
    std::get<0>(historyTimeSeries.get_log())
        .print_table(outfile,
                     {"S", "E", "I_NS", "I_Sy", "I_Sev", "I_Crit", "R", "D"},
                     7, 4);
    std::cout << "Results written to abm_household.txt\n";

    return 0;
}