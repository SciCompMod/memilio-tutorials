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
#include "abm/household.h"
#include "abm/lockdown_rules.h"
#include "abm/model.h"
#include "abm/common_abm_loggers.h"

#include <fstream>

int main()
{
    // This is a minimal example with children and adults < 60 year old.
    // We divided them into 4 different age groups, which are defined as follows:
    mio::set_log_level(mio::LogLevel::warn);
    size_t num_age_groups         = 4;
    const auto age_group_0_to_4   = mio::AgeGroup(0);
    const auto age_group_5_to_14  = mio::AgeGroup(1);
    const auto age_group_15_to_34 = mio::AgeGroup(2);
    const auto age_group_35_to_59 = mio::AgeGroup(3);

    // Create the model with 4 age groups.
    auto model = mio::abm::Model(num_age_groups);
    // Set same infection parameter for all age groups. For example, the incubation period is log normally distributed with parameters 4 and 1.
    model.parameters.get<mio::abm::TimeExposedToNoSymptoms>() = mio::ParameterDistributionLogNormal(4., 1.);

    // Set the age group the can go to school is AgeGroup(1) (i.e. 5-14)
    model.parameters.get<mio::abm::AgeGroupGotoSchool>()                    = false;
    model.parameters.get<mio::abm::AgeGroupGotoSchool>()[age_group_5_to_14] = true;
    // Set the age group the can go to work is AgeGroup(2) and AgeGroup(3) (i.e. 15-34 and 35-59)
    model.parameters.get<mio::abm::AgeGroupGotoWork>().set_multiple({age_group_15_to_34, age_group_35_to_59}, true);

    // Check if the parameters satisfy their contraints.
    model.parameters.check_constraints();

    // There are 10 households for each household group.
    int n_households = 10;

    // For more than 1 family households we need families. These are parents and children and randoms (which are distributed like the data we have for these households).
    auto child = mio::abm::HouseholdMember(num_age_groups); // A child is 50/50% 0-4 or 5-14.
    child.set_age_weight(age_group_0_to_4, 1);
    child.set_age_weight(age_group_5_to_14, 1);

    auto parent = mio::abm::HouseholdMember(num_age_groups); // A parent is 50/50% 15-34 or 35-59.
    parent.set_age_weight(age_group_15_to_34, 1);
    parent.set_age_weight(age_group_35_to_59, 1);

    // Two-person household with one parent and one child.
    auto twoPersonHousehold_group = mio::abm::HouseholdGroup();
    auto twoPersonHousehold_full  = mio::abm::Household();
    twoPersonHousehold_full.add_members(child, 1);
    twoPersonHousehold_full.add_members(parent, 1);
    twoPersonHousehold_group.add_households(twoPersonHousehold_full, n_households);
    add_household_group_to_model(model, twoPersonHousehold_group);

    // Three-person household with two parent and one child.
    auto threePersonHousehold_group = mio::abm::HouseholdGroup();
    auto threePersonHousehold_full  = mio::abm::Household();
    threePersonHousehold_full.add_members(child, 1);
    threePersonHousehold_full.add_members(parent, 2);
    threePersonHousehold_group.add_households(threePersonHousehold_full, n_households);
    add_household_group_to_model(model, threePersonHousehold_group);


    // Assign infection state to each person.
    // The infection states are chosen randomly with the following distribution
    std::vector<ScalarType> infection_distribution{0.5, 0.3, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0};
    for (auto& person : model.get_persons()) {
        mio::abm::InfectionState infection_state = mio::abm::InfectionState(
            mio::DiscreteDistribution<size_t>::get_instance()(mio::thread_local_rng(), infection_distribution));
        auto rng = mio::abm::PersonalRandomNumberGenerator(person);
        if (infection_state != mio::abm::InfectionState::Susceptible) {
            person.add_new_infection(mio::abm::Infection(rng, mio::abm::VirusVariant::Wildtype, person.get_age(),
                                                         model.parameters, start_date, infection_state));
        }
    }

    // Assign locations to the people
    for (auto& person : model.get_persons()) {
        const auto id = person.get_id();
        //assign shop and event
        model.assign_location(id, event);
        model.assign_location(id, shop);
        //assign hospital and ICU
        model.assign_location(id, hospital);
        model.assign_location(id, icu);
        //assign work/school to people depending on their age
        if (person.get_age() == age_group_5_to_14) {
            model.assign_location(id, school);
        }
        if (person.get_age() == age_group_15_to_34 || person.get_age() == age_group_35_to_59) {
            model.assign_location(id, work);
        }
    }


    // Set start and end time for the simulation.
    auto t0   = mio::abm::TimePoint(0);
    auto tmax = t0 + mio::abm::days(10);
    auto sim  = mio::abm::Simulation(t0, std::move(model));

    // Create a history object to store the time series of the infection states.
    mio::History<mio::abm::TimeSeriesWriter, mio::abm::LogInfectionState> historyTimeSeries{
        Eigen::Index(mio::abm::InfectionState::Count)};

    // Run the simulation until tmax with the history object.
    sim.advance(tmax, historyTimeSeries);

    // The results are written into the file "abm_minimal.txt" as a table with 9 columns.
    // The first column is Time. The other columns correspond to the number of people with a certain infection state at this Time:
    // Time = Time in days, S = Susceptible, E = Exposed, I_NS = InfectedNoSymptoms, I_Sy = InfectedSymptoms, I_Sev = InfectedSevere,
    // I_Crit = InfectedCritical, R = Recovered, D = Dead
    std::ofstream outfile("abm_minimal.txt");
    std::get<0>(historyTimeSeries.get_log())
        .print_table(outfile, {"S", "E", "I_NS", "I_Sy", "I_Sev", "I_Crit", "R", "D"}, 7, 4);
    std::cout << "Results written to abm_minimal.txt" << std::endl;

    return 0;
}
