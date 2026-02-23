#include "lct_secir/infection_state.h"
#include "memilio/config.h"
#include "memilio/utils/logging.h"

#include <vector>

/**
* @brief Checks that a vector of initial values per subcompartment for one group has the correct shape. 
* @tparam LctState LctInfectionState object that defines the number of subcompartments of the InfectionStates for the 
* considered group. 
* @param initial_populations_per_group Vector containing vectors of initial values per subcompartment for all 
* InfectionStates of the LCT-SECIR model for the considered group. 
* @return Returns 1 if one or more constraints are not satisfied, 0 otherwise.
*/
template <class LctState>
int check_initial_population_per_group(const std::vector<std::vector<ScalarType>>& initial_populations_per_group)
{
    using InfState = mio::lsecir::InfectionState;

    if (initial_populations_per_group.size() != (size_t)InfState::Count) {
        mio::log_error("The number of vectors in initial_populations does not match the number of InfectionStates.");
        return 1;
    }
    if ((initial_populations_per_group[(size_t)InfState::Susceptible].size() !=
         LctState::template get_num_subcompartments<InfState::Susceptible>()) ||
        (initial_populations_per_group[(size_t)InfState::Exposed].size() !=
         LctState::template get_num_subcompartments<InfState::Exposed>()) ||
        (initial_populations_per_group[(size_t)InfState::InfectedNoSymptoms].size() !=
         LctState::template get_num_subcompartments<InfState::InfectedNoSymptoms>()) ||
        (initial_populations_per_group[(size_t)InfState::InfectedSymptoms].size() !=
         LctState::template get_num_subcompartments<InfState::InfectedSymptoms>()) ||
        (initial_populations_per_group[(size_t)InfState::InfectedSevere].size() !=
         LctState::template get_num_subcompartments<InfState::InfectedSevere>()) ||
        (initial_populations_per_group[(size_t)InfState::InfectedCritical].size() !=
         LctState::template get_num_subcompartments<InfState::InfectedCritical>()) ||
        (initial_populations_per_group[(size_t)InfState::Recovered].size() !=
         LctState::template get_num_subcompartments<InfState::Recovered>()) ||
        (initial_populations_per_group[(size_t)InfState::Dead].size() !=
         LctState::template get_num_subcompartments<InfState::Dead>())) {
        mio::log_error("The length of at least one vector in initial_populations does not match the related number of "
                       "subcompartments.");
        return 1;
    }

    return 0;
}