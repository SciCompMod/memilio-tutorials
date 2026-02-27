import pandas as pd
import matplotlib.pyplot as plt

# Here we provide a simple script for plotting results of SECIR-type models, working for ODE-based, LCT-based as well
# as IDE-based models. When multiple age groups are used, we plot the results aggregated over all age groups.
# We expect a csv with a name that contains the model name, e.g. "*ode*.csv" or "*lct*.csv".


def plot_secir_results(file):

    # In contrast to LCT-based and IDE-based models, the ODE-based SECIR-type model contains compartments for confirmed
    # cases of Carrier and Infected compartments. These will not be plotted.
    if "ode" in file:
        secir_dict = {'Susceptible': 0, 'Exposed': 1,  'Carrier': 2,  'CarrierConfirmed': 3, 'Infected': 3,
                      'InfectedConfirmed': 4, 'Hospitalized': 5, 'ICU': 6, 'Recovered': 7,  'Dead': 8}
    else:
        secir_dict = {'Susceptible': 0, 'Exposed': 1,  'Carrier': 2, 'Infected': 3, 'Hospitalized': 4,
                      'ICU': 5, 'Recovered': 6,  'Dead': 7}

    # Read results.
    results = pd.read_csv(file)
    # Get number of age groups based on shape of results.
    num_age_groups = (len(results.columns)-1)//len(secir_dict)

    # Define plot.
    fig, ax = plt.subplots()

    time = results.iloc[:, 0]

    # Aggregate results over age groups.
    aggregated_susceptibles = 0
    aggregated_exposed = 0
    aggregated_carrier = 0
    aggregated_infected = 0
    aggregated_hospitalized = 0
    aggregated_icu = 0
    aggregated_recovered = 0
    aggregated_dead = 0
    for group in range(num_age_groups):
        aggregated_susceptibles += results.iloc[:,
                                                1 + group*num_age_groups + secir_dict['Susceptible']]
        aggregated_exposed += results.iloc[:,
                                           1 + group*num_age_groups + secir_dict['Exposed']]
        aggregated_carrier += results.iloc[:,
                                           1 + group*num_age_groups + secir_dict['Carrier']]
        aggregated_infected += results.iloc[:,
                                            1 + group*num_age_groups + secir_dict['Infected']]
        aggregated_hospitalized += results.iloc[:,
                                                1 + group*num_age_groups + secir_dict['Hospitalized']]
        aggregated_icu += results.iloc[:,
                                       1 + group*num_age_groups + secir_dict['ICU']]
        aggregated_recovered += results.iloc[:,
                                             1 + group*num_age_groups + secir_dict['Recovered']]
        aggregated_dead += results.iloc[:,
                                        1 + group*num_age_groups + secir_dict['Dead']]

    # Add results to plot.
    ax.plot(
        time, aggregated_susceptibles, label='Susceptible')
    ax.plot(
        time, aggregated_exposed, label='Exposed')
    ax.plot(
        time, aggregated_carrier, label='Carrier')
    ax.plot(
        time, aggregated_infected, label='Infected')
    ax.plot(
        time, aggregated_hospitalized, label='Hospitalized')
    ax.plot(
        time, aggregated_icu, label='ICU')
    ax.plot(
        time, aggregated_recovered, label='Recovered')
    ax.plot(
        time, aggregated_dead, label='Dead')

    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Individuals [#]')
    ax.legend()

    plt.show()


def main():
    # Example for plotting results of LCT simulation created with tutorial_lct.cpp.
    file = "results_lct.csv"
    plot_secir_results(file)


if __name__ == '__main__':
    main()
