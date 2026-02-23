#!/usr/bin/env python3
"""
Visualize ABM household simulation results.
Plots the time evolution of different infection states.
"""

import matplotlib.pyplot as plt
import pandas as pd

# Read the data
data = pd.read_csv('build/bin/abm_household.txt', sep=r'\s+')

# Create the plot
fig, ax = plt.subplots(figsize=(12, 7))

# Plot each infection state with different colors
ax.plot(data['Time'], data['S'], label='Susceptible', color='blue', linewidth=2)
ax.plot(data['Time'], data['E'], label='Exposed', color='orange', linewidth=2)
ax.plot(data['Time'], data['I_NS'], label='Infected (No Symptoms)', color='yellow', linewidth=2)
ax.plot(data['Time'], data['I_Sy'], label='Infected (Symptomatic)', color='red', linewidth=2)
ax.plot(data['Time'], data['I_Sev'], label='Infected (Severe)', color='darkred', linewidth=2)
ax.plot(data['Time'], data['I_Crit'], label='Infected (Critical)', color='purple', linewidth=2)
ax.plot(data['Time'], data['R'], label='Recovered', color='green', linewidth=2)
ax.plot(data['Time'], data['D'], label='Dead', color='black', linewidth=2)

# calculate number of agents in each state at the last time point
number = data.iloc[-1][['S', 'E', 'I_NS', 'I_Sy', 'I_Sev', 'I_Crit', 'R', 'D']]
print("Final counts of agents in each state at the last time point:")
print(number)

# Customize the plot
ax.set_xlabel('Time (days)', fontsize=12)
ax.set_ylabel('Number of Persons', fontsize=12)
ax.set_title('ABM Household Simulation: Infection States Over Time (Number of Agents: {})'.format(number.sum()), fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

# Set y-axis to start at 0
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('abm_household_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'abm_household_plot.png'")

plt.show()
