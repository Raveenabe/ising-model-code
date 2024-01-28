import numpy as np
import matplotlib.pyplot as plt
import time

def initialize_lattice(size):
    probability=0.75
    lattice = np.random.choice([-1, 1], size=(size, size), p=[1 - probability, probability])
    return lattice

def calculate_energy(lattice):
    energy = 0
    size = lattice.shape[0]
    for i in range(size):
        for j in range(size):
            spin = lattice[i, j]
            neighbors = lattice[(i+1)%size, j] + lattice[i, (j+1)%size] + lattice[(i-1)%size, j] + lattice[i, (j-1)%size]
            energy += -spin * neighbors
    return energy

def calculate_magnetization(lattice):
    return np.sum(lattice)

def metropolis_step(lattice, temperature):
    size = lattice.shape[0]
    i = np.random.randint(size)
    j = np.random.randint(size)
    spin = lattice[i, j]
    neighbors = lattice[(i+1)%size, j] + lattice[i, (j+1)%size] + lattice[(i-1)%size, j] + lattice[i, (j-1)%size]
    energy_diff = 2 * spin * neighbors
    if energy_diff < 0 or np.random.rand() < np.exp(-energy_diff / temperature):
        lattice[i, j] = -spin
    return lattice

def run_simulation(size, temperature, num_steps, equilibration_steps):
    lattice = initialize_lattice(size)
    energy_vals = []
    magnetization_vals = []
    for step in range(num_steps):
        for _ in range(equilibration_steps):
            lattice = metropolis_step(lattice, temperature)
            print('looping')
        energy = calculate_energy(lattice)
        magnetization = calculate_magnetization(lattice)
        energy_vals.append(energy)
        magnetization_vals.append(magnetization)
    return energy_vals, magnetization_vals

def calculate_mean(values):
    return np.mean(values)

def calculate_specific_heat(energy_vals, temperature):
    energy_squared = np.mean(np.array(energy_vals)**2)
    energy_mean = calculate_mean(energy_vals)
    return (energy_squared - energy_mean**2) / temperature**2

def calculate_magnetic_susceptibility(magnetization_vals, temperature):
    magnetization_squared = np.mean(np.array(magnetization_vals)**2)
    magnetization_mean = calculate_mean(magnetization_vals)
    return (magnetization_squared - magnetization_mean**2) / temperature

def calculate_binder_cumulant(magnetization_vals):
    magnetization_fourth = np.mean(np.array(magnetization_vals)**4)
    magnetization_squared = np.mean(np.array(magnetization_vals)**2)
    return 1 - magnetization_fourth / (3 * magnetization_squared**2)


# Parameters
size = 50                   # Size of the lattice
temperature_range = np.linspace(0.1, 4.0, num=400)  # Temperature range
num_steps = 10000           # Number of Monte Carlo steps
equilibration_steps = 1000  # Equilibration steps

# Store thermodynamic values and errors for each temperature
mean_energy_values = []
mean_magnetization_values = []
specific_heat_values = []
magnetic_susceptibility_values = []
binder_cumulant_values = []
mean_energy_errors = []
mean_magnetization_errors = []
specific_heat_errors = []
magnetic_susceptibility_errors = []
binder_cumulant_errors = []

# Measure runtime
start_time = time.time()

# Run the simulation for each temperature
for temperature in temperature_range:
    energy_values, magnetization_values = run_simulation(size, temperature, num_steps, equilibration_steps)
    
    # Calculate thermodynamic values
    mean_energy = calculate_mean(energy_values)
    mean_magnetization = calculate_mean(magnetization_values)
    specific_heat = calculate_specific_heat(energy_values, temperature)
    magnetic_susceptibility = calculate_magnetic_susceptibility(magnetization_values, temperature)
    binder_cumulant = calculate_binder_cumulant(magnetization_values)
    
    # Append values to lists
    mean_energy_values.append(mean_energy)
    mean_magnetization_values.append(mean_magnetization)
    specific_heat_values.append(specific_heat)
    magnetic_susceptibility_values.append(magnetic_susceptibility)
    binder_cumulant_values.append(binder_cumulant)
    
    # Calculate standard deviation for each thermodynamic value
    mean_energy_std = np.std(energy_values)
    mean_magnetization_std = np.std(magnetization_values)
    specific_heat_std = np.std(energy_values) / temperature**2
    magnetic_susceptibility_std = np.std(magnetization_values) / temperature
    binder_cumulant_std = np.std(magnetization_values)**2 / np.mean(magnetization_values)**4
    
    # Append errors to lists
    mean_energy_errors.append(mean_energy_std)
    mean_magnetization_errors.append(mean_magnetization_std)
    specific_heat_errors.append(specific_heat_std)
    magnetic_susceptibility_errors.append(magnetic_susceptibility_std)
    binder_cumulant_errors.append(binder_cumulant_std)

# Calculate runtime
end_time = time.time()
runtime = end_time - start_time

# Plot Mean Energy with error bars
plt.errorbar(temperature_range, mean_energy_values, yerr=mean_energy_errors, fmt='o')
plt.xlabel('Temperature', fontsize=18)
plt.ylabel('Mean Energy', fontsize=18)
plt.title('Mean Energy vs Temperature', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Plot Mean Magnetization with error bars
plt.errorbar(temperature_range, mean_magnetization_values, yerr=mean_magnetization_errors, fmt='o')
plt.xlabel('Temperature', fontsize=18)
plt.ylabel('Mean Magnetization', fontsize=18)
plt.title('Mean Magnetization vs Temperature', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Plot Specific Heat Capacity with error bars
plt.errorbar(temperature_range, specific_heat_values, yerr=specific_heat_errors, fmt='o')
plt.xlabel('Temperature', fontsize=18)
plt.ylabel('Specific Heat Capacity', fontsize=18)
plt.title('Specific Heat Capacity vs Temperature', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Plot Magnetic Susceptibility with error bars
plt.errorbar(temperature_range, magnetic_susceptibility_values, yerr=magnetic_susceptibility_errors, fmt='o')
plt.xlabel('Temperature', fontsize=18)
plt.ylabel('Magnetic Susceptibility', fontsize=18)
plt.title('Magnetic Susceptibility vs Temperature', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Plot Binder Cumulant with error bars
plt.errorbar(temperature_range, binder_cumulant_values, yerr=binder_cumulant_errors, fmt='o')
plt.xlabel('Temperature', fontsize=18)
plt.ylabel('Binder Cumulant', fontsize=18)
plt.title('Binder Cumulant vs Temperature', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

def calculate_thermodynamic_values(size, temperature_range, num_steps, equilibration_steps):
    mean_energy_values = []
    mean_magnetization_values = []
    specific_heat_values = []
    magnetic_susceptibility_values = []
    binder_cumulant_values = []

    for temperature in temperature_range:
        energy_values, magnetization_values = run_simulation(size, temperature, num_steps, equilibration_steps)

        mean_energy = calculate_mean(energy_values)
        mean_magnetization = calculate_mean(magnetization_values)
        specific_heat = calculate_specific_heat(energy_values, temperature)
        magnetic_susceptibility = calculate_magnetic_susceptibility(magnetization_values, temperature)
        binder_cumulant = calculate_binder_cumulant(magnetization_values)

        mean_energy_values.append(mean_energy)
        mean_magnetization_values.append(mean_magnetization)
        specific_heat_values.append(specific_heat)
        magnetic_susceptibility_values.append(magnetic_susceptibility)
        binder_cumulant_values.append(binder_cumulant)

    return mean_energy_values, mean_magnetization_values, specific_heat_values, magnetic_susceptibility_values, binder_cumulant_values

# Parameters
sizes = [20, 30, 40, 50]    # Sizes of the lattices
temperature_range = np.linspace(1.0, 3.0, num=100)  # Temperature range
num_steps = 10000           # Number of Monte Carlo steps
equilibration_steps = 1000  # Equilibration steps

# Store thermodynamic value sets for each lattice size
mean_energy_sets = []
mean_magnetization_sets = []
specific_heat_sets = []
magnetic_susceptibility_sets = []
binder_cumulant_sets = []

for size in sizes:
    thermodynamic_values = calculate_thermodynamic_values(size, temperature_range, num_steps, equilibration_steps)
    mean_energy_sets.append(thermodynamic_values[0])
    mean_magnetization_sets.append(thermodynamic_values[1])
    specific_heat_sets.append(thermodynamic_values[2])
    magnetic_susceptibility_sets.append(thermodynamic_values[3])
    binder_cumulant_sets.append(thermodynamic_values[4])

# Plot Mean Energy for different lattice sizes
for i in range(len(sizes)):
    plt.plot(temperature_range, mean_energy_sets[i], label='Lattice Size {}'.format(sizes[i]))

plt.xlabel('Temperature', fontsize=18)
plt.ylabel('Mean Energy', fontsize=18)
plt.title('Mean Energy vs Temperature', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.show()

# Plot Mean Magnetization for different lattice sizes
for i in range(len(sizes)):
    plt.plot(temperature_range, mean_magnetization_sets[i], label='Lattice Size {}'.format(sizes[i]))

plt.xlabel('Temperature', fontsize=18)
plt.ylabel('Mean Magnetization', fontsize=18)
plt.title('Mean Magnetization vs Temperature', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.show()

# Plot Specific Heat Capacity for different lattice sizes
for i in range(len(sizes)):
    plt.plot(temperature_range, specific_heat_sets[i], label='Lattice Size {}'.format(sizes[i]))

plt.xlabel('Temperature', fontsize=18)
plt.ylabel('Specific Heat Capacity', fontsize=18)
plt.title('Specific Heat Capacity vs Temperature', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.show()

# Plot Magnetic Susceptibility for different lattice sizes
for i in range(len(sizes)):
    plt.plot(temperature_range, magnetic_susceptibility_sets[i], label='Lattice Size {}'.format(sizes[i]))

plt.xlabel('Temperature', fontsize=18)
plt.ylabel('Magnetic Susceptibility', fontsize=18)
plt.title('Magnetic Susceptibility vs Temperature', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.show()

# Plot Binder Cumulant for different lattice sizes
for i in range(len(sizes)):
    plt.plot(temperature_range, binder_cumulant_sets[i], label='Lattice Size {}'.format(sizes[i]))

plt.xlabel('Temperature', fontsize=18)
plt.ylabel('Binder Cumulant', fontsize=18)
plt.title('Binder Cumulant vs Temperature', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.show()
