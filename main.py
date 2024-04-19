import numpy as np
import matplotlib.pyplot as plt

# Estimated lambda values for male patients from 8 am to 8 pm.
# The lambdas are scaled by dividing by 500 to represent arrival rates.
male_lambda = np.array([3.0, 2.8, 2.5, 2.3, 2.0, 2.3, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8]) / 500
female_lambda = np.array([3.0, 2.8, 2.5, 2.3, 2.0, 2.3, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8]) / 500
# Total lambda values from 8 am to 8 pm, obtained by adding male and female lambda values.
total_lambda = male_lambda + female_lambda

class Hospital:
    """A class representing a hospital with specific attributes and functions related to patient management."""

    def __init__(self):
        """Initialize the hospital with default values for various parameters."""
        self.hours = 12
        self.fixed_doctors = 10
        self.doctor_cost = 500
        self.waiting_cost = 30
        self.untreated_cost = 300
        self.patients_per_doctor = 2
        self.state = 0

    def transition(self, action, arrivals):
        """Simulate the transition of patients based on given action and arrivals.

        Parameters:
        action (int): The number of on-demand doctors to hire for the given hour.
        arrivals (int): The number of patients arriving during the hour.

        The function updates the state of the hospital based on the action and arriv、als.
        """
        treated_patients = min(self.state + arrivals, (self.fixed_doctors + action) * self.patients_per_doctor)
        self.state = max(self.state + arrivals - treated_patients, 0)

    def cost(self, action, arrivals):
        """Calculate the total cost for the given hour and action based on arrivals.

        Parameters:
        action (int): The number of on-demand doctors to hire for the given hour.
        arrivals (int): The number of patients arriving during the hour.

        Returns:
        int: The total cost incurred during the hour.
        """
        cost = action * self.doctor_cost + self.state * self.waiting_cost
        return cost

    def final_cost(self):
        """Calculate the final cost for the hospital at the end of the day.

        The final cost considers the untreated patients and fixed doctor costs over the full day.

        Returns:
        int: The final cost for the hospital.
        """
        untreated_patients = max(self.state - self.fixed_doctors * self.patients_per_doctor, 0)
        cost = untreated_patients * self.untreated_cost + self.fixed_doctors * self.doctor_cost * self.hours
        return cost


class Agent:
    """A class representing an agent that learns to make decisions for the hospital."""

    def __init__(self, hospital):
        """Initialize the agent with the hospital and setup initial values for learning.

        Parameters:
        hospital (Hospital): The hospital object that the agent will make decisions for.
        """
        self.hospital = hospital
        self.costs = np.full((hospital.hours + 1, 100), np.inf)
        self.policies = np.zeros((hospital.hours + 1, 100), dtype=int)
        self.costs[hospital.hours, :] = [hospital.final_cost() for hospital.state in range(100)]
        self.policies[hospital.hours, :] = 0

    def bellman(self, t, xt, ut, dt_costs):
        """Perform Bellman update for the given time, state, action, and future costs.

        Parameters:
        t (int): The current time (hour).
        xt (int): The current state (queue length).
        ut (int): The current action (number of on-demand doctors to hire).
        dt_costs (list): List of costs associated with different patient treatments.

        The function updates the cost-to-go and optimal action for the given state and time.
        """
        average_cost = np.mean(dt_costs)
        if average_cost < self.costs[t, xt]:
            self.costs[t, xt] = average_cost
            self.policies[t, xt] = ut

    def learn(self):
        """Learn optimal policies using dynamic programming and Bellman equations."""
        for t in range(self.hospital.hours - 1, -1, -1):
            for xt in range(100):
                for ut in range(xt // self.hospital.patients_per_doctor + 1):
                    dt_costs = []
                    for dt in range(21):
                        self.hospital.state = xt
                        self.hospital.transition(ut, dt)
                        cost = self.hospital.cost(ut, dt) + self.costs[t + 1, self.hospital.state]
                        dt_costs.append(cost)
                    self.bellman(t, xt, ut, dt_costs)

    def decide(self, t):
        """Get the optimal action (number of on-demand doctors to hire) for the given time.

        Parameters:
        t (int): The current time (hour).

        Returns:
        int: The optimal action to take at the given time.
        """
        return self.policies[t, self.hospital.state]

if __name__ == "__main__":
    hospital = Hospital()
    agent = Agent(hospital)
    agent.learn()

    print("Policy: \n", agent.policies)

    total_cost = 0
    for day in range(100):
        for hour in range(hospital.hours):
            arrivals = np.random.poisson(total_lambda[hour])
            action = agent.decide(hour)
            cost = hospital.cost(action, arrivals)
            hospital.transition(action, arrivals)
            total_cost += cost
        total_cost += hospital.final_cost()
    print("Total cost over 100 days: ", total_cost)

    # Extend total_lambda to cover the full 24 hours.
    full_day_lambda = np.concatenate((total_lambda, total_lambda[::-1])) * 500

    # Plotting the average patient arrivals over the full day.
    plt.figure(figsize=(12, 6))
    colors = ['grey'] * 24
    colors[8:20] = ['blue'] * 12
    plt.bar(np.arange(24), full_day_lambda, color=colors)
    plt.title('Average Patient Arrivals Over 24 Hours')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Patient Arrivals')
    plt.xticks(np.arange(24))
    plt.grid(True)
    plt.show()

    # Plotting the number of on-demand doctors to hire for each hour and each queue length.
    plt.figure(figsize=(10, 6))
    plt.imshow(agent.policies[1:], aspect='auto', cmap='YlOrBr', origin='lower')
    plt.title('Number of On-Demand Doctors to Hire')
    plt.xlabel('Queue Length (Number of Patients)')
    plt.ylabel('Hour of the Day')
    plt.colorbar(label='Number of Doctors')
    plt.grid(False)
    plt.show()
