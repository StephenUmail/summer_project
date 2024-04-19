import numpy as np
import matplotlib.pyplot as plt


class Hospital:
    """Defines the hospital environment."""

    def __init__(self):
        """Initializes the hospital environment with default parameters."""
        self.hours = 12  # The hospital operates for 12 hours a day
        self.fixed_doctors = 10  # There are always 10 doctors in the hospital
        self.doctor_cost = 1  # The cost of hiring one doctor for one hour
        self.waiting_cost = 10  # The cost of one patient waiting for one hour
        self.untreated_cost = 300  # The cost of one patient not being treated by the end of the day
        self.patients_per_doctor = 2  # Each doctor can treat two patients per hour
        self.state = 0  # The initial number of patients in the hospital is 0

    def transition(self, action, arrivals):
        """
        Defines the transition function, which changes the state based on the action and the number of arrivals.
        Parameters:
        action (int): number of doctors to hire.
        arrivals (int): number of patient arrivals.
        """
        # The number of patients who get treated is the minimum of the total number of patients and the total treatment capacity
        treated_patients = min(self.state + arrivals, (self.fixed_doctors + action) * self.patients_per_doctor)
        # Update the state, which is the maximum of zero and the total number of patients minus the treated patients
        self.state = max(self.state + arrivals - treated_patients, 0)

    def cost(self, action, arrivals):
        """
        Defines the cost function, which calculates the total cost based on the action and the number of arrivals.
        Parameters:
        action (int): number of doctors to hire.
        arrivals (int): number of patient arrivals.
        Returns:
        cost (float): total cost.
        """
        # The cost is the sum of the cost of hiring doctors and the cost of patients waiting
        cost = action * self.doctor_cost + self.state * self.waiting_cost
        return cost

    def final_cost(self):
        """
        Defines the final cost function, which calculates the total cost at the end of the day.
        Returns:
        cost (float): total cost.
        """
        # The number of untreated patients is the maximum of zero and the total number of patients minus the treatment capacity
        untreated_patients = max(self.state - self.fixed_doctors * self.patients_per_doctor, 0)
        # The final cost is the sum of the cost of untreated patients and the cost of fixed doctors
        cost = untreated_patients * self.untreated_cost + self.fixed_doctors * self.doctor_cost * self.hours
        return cost


class QAgent:
    """Defines the Q-learning agent."""

    def __init__(self, hospital):
        """
        Initializes the agent with a reference to the hospital environment and default learning parameters.
        Parameters:
        hospital (Hospital): a reference to the hospital environment.
        """
        self.hospital = hospital  # A reference to the hospital environment
        self.alpha = 0.5  # The learning rate for Q-learning
        self.epsilon = 0.1  # The exploration rate for epsilon-greedy policy
        self.Q = np.zeros((hospital.hours + 1, 500, 300))  # The Q-table, initialized to zeros
        self.policy = np.zeros((hospital.hours + 1, 500), dtype=int)  # The policy, initialized to zeros

    def decide(self, t):
        """
        Defines the decision-making function, which chooses an action based on the policy.
        Parameters:
        t (int): current time step.
        Returns:
        action (int): number of doctors to hire.
        """
        # If a random number is less than epsilon, choose a random action (exploration)
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.hospital.state // self.hospital.patients_per_doctor + 1)
        else:
            # Otherwise, choose the action with the highest Q-value (exploitation)
            action = self.policy[t, self.hospital.state]
        return action

    def learn(self):
        """
        Defines the learning function, which updates the Q-table and the policy.
        """
        # The agent learns over multiple days
        for day in range(1000):
            # For each hour of the day
            for hour in range(self.hospital.hours):
                # The agent decides how many doctors to hire based on its policy
                action = self.decide(hour)
                # The current state is recorded
                old_state = self.hospital.state
                # The number of arrivals is a random number from a Poisson distribution
                arrivals = np.random.poisson(total_lambda[hour])
                # The cost is calculated based on the action and the arrivals
                cost = self.hospital.cost(action, arrivals)
                # The state is updated based on the action and the arrivals
                self.hospital.transition(action, arrivals)
                # The new state is recorded
                new_state = self.hospital.state
                # If it's not the last hour of the day
                if hour < self.hospital.hours - 1:
                    # The future cost is the minimum Q-value of the new state
                    future_cost = np.min(self.Q[hour + 1, new_state, :])
                else:
                    # Otherwise, the future cost is the final cost of the day
                    future_cost = self.hospital.final_cost()
                # The Q-value of the current state and action is updated based on the learning rate, the cost, and the future cost
                self.Q[hour, old_state, action] = (1 - self.alpha) * self.Q[hour, old_state, action] + self.alpha * (
                            cost + future_cost)
                # The policy of the current state is updated to the action with the minimum Q-value
                self.policy[hour, old_state] = np.argmin(self.Q[hour, old_state, :])


# Define the lambda values for the Poisson distribution for each hour
lambda_values = np.array(
    [7.2, 6.6, 6, 5.6, 5.2, 5, 5.2, 5.6, 6, 6.6, 7.2, 7.8, 8.4, 9, 9.6, 10.2, 9.6, 9, 8.4, 7.8, 7.2, 6.6, 6, 5.6])
# We only need the data from 8am to 8pm, so we double the lambda values for these hours
total_lambda = 2 * lambda_values[8:20]

if __name__ == "__main__":
    """
    The main function where the simulation is run.
    """
    # Create a hospital and a Q-learning agent
    hospital = Hospital()
    agent = QAgent(hospital)
    # The agent learns the policy
    agent.learn()

    print("Policy: \n", agent.policy)
    print("Q-table: \n", agent.Q)

    total_cost = 0
    # The agent applies its policy over multiple days
    for day in range(100):
        # For each hour of the day
        for hour in range(hospital.hours):
            # The number of arrivals is a random number from a Poisson distribution
            arrivals = np.random.poisson(total_lambda[hour])
            # The agent decides how many doctors to hire based on its policy
            action = agent.decide(hour)
            # The cost is calculated based on the action and the arrivals
            cost = hospital.cost(action, arrivals)
            # The state is updated based on the action and the arrivals
            hospital.transition(action, arrivals)
            # The cost is added to the total cost
            total_cost += cost
        # The final cost of the day is added to the total cost
        total_cost += hospital.final_cost()
    print("Total cost over 100 days: ", total_cost)

    # The lambda values are scaled up for the purpose of visualization
    full_day_lambda = np.concatenate((total_lambda, total_lambda[::-1])) * 500

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

    plt.figure(figsize=(10, 6))
    plt.imshow(agent.policy[1:13], aspect='auto', cmap='YlOrBr', origin='lower')
    plt.title('Number of On-Demand Doctors to Hire')
    plt.xlabel('Queue Length (Number of Patients)')
    plt.ylabel('Hour of the Day')
    plt.colorbar(label='Number of Doctors')
    plt.grid(False)
    plt.show()

    plt.figure(figsize=(10, 6))
    max_q_values = np.max(agent.Q[1:13, :, :], axis=-1)  # Get the maximum Q-value for each hour and state
    plt.imshow(max_q_values, aspect='auto', cmap='YlOrBr', origin='lower')
    plt.title('Maximum Q-values for each hour and state')
    plt.xlabel('Queue Length (Number of Patients)')
    plt.ylabel('Hour of the Day')
    plt.colorbar(label='Maximum Q-value')
    plt.grid(False)
    plt.show()

    plt.figure(figsize=(10, 6))
    max_q_values = np.max(agent.Q[1:13, :200, :], axis=-1)  # Get the maximum Q-value for each hour and state
    plt.imshow(max_q_values, aspect='auto', cmap='YlOrBr', origin='lower')
    plt.title('Maximum Q-values for each hour and state')
    plt.xlabel('Queue Length (Number of Patients)')
    plt.ylabel('Hour of the Day')
    plt.colorbar(label='Maximum Q-value')
    plt.grid(False)
    plt.show()
