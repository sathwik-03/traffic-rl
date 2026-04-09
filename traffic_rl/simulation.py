import numpy as np

class TrafficIntersectionEnv:
    def __init__(self):
        self.num_lanes = 4
        self.queues = np.zeros(self.num_lanes)

        self.current_phase = 0  # 0: NS, 1: EW
        self.max_queue = 30  # increased for better visualization

        self.time_step = 0
        self.max_steps = 1000

        # 🔥 Increased traffic (more realistic)
        self.base_arrival_rates = [0.4, 0.4, 0.4, 0.4]
        self.arrival_rates = self.base_arrival_rates[:]

        # 🔥 Better flow
        self.flow_rate = 0.7

    def reset(self):
        self.queues = np.zeros(self.num_lanes)
        self.current_phase = 0
        self.time_step = 0
        return self._get_state()

    def _get_state(self):
        return np.append(self.queues / self.max_queue, [self.current_phase])

    def step(self, action):
        self.time_step += 1

        # Switch signal phase
        if action != self.current_phase:
            self.current_phase = action

        # 🔥 Dynamic traffic pattern (rush hours)
        if (100 <= self.time_step <= 300) or (500 <= self.time_step <= 700):
            self.arrival_rates = [x * 3 for x in self.base_arrival_rates]
        else:
            self.arrival_rates = self.base_arrival_rates[:]

        # Generate incoming traffic
        for i in range(self.num_lanes):
            if np.random.rand() < self.arrival_rates[i]:
                self.queues[i] += 1

        # Determine active lanes
        active_lanes = [0, 2] if self.current_phase == 0 else [1, 3]

        # Clear traffic
        passed_cars = 0
        for i in active_lanes:
            if self.queues[i] > 0:
                if np.random.rand() < self.flow_rate:
                    self.queues[i] -= 1
                    passed_cars += 1

        # Clip queues
        self.queues = np.clip(self.queues, 0, self.max_queue)

        # 🔥 Better reward (more meaningful)
        reward = -(np.sum(self.queues)) + (passed_cars * 2)

        done = self.time_step >= self.max_steps

        return self._get_state(), reward, done, {}

    def render(self):
        phase_str = "NS Green" if self.current_phase == 0 else "EW Green"
        print(f"Step: {self.time_step} | Phase: {phase_str}")
        print(f"Queues: N:{self.queues[0]:.0f} E:{self.queues[1]:.0f} S:{self.queues[2]:.0f} W:{self.queues[3]:.0f}")
        print("-" * 30)