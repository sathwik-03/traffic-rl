class FixedTimeAgent:
    def __init__(self, switch_interval=30):
        self.switch_interval = switch_interval
        self.action = 0
        self.step_counter = 0

    def act(self, state):
        self.step_counter += 1
        if self.step_counter >= self.switch_interval:
            self.action = 1 - self.action # Toggle between 0 and 1
            self.step_counter = 0
        return self.action
