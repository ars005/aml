# Design an Artificial Intelligence application to implement intelligent agents.

environment = {
    "A": "Dirty",
    "B": "Dirty"
}


class VacuumCleanerAgent:

    def __init__(self, location="A"):
        self.location = location
        self.actions = 0
        self.cleaned_rooms = 0

    def perceive(self, environment):
        return environment[self.location]

    def decide_action(self, percept):

        if percept == "Dirty":
            return "Clean"

        if self.location == "A":
            return "Move Right"

        return "Move Left"

    def act(self, action, environment):

        self.actions += 1

        if action == "Clean":
            environment[self.location] = "Clean"
            self.cleaned_rooms += 1

        elif action == "Move Right":
            self.location = "B"

        elif action == "Move Left":
            self.location = "A"


agent = VacuumCleanerAgent("A")

print("Initial Environment:", environment)
print("Initial Location:", agent.location)

for step in range(1, 10):

    percept = agent.perceive(environment)

    action = agent.decide_action(percept)

    agent.act(action, environment)

    print("\nStep:", step)
    print("Location:", agent.location)
    print("Percept:", percept)
    print("Action:", action)
    print("Environment:", environment)

    if environment["A"] == "Clean" and environment["B"] == "Clean":
        print("\nGoal Achieved! All rooms are clean.")
        break


total_rooms = len(environment)
clean_rooms = agent.cleaned_rooms
cleaning_rate = (clean_rooms / total_rooms) * 100

print("\n--- Performance Evaluation ---")
print("Total Rooms:", total_rooms)
print("Clean Rooms:", clean_rooms)
print("Cleaning Rate:", cleaning_rate, "%")
print("Actions Performed:", agent.actions)
print("Final Agent Location:", agent.location)