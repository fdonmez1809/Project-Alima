import random


def get_input():
    # Generate a random state and percentage
    states = ["neutral", "disgust","sadness","anger"]
    state = random.choice(states)
    percentage = f"{random.randint(50, 80)}%"  # Generate percentage between 50% and 80%
    return state, percentage

if __name__ == "__main__":
    state, percentage = get_input()
    print(f"{state} {percentage}")  # Output as "state percentage"
