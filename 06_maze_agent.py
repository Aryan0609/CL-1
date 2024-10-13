"""
Implement Reinforcement Learning using an example of a maze environment that the 
agent needs to explore.
"""
import numpy as np

def create_maze():
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))
    maze = np.zeros((rows, cols), dtype=int)
    print("Enter the maze layout:")
    for row in range(rows):
        row_data = input().strip()
        maze[row] = [int(cell) for cell in row_data]
    return maze
maze = create_maze()
class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.2):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_prob:
            return np.random.choice(self.num_actions)  
        else:
            return np.argmax(self.q_table[state])  

    def learn(self, state, action, reward, next_state):
        predicted = self.q_table[state, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (target - predicted)
num_states = maze.size
num_actions = 4  

initial_state = 0
goal_state = num_states - 1

agent = QLearningAgent(num_states, num_actions)
def train_agent(agent, num_episodes=1000):
    for episode in range(num_episodes):
        state = initial_state
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state = state
            if action == 0:  # Move Up
                next_state = state - maze.shape[1]
            elif action == 1:  # Move Down
                next_state = state + maze.shape[1]
            elif action == 2:  # Move Left
                next_state = state - 1
            elif action == 3:  # Move Right
                next_state = state + 1

            if (0 <= next_state < num_states) and (maze.flat[next_state] == 0):  # Check if the move is valid
                if next_state == goal_state:
                    reward = 1  # Reached the goal
                    done = True
                else:
                    reward = 0  # Moved to an empty cell
                agent.learn(state, action, reward, next_state)
                state = next_state

train_agent(agent, num_episodes=1000)

def test_agent(agent):
    state = initial_state
    while state != goal_state:
        action = agent.choose_action(state)
        print(f"Current State: {state}, Chosen Action: {action}")
        if action == 0:
            state = state - maze.shape[1]
        elif action == 1:
            state = state + maze.shape[1]
        elif action == 2:
            state = state - 1
        elif action == 3:
            state = state + 1
        print(f"New State: {state}")
    print("Agent reached the goal!")

test_agent(agent)
