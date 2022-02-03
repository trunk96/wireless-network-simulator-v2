import os
import platform
from time import sleep, time
import numpy as np

RENDER_REFRESH_TIME = 0.02
NON_RENDER_REFRESH_TIME = 0.008


class LexicographicQTableLearner:

    # based on: Martina Panfili, Antonio Pietrabissa, Guido Oddi & Vincenzo Suraci (2016) 
    # A lexicographic approach to constrained MDP admission control, International Journal of Control,
    # 89:2, 235-247, DOI: 10.1080/00207179.2015.1068955

    def __init__(self, env, model_name, constraints):
        """ Constructor for LexicographicQTableLearner.

        Parameters: 
        env (gym environment): Open AI gym (Toy Text) environment of the game.
        model_name (str): Name of the Open AI gym game

        Returns: None
        """
        self.env = env
        self.model_name = model_name
        action_size = self.env.action_space.n
        state_size = self.env.observation_space.n
        constraint_size = len(constraints)
        self.constraints = constraints
        self.qtable_constraints = np.zeros((constraint_size, state_size, action_size))
        self.qtable = np.zeros((state_size, action_size))
        self.avg_time = 0
        self.steps_per_episode = 1000

    def _clear_screen(self):
        """Clears the terminal screen according to OS.

        Parameters: None

        Returns: None
        """
        os.system('cls') if platform.system() == \
            'Windows' else os.system('clear')

    def _render_train_logs(self, episode, total_episodes, epsilon, step, action, reward, done, done_count):
        """Rendering text logs on console window.

        Parameters:
        episode (int): Running episode number.
        total_episodes (int): The total number of gameplay episodes.
        epsilon (int): Exploration Exploitation tradeoff rate for running episode.
        step (int): Step count for current episode.
        action (int): Action taken for the current state of environment.
        reward (int): Reward for the action taken in the environment.
        done (bool): Flag to know where the episode is finished or not.
        done_count (int): Counter for how many time the agent finished the episode before timeout.

        Returns: None
        """

        # Clear Screen
        self._clear_screen()

        # Printing Logs
        print(f'Model Name     :\t{self.model_name}')
        print(f'Q - Table Shape:\t{self.qtable.shape}')
        print(f'Q - Table Constraints Shape:\t{self.qtable_constraints.shape}')
        print(f'Episode Number :\t{episode}/{total_episodes}')
        print(f'Episode Epsilon:\t{epsilon}')
        print(f'Episode Step   :\t{step+1}/{self.steps_per_episode}')
        print(f'Episode Action :\t{action}')
        print(f'Episode Reward :\t{reward}')
        print(f'Episode Done ? :\t{"Yes" if done else "No"}')
        print(f'Done Count     :\t{done_count}')

    def _render_train_env(self):
        """Renders the environment."""
        print()
        self.env.render()

    def _render_train_time(self, episode_left, episode_t,  step_t, done, step_end, render):
        """ Calculates and renders time metrics for training.

        Parameters:
        episode_left (int): Number of episodes left out of total episodes.
        episode_t (int): Running episode time in seconds.
        step_t (int): Running episode step in seconds.
        done (bool): Flag to know where the episode is finished or not.
        step_end (bool): Flag to know if running step is last step of episode limit.
        render (bool): Flag to render the training environment if possible.
        wait (float): Waiting time to continue process. (default:0.02)

        Returns: None
        """
        if self.avg_time == 0:
            self.avg_time = episode_t
        elif done or step_end:
            self.avg_time = (self.avg_time+episode_t)/2

        time_left = int(self.avg_time*episode_left)
        time_left = (time_left//60, time_left % 60)
        print()
        print(
            f'Time Left            :\t{time_left[0]} mins  {time_left[1]} secs')
        print(f'Average Episode Time :\t{np.round(self.avg_time,4)} secs')
        print(f'Current Episode Time :\t{np.round(episode_t,4)} secs')
        print(f'Current Step Time    :\t{np.round(step_t,4)} secs')
        sleep(RENDER_REFRESH_TIME if render else NON_RENDER_REFRESH_TIME)

    def train(self, train_episodes=10000, lr_init=0.7, gamma=0.9, render=False):
        """ Calling this method will start the training process.

        Parameters: 
        train_episodes (int): The total number of gameplay episodes to learn from for agent. (default:10000)
        lr (float): Learning Rate used by the agent to update the Q-Table after each episode. (default:0.7)
        gamma (float): Discount Rate used by the agent in Bellman's Equation. (default:0.6)
        render (bool): Flag to render the training environment if possible. (default:False)

        Returns: None
        """
        (epsilon, max_epsilon, min_epsilon, decay_rate) = (1.0, 1.0, 0.01, 0.01)
        done_count = 0
        lr = lr_init
        t_episode = 0
        for episode in range(train_episodes):
            t_s_episode = time()

            curr_state = self.env.reset()
            curr_step = 0
            episode_done = False
            lr = lr_init/(episode+1)
            for curr_step in range(self.steps_per_episode):
                t_s_step = time()
                # Exploration Exploitation Tradeoff for the current step.
                ee_tradeoff = np.random.random()
                # Choosing QTable to be used based on which constraint is not satisfied
                c = 0
                constraint_violated = False
                for c in range(len(self.constraints)):
                    #print(curr_state)
                    #print(self.qtable_constraints[c][curr_state, :])
                    if max(self.qtable_constraints[c][curr_state, :]) > self.constraints[c]:
                        constraint_violated = True
                        break
                selected_q_table = None
                if constraint_violated:
                    selected_q_table = self.qtable_constraints[c]
                else:
                    selected_q_table = self.qtable
                # Choosing action based on tradeoff. Random action or action from QTable.               
                curr_action = np.argmin(
                    selected_q_table[curr_state, :]) if ee_tradeoff > epsilon else self.env.action_space.sample()
                # Taking an action, reward will contain the reward of the classic QTable, while info will contain the reward of all the Constraint QTables
                new_state, reward, episode_done, info = self.env.step(
                    curr_action)
                # Keeping track of done count
                done_count += 1 if episode_done else 0
                # Rendering Logs
                self._render_train_logs(episode, train_episodes, epsilon, curr_step, curr_action,
                                        reward, episode_done, done_count)
                # Rendering environment
                if render:
                    self._render_train_env()
                # Updating all the QTables using Bellman Equation
                self.qtable[curr_state, curr_action] = \
                    (1-lr)*self.qtable[curr_state, curr_action] + lr*(reward + gamma * min(self.qtable[new_state, :]))
                for c in range(len(self.constraints)):
                    self.qtable_constraints[c][curr_state, curr_action] = \
                        (1-lr)*self.qtable_constraints[c][curr_state, curr_action] + lr*(info[c] + gamma * min(self.qtable_constraints[c][new_state, :]))
                # Environment state change
                curr_state = new_state

                # Step Time Calculation
                t_step = time() - t_s_step
                self._render_train_time(train_episodes-episode, t_episode,
                                        t_step, episode_done, self.steps_per_episode-1 == curr_step, render)

                if episode_done:
                    break

            # Updating Epsilon for Exploration Exploitation Tradeoff
            epsilon = min_epsilon + \
                (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)

            # Episode Time Calculation
            t_episode = time()-t_s_episode
        self.env.close()

    def _render_test_logs(self, episode, total_episodes, step, episode_reward, step_reward, done, done_count):
        """Rendering text logs on console window.

        Parameters:
        episode (int): Running episode number.
        total_episodes (int): The total number of gameplay episodes.
        step (int): Step count for current episode.
        episode_reward (int): Reward for the action taken in the environment for current episode.
        step_reward (int): Reward for the action taken in the environment for current step.
        done (bool): Flag to know where the episode is finished or not.
        done_count (int): Counter for how many time the agent finished the episode before timeout.

        Returns: None
        """

        # Clear Screen
        self._clear_screen()

        # Printing Logs
        print(f'Model Name     :\t{self.model_name}')
        print(f'Episode Number :\t{episode}/{total_episodes}')
        print(f'Episode Step   :\t{step+1}/{self.steps_per_episode}')
        print(f'Episode Reward :\t{episode_reward}')
        print(f'Step Reward    :\t{step_reward}')
        print(f'Episode Done ? :\t{"Yes" if done else "No"}')
        print(f'Done Count     :\t{done_count}')

    def _render_test_env(self, render):
        """Renders the environment

        Parameters:
        render (bool): Flag to render the environment or not.

        Returns: None
        """

        if render:
            self.env.render()
            sleep(RENDER_REFRESH_TIME)
        else:
            sleep(NON_RENDER_REFRESH_TIME)

    def test(self, test_episodes=200, render=False):
        """ Testing method to know our environment performance.

        Parameters:
        test_episodes (int): Total number of episodes to evaluate performance.
        render (bool): Flag to render the training environment if possible. (default:False)

        Returns: None
        """
        self.env.reset()
        # Collecting the rewards over time.
        rewards = list()
        done_count = 0
        for episode in range(test_episodes):
            state = self.env.reset()
            # Reward for current episode.
            total_rewards = 0
            for _ in range(self.steps_per_episode):
                # Selecting the best action from the appropriate QTable
                c = 0
                constraint_violated = False
                for c in range(len(self.constraints)):
                    if max(self.qtable_constraints[c][state, :]) > self.constraints[c]:
                        constraint_violated = True
                        break
                selected_q_table = None
                if constraint_violated:
                    selected_q_table = self.qtable_constraints[c]
                else:
                    selected_q_table = self.qtable             
                action = np.argmin(selected_q_table[state, :])
                # Performing the action.
                new_state, reward, done, info = self.env.step(action)

                total_rewards += reward
                # Printing logs
                self._render_test_logs(
                    episode, test_episodes, _, total_rewards, reward, done, done_count)
                # Render Environment
                self._render_test_env(render)

                if done:
                    rewards.append(total_rewards)
                    done_count += 1
                    break
                # Changing states for next step
                state = new_state
        self.env.close()
        print(f"\n\nScore over time: \t{sum(rewards)/test_episodes}")

    def set_refresh_time(self, time, render):
        """ Sets the refresh time for render mode TRUE or FALSE.

        Parameters:
        time (float): Refresh time in seconds.
        render (bool): Render flag for which time parameter is to be set.

        Returns: None
        """
        if render:
            RENDER_REFRESH_TIME = time
        else:
            NON_RENDER_REFRESH_TIME = time

    def save_model(self, model_name=None, model_path='saved_models'):
        """ Save the QTable in storage for future use.

        Parameters:
        model_name (str): Takes the model name to save the file.
                          If None is given it will take the default model_name (default: None)
        model_path (str): Folder path of the location where model should be saved.

        Returns: None
        """

        if not model_name:
            model_name = self.model_name

        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        path = os.path.join(model_path, model_name)
        path_constraints = [os.path.join(model_path, model_name+"_constraint_"+str(i)) for i in range(len(self.constraints))]
        np.save(path, self.qtable)
        for i in range(len(path_constraints)):
            np.save(path_constraints[i], self.qtable_constraints[i])
        print(f'Model saved at location :\t{path}')
        sleep(3)

    def load_model(self, model_name):
        """ Load the QTable from storage.

        Parameters:
        model_name (str): The path of the model with extension .npy

        Returns: None
        """
        if not os.path.isfile(model_name):
            print(f'File not found for the location {model_name}')
        else:
            self.qtable = np.load(model_name)
            print(f'Model loaded from location {model_name}')
        path_constraints = [model_name+"_constraint_"+str(i) for i in range(len(self.constraints))]
        for i in range(len(path_constraints)):
            if not os.path.isfile(path_constraints[i]):
                print(f'File not found for the location {path_constraints[i]}')
            else:
                self.qtable_constraints[i] = np.load(path_constraints[i])
                print(f'Model loaded from location {path_constraints[i]}')  
        sleep(3)



