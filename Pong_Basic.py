import gym
import gym.spaces
import numpy as np
import pickle

# if pickle.load
# resume = False
# episode_number = 0
running_reward = None
expectation_g_squared = {}
g_dict = {}

def downsample(image):
    """ Take only alternate pixels, halving the resolution of the image"""
    return image[::2, ::2, :]

def remove_color(image):
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector"""
    processed_observation = input_observation[35:195]
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1
    processed_observation = processed_observation.astype(np.float).ravel()

    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)

    prev_processed_observation = processed_observation
    return input_observation, prev_processed_observation

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(vector):
    vector[vector < 0] = 0
    return vector

def apply_neural_nets(observation_matrix, weights):
    """ Returns the hidden layer values and the output layer values.
    Network architecture: FC -> Relu -> FC -> Sigmoid """
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values

def choose_action(probabilty):
    random_value = np.random.uniform()
    if random_value < probabilty:
        return 2 #up in openai gym
    else:
        return 3 #down in openai gym

def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 #this is a game boundary, pong specific
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    #normalize the rewards
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {'1': dC_dw1,'2': dC_dw2}

def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
        weights[layer_name] += (learning_rate * g) / (np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

def main():
    env = gym.make('Pong-v0')
    observation = env.reset()

#Hyperparameters
    batch_size = 10
    gamma = 0.99
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 80 * 80
    learning_rate = 1e-4

#Initialise
    # global resume
    # if resume:
    global running_reward, expectation_g_squared, g_dict
    episode_number = 0
    reward_sum = 0
    prev_processed_observations = None
    weights = pickle.load(open('save.p', 'rb'))
    running_reward = pickle.load(open('save1.p', 'rb'))
    expectation_g_squared = pickle.load(open('save2.p', 'rb'))
    g_dict = pickle.load(open('save3.p', 'rb'))
    print("Loaded")
    # print("Continuing from last run")
    # else:
    # weights = {'1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
    # '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)}

    #Setup for rmsprop
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

    while True:
        env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
        # print(processed_observations.shape)
        hidden_layer_values, up_probabilty  = apply_neural_nets(processed_observations, weights)
        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probabilty)
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        episode_rewards.append(reward)

        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probabilty
        episode_gradient_log_ps.append(loss_function_gradient)

        if done:
            episode_number = episode_number + 1
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps, episode_rewards, gamma)
            gradient = compute_gradient(episode_gradient_log_ps_discounted, episode_hidden_layer_values,
            episode_observations, weights)

            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]

            if episode_number % batch_size == 0:
                update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)
#Reset the parameters
            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []
            observation = env.reset()
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print("Resetting environment. Episode reward total was {}, running mean: {}".format(reward_sum, running_reward))
            reward_sum = 0
            prev_processed_observations = None
            if episode_number % 100 == 0:
                pickle.dump(weights, open('save.p', 'wb'))
                pickle.dump(running_reward, open('save1.p', 'wb'))
                pickle.dump(expectation_g_squared, open('save2.p', 'wb'))
                pickle.dump(g_dict, open('save3.p', 'wb'))
                print(100)
                # resume = True

main()
