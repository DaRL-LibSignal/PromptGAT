from . import RLAgent
from common.registry import Registry
from agent import utils
import numpy as np
import os
import random
from collections import deque
import gym

from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

mode = 'dqn'
# mode = 'ac_dqn' # action_transformed

if mode == 'dqn':
    @Registry.register_model('dqn')
    class DQNAgent(RLAgent):
        '''
        DQNAgent determines each intersection's action with its own intersection information.
        '''

        def __init__(self, world, rank):
            super().__init__(world, world.intersection_ids[rank])
            self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
            self.replay_buffer = deque(maxlen=self.buffer_size)

            self.world = world
            self.sub_agents = 1
            self.rank = rank

            self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
            self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']

            # get generator for each DQNAgent
            inter_id = self.world.intersection_ids[self.rank]
            inter_obj = self.world.id2intersection[inter_id]
            self.inter = inter_obj
            self.ob_generator = LaneVehicleGenerator(self.world, self.inter, ['lane_count'], in_only=True, average=None)

            self.phase_generator = IntersectionPhaseGenerator(world, self.inter, ["phase"],
                                                              targets=["cur_phase"], negative=False)
            self.reward_generator = LaneVehicleGenerator(self.world, self.inter, ["lane_waiting_count"],
                                                         in_only=True, average='all', negative=True)
            self.action_space = gym.spaces.Discrete(len(self.world.id2intersection[inter_id].phases))

            if self.phase:
                if self.one_hot:
                    self.ob_length = self.ob_generator.ob_length + len(self.world.id2intersection[inter_id].phases)
                else:
                    self.ob_length = self.ob_generator.ob_length + 1
            else:
                self.ob_length = self.ob_generator.ob_length

            self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
            self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
            self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
            self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
            self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
            self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
            self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
            self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']

            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_network()
            self.criterion = nn.MSELoss(reduction='mean')
            self.optimizer = optim.RMSprop(self.model.parameters(),
                                           lr=self.learning_rate,
                                           alpha=0.9, centered=False, eps=1e-7)
            print(self.ob_length, self.action_space.n)

        def __repr__(self):
            return self.model.__repr__()

        def reset(self):
            '''
            reset
            Reset information, including ob_generator, phase_generator, queue, delay, etc.

            :param: None
            :return: None
            '''
            inter_id = self.world.intersection_ids[self.rank]
            inter_obj = self.world.id2intersection[inter_id]
            self.ob_generator = LaneVehicleGenerator(self.world, inter_obj, ['lane_count'], in_only=True, average=None)
            self.phase_generator = IntersectionPhaseGenerator(self.world, inter_obj, ["phase"],
                                                              targets=["cur_phase"], negative=False)
            self.reward_generator = LaneVehicleGenerator(self.world, inter_obj, ["lane_waiting_count"],
                                                         in_only=True, average='all', negative=True)
            self.queue = LaneVehicleGenerator(self.world, inter_obj,
                                              ["lane_waiting_count"], in_only=True,
                                              negative=False)
            self.delay = LaneVehicleGenerator(self.world, inter_obj,
                                              ["lane_delay"], in_only=True, average="all",
                                              negative=False)

        def get_ob(self):
            '''
            get_ob
            Get observation from environment.

            :param: None
            :return x_obs: observation generated by ob_generator
            '''
            x_obs = []
            x_obs.append(self.ob_generator.generate())
            x_obs = np.array(x_obs, dtype=np.float32)
            return x_obs

        def get_reward(self):
            '''
            get_reward
            Get reward from environment.

            :param: None
            :return rewards: rewards generated by reward_generator
            '''
            rewards = []
            rewards.append(self.reward_generator.generate())
            rewards = np.squeeze(np.array(rewards)) * 12
            return rewards

        def get_phase(self):
            '''
            get_phase
            Get current phase of intersection(s) from environment.

            :param: None
            :return phase: current phase generated by phase_generator
            '''
            phase = []
            phase.append(self.phase_generator.generate())
            # phase = np.concatenate(phase, dtype=np.int8)
            phase = (np.concatenate(phase)).astype(np.int8)
            return phase

        def get_action(self, ob, phase, test=False):
            '''
            get_action
            Generate action.

            :param ob: observation
            :param phase: current phase
            :param test: boolean, decide whether is test process
            :return action: action that has the highest score
            '''
            if not test:
                if np.random.rand() <= self.epsilon:
                    return self.sample()
            if self.phase:
                if self.one_hot:
                    feature = np.concatenate([ob, utils.idx2onehot(phase, self.action_space.n)], axis=1)
                else:
                    feature = np.concatenate([ob, phase], axis=1)
            else:
                feature = ob
            observation = torch.tensor(feature, dtype=torch.float32)
            # TODO: no need to calculate gradient when interacting with environment
            actions = self.model(observation, train=False)
            actions = actions.clone().detach().numpy()
            return np.argmax(actions, axis=1)

        def sample(self):
            '''
            sample
            Sample action randomly.

            :param: None
            :return: action generated randomly.
            '''
            return np.random.randint(0, self.action_space.n, self.sub_agents)

        def _build_model(self):
            '''
            _build_model
            Build a DQN model.

            :param: None
            :return model: DQN model
            '''
            model = DQNNet(self.ob_length, self.action_space.n)
            return model

        def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
            '''
            remember
            Put current step information into replay buffer for training agent later.

            :param last_obs: last step observation
            :param last_phase: last step phase
            :param actions: actions executed by intersections
            :param actions_prob: the probability that the intersections execute the actions
            :param rewards: current step rewards
            :param obs: current step observation
            :param cur_phase: current step phase
            :param done: boolean, decide whether the process is done
            :param key: key to store this record, e.g., episode_step_agentid
            :return: None
            '''
            self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))

        def _batchwise(self, samples):
            '''
            _batchwise
            Reconstruct the samples into batch form(last state, current state, reward, action).

            :param samples: original samples record in replay buffer
            :return state_t, state_tp, rewards, actions: information with batch form
            '''
            obs_t = np.concatenate([item[1][0] for item in samples])
            obs_tp = np.concatenate([item[1][4] for item in samples])
            if self.phase:
                if self.one_hot:
                    phase_t = np.concatenate([utils.idx2onehot(item[1][1], self.action_space.n) for item in samples])
                    phase_tp = np.concatenate([utils.idx2onehot(item[1][5], self.action_space.n) for item in samples])
                else:
                    phase_t = np.concatenate([item[1][1] for item in samples])
                    phase_tp = np.concatenate([item[1][5] for item in samples])
                feature_t = np.concatenate([obs_t, phase_t], axis=1)
                feature_tp = np.concatenate([obs_tp, phase_tp], axis=1)
            else:
                feature_t = obs_t
                feature_tp = obs_tp
            state_t = torch.tensor(feature_t, dtype=torch.float32)
            state_tp = torch.tensor(feature_tp, dtype=torch.float32)
            rewards = torch.tensor(np.array([item[1][3] for item in samples]), dtype=torch.float32)  # TODO: BETTER WA
            actions = torch.tensor(np.array([item[1][2] for item in samples]), dtype=torch.long)
            return state_t, state_tp, rewards, actions

        def train(self):
            '''
            train
            Train the agent, optimize the action generated by agent.

            :param: None
            :return: value of loss
            '''
            samples = random.sample(self.replay_buffer, self.batch_size)
            b_t, b_tp, rewards, actions = self._batchwise(samples)
            out = self.target_model(b_tp, train=False)
            target = rewards + self.gamma * torch.max(out, dim=1)[0]
            target_f = self.model(b_t, train=False)
            for i, action in enumerate(actions):
                target_f[i][action] = target[i]
            loss = self.criterion(self.model(b_t, train=True), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return loss.clone().detach().numpy()

        def update_target_network(self):
            '''
            update_target_network
            Update params of target network.

            :param: None
            :return: None
            '''
            weights = self.model.state_dict()
            self.target_model.load_state_dict(weights)

        def load_model(self, e, customized_path=""):
            '''
            load_model
            Load model params of an episode.

            :param e: specified episode
            :return: None
            '''
            if customized_path != "" and (e == "" or None):
                model_name = customized_path

            else:
                model_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                          'model', f'{e}_{self.rank}.pt')
            self.model = self._build_model()
            self.model.load_state_dict(torch.load(model_name))
            self.target_model = self._build_model()
            self.target_model.load_state_dict(torch.load(model_name))
            self.optimizer = optim.RMSprop(self.model.parameters(),
                                           lr=self.learning_rate,
                                           alpha=0.9, centered=False, eps=1e-7)
            print(f'model loaded at {model_name}')

        def save_model(self, e):
            '''
            save_model
            Save model params of an episode.

            :param e: specified episode, used for file name
            :return: None
            '''
            path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
            if not os.path.exists(path):
                os.makedirs(path)
            model_name = os.path.join(path, f'{e}_{self.rank}.pt')
            torch.save(self.model.state_dict(), model_name)
            print(f'model saved at {model_name}')
            return model_name


    class DQNNet(nn.Module):
        '''
        DQNNet consists of 3 dense layers.
        '''

        def __init__(self, input_dim, output_dim):
            super(DQNNet, self).__init__()
            self.dense_1 = nn.Linear(input_dim, 20)
            self.dense_2 = nn.Linear(20, 20)
            self.dense_3 = nn.Linear(20, output_dim)

        def _forward(self, x):
            x = F.relu(self.dense_1(x))
            x = F.relu(self.dense_2(x))
            x = self.dense_3(x)
            return x

        def forward(self, x, train=True):
            if train:
                return self._forward(x)
            else:
                with torch.no_grad():
                    return self._forward(x)


elif mode == 'ac_dqn':
    print("-----using ac_dqn------")
    @Registry.register_model('dqn')
    class DQNAgent(RLAgent):
        '''
        DQNAgent determines each intersection's action with its own intersection information.
        '''

        def __init__(self, world, rank):
            super().__init__(world, world.intersection_ids[rank])
            self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
            self.replay_buffer = deque(maxlen=self.buffer_size)

            self.world = world
            self.sub_agents = 1
            self.rank = rank

            self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
            self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']

            # get generator for each DQNAgent
            inter_id = self.world.intersection_ids[self.rank]
            inter_obj = self.world.id2intersection[inter_id]
            self.inter = inter_obj
            self.ob_generator = LaneVehicleGenerator(self.world, self.inter, ['lane_count'], in_only=True, average=None)

            self.phase_generator = IntersectionPhaseGenerator(world, self.inter, ["phase"],
                                                              targets=["cur_phase"], negative=False)
            self.reward_generator = LaneVehicleGenerator(self.world, self.inter, ["lane_waiting_count"],
                                                         in_only=True, average='all', negative=True)
            self.action_space = gym.spaces.Discrete(len(self.world.id2intersection[inter_id].phases))

            if self.phase:
                if self.one_hot:
                    self.ob_length = self.ob_generator.ob_length + len(self.world.id2intersection[inter_id].phases)
                else:
                    self.ob_length = self.ob_generator.ob_length + 1
            else:
                self.ob_length = self.ob_generator.ob_length

            self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
            self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
            self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
            self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
            self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
            self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
            self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
            self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']

            self.forward_model = self._build_for_model()

            self.backward_model = self._build_back_model()
            self.firstTime = 0  # TODO find a nicer way to do this
            self.passable = False

            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_network()
            self.criterion = nn.MSELoss(reduction='mean')
            self.criterion2 = nn.CrossEntropyLoss()

            self.optimizer = optim.RMSprop(self.model.parameters(),
                                           lr=self.learning_rate,
                                           alpha=0.9, centered=False, eps=1e-7)

            self.optimizerFor = optim.Adam(self.forward_model.parameters(),
                                           lr=self.learning_rate)

            self.optimizerBack = optim.Adam(self.backward_model.parameters(),
                                            lr=3e-3)

        def reset(self):
            '''
            reset
            Reset information, including ob_generator, phase_generator, queue, delay, etc.
            :param: None
            :return: None
            '''
            inter_id = self.world.intersection_ids[self.rank]
            inter_obj = self.world.id2intersection[inter_id]
            self.ob_generator = LaneVehicleGenerator(self.world, inter_obj, ['lane_count'], in_only=True, average=None)
            self.phase_generator = IntersectionPhaseGenerator(self.world, inter_obj, ["phase"],
                                                              targets=["cur_phase"], negative=False)
            self.reward_generator = LaneVehicleGenerator(self.world, inter_obj, ["lane_waiting_count"],
                                                         in_only=True, average='all', negative=True)
            self.queue = LaneVehicleGenerator(self.world, inter_obj,
                                              ["lane_waiting_count"], in_only=True,
                                              negative=False)
            self.delay = LaneVehicleGenerator(self.world, inter_obj,
                                              ["lane_delay"], in_only=True, average="all",
                                              negative=False)

        def get_ob(self):
            '''
            get_ob
            Get observation from environment.
            :param: None
            :return x_obs: observation generated by ob_generator
            '''
            x_obs = []
            x_obs.append(self.ob_generator.generate())
            x_obs = np.array(x_obs, dtype=np.float32)
            return x_obs

        def get_reward(self):
            '''
            get_reward
            Get reward from environment.
            :param: None
            :return rewards: rewards generated by reward_generator
            '''
            rewards = []
            rewards.append(self.reward_generator.generate())
            rewards = np.squeeze(np.array(rewards)) * 12
            return rewards

        def get_phase(self):
            '''
            get_phase
            Get current phase of intersection(s) from environment.
            :param: None
            :return phase: current phase generated by phase_generator
            '''
            phase = []
            phase.append(self.phase_generator.generate())
            # phase = np.concatenate(phase, dtype=np.int8)
            phase = (np.concatenate(phase)).astype(np.int8)
            return phase

        def get_action(self, ob, phase, test=False):
            if len(self.replay_buffer) == self.buffer_size and self.passable == False:
                self.passable = True
                print("starting action transformation.....")
                lost = 100000
                lost2 = 100000
                # lost3 = 100000

                for i in range(2000):
                    lost2 = self.train_inv()
                    if (i % 100 == 0):
                        print("two ", lost2)
                print("done")
                for i in range(3000):
                    lost = self.train_for()
                    if (i % 100 == 0):
                        print("one ", lost)
                print("done")
                for i in range(200):
                    lost3 = self.train_together()
                    if (i % 10 == 0):
                        print("three ", lost3)
                print("done")
            '''
            get_action
            Generate action.
            :param ob: observation
            :param phase: current phase
            :param test: boolean, decide whether is test process
            :return action: action that has the highest score
            '''
            if not test:
                if np.random.rand() <= self.epsilon:
                    return self.sample()
            if self.phase:
                if self.one_hot:
                    feature = np.concatenate([ob, utils.idx2onehot(phase, self.action_space.n)], axis=1)
                else:
                    feature = np.concatenate([ob, phase], axis=1)
            else:
                feature = ob
            observation = torch.tensor(feature, dtype=torch.float32)
            # TODO: no need to calculate gradient when interacting with environment
            actions = self.model(observation, train=False)
            actions = actions.clone().detach().numpy()

            # if not test:
            #     if np.random.rand() <= self.epsilon:
            #         return self.sample()
            # if self.phase:
            #     if self.one_hot:
            #         feature = np.concatenate([ob/39, utils.idx2onehot(phase, self.action_space.n)], axis=1)
            #     else:
            #         feature = np.concatenate([ob/39, phase], axis=1)
            # else:
            #     feature = ob/39
            # observation = torch.tensor(feature, dtype=torch.float32)

            # print("actions", actions)
            # print("their future", self.forward_model(torch.concat([observation, torch.tensor(utils.idx2onehot(np.argmax(actions, axis=1), self.action_space.n), dtype=torch.float32)], axis=1), train=False).numpy())
            if (self.passable):
                # print("prepass", np.argmax(actions, axis=1))
                # print(actions)
                actions = self.action_transformation(np.argmax(actions, axis=1), feature)

                # actions = self.action_transformation(actions, feature)
                # print(feature)
                # print("done", np.argmax(actions, axis=1))

                # startArr = [0,1,2,3,4,5,6,7]
                # for i in startArr:
                #    print("begin ", i, " end ", np.argmax(self.action_transformation(np.array([i]), feature), axis=1))
                # print("my future", self.forward_model(torch.concat([observation, torch.tensor(utils.idx2onehot(np.argmax(actions, axis=1), self.action_space.n), dtype=torch.float32)], axis=1), train=False).numpy())

            return np.argmax(actions, axis=1)

        def action_transformation(self, action,
                                  pastState):  # no clue if this works - should do as close to nothing to the results in theory (until transformation is applied)
            # trueAction = utils.idx2onehot(action, self.action_space.n) #should take in the action that the code outputs via DQN
            trueAction = action.reshape(1, 1)
            # print(action)
            # print(pastState)

            feature = np.concatenate([pastState, trueAction],
                                     axis=1)  # maps the past state and the one hot action to the next state
            stateAction = torch.tensor(feature, dtype=torch.float32)
            # print(stateAction)
            futureState = self.forward_model(stateAction, train=False)  # uses the forward model to find the next state

            futureState = futureState.clone().detach().numpy()
            # print(futureState)

            futurePast = np.concatenate([futureState, pastState], axis=1)
            inputs = torch.tensor(futurePast, dtype=torch.float32)

            transformedAction = self.backward_model(inputs, train=False)  # finds action transformed

            transformedAction = transformedAction.clone().detach().numpy()
            # print(transformedAction)
            return transformedAction

        def sample(self):
            '''
            sample
            Sample action randomly.
            :param: None
            :return: action generated randomly.
            '''
            return np.random.randint(0, self.action_space.n, self.sub_agents)

        def _build_back_model(self):  # fp to action, uses sumo
            model = backwardModel(self.ob_length * 2, self.action_space.n)
            return model

        def _build_for_model(self):  # p action to f, uses cityflow
            # model = forwardModel(self.ob_length+self.action_space.n, self.ob_length)
            model = forwardModel(self.ob_length + 1, self.ob_length)
            return model

        def _build_model(self):  # serves the same purpose as DQN, finds best action at a state
            model = DQNNet(self.ob_length, self.action_space.n)
            return model

        def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
            '''
            remember
            Put current step information into replay buffer for training agent later.
            :param last_obs: last step observation
            :param last_phase: last step phase
            :param actions: actions executed by intersections
            :param actions_prob: the probability that the intersections execute the actions
            :param rewards: current step rewards
            :param obs: current step observation
            :param cur_phase: current step phase
            :param done: boolean, decide whether the process is done
            :param key: key to store this record, e.g., episode_step_agentid
            :return: None
            '''
            self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))

        def _batchwise(self, samples, normalize=False):
            '''
            _batchwise
            Reconstruct the samples into batch form(last state, current state, reward, action).
            :param samples: original samples record in replay buffer
            :return state_t, state_tp, rewards, actions: information with batch form
            '''
            obs_t = np.concatenate([item[1][0] for item in samples])
            obs_tp = np.concatenate([item[1][4] for item in samples])
            # if(normalize):
            #     obs_t/=39
            #     obs_tp/=39
            if self.phase:
                if self.one_hot:
                    phase_t = np.concatenate([utils.idx2onehot(item[1][1], self.action_space.n) for item in samples])
                    phase_tp = np.concatenate([utils.idx2onehot(item[1][5], self.action_space.n) for item in samples])
                else:
                    phase_t = np.concatenate([item[1][1] for item in samples])
                    phase_tp = np.concatenate([item[1][5] for item in samples])
                feature_t = np.concatenate([obs_t, phase_t], axis=1)
                feature_tp = np.concatenate([obs_tp, phase_tp], axis=1)
            else:
                feature_t = obs_t
                feature_tp = obs_tp
            state_t = torch.tensor(feature_t, dtype=torch.float32)
            state_tp = torch.tensor(feature_tp, dtype=torch.float32)
            rewards = torch.tensor(np.array([item[1][3] for item in samples]), dtype=torch.float32)  # TODO: BETTER WA
            actions = torch.tensor(np.array([item[1][2] for item in samples]), dtype=torch.long)
            return state_t, state_tp, rewards, actions

        def train_for(self):  # forward training, trained on cityflow
            samples = random.sample(self.replay_buffer, self.batch_size)
            # print(self.batch_size, " batches ", len(self.replay_buffer), " overall")
            b_t, b_tp, rewards, actions = self._batchwise(samples, normalize=True)
            # npinputs = np.concatenate((b_t, utils.idx2onehot(actions, self.action_space.n)), axis=1)
            npinputs = np.concatenate((b_t, actions), axis=1)
            inputs = torch.tensor(npinputs, dtype=torch.float32)
            pred = self.forward_model(inputs, train=True)
            loss = self.criterion(pred, b_tp)
            self.optimizerFor.zero_grad()
            loss.backward()
            clip_grad_norm_(self.forward_model.parameters(), self.grad_clip)
            self.optimizerFor.step()
            return loss.clone().detach().numpy()

        def train_inv(self):  # has to be trained on sumo (?), no clue how - worry abt this later !
            samples = random.sample(self.replay_buffer, self.batch_size)

            b_t, b_tp, rewards, actions = self._batchwise(samples, normalize=True)
            npinputs = np.concatenate([b_tp, b_t], axis=1)  # state i+1, state i
            inputs = torch.tensor(npinputs, dtype=torch.float32)
            pred = self.backward_model(inputs, train=True)
            loss = self.criterion2(pred, actions.flatten())
            # print(torch.argmax(pred, axis=1))
            # print("true", actions.flatten())
            self.optimizerBack.zero_grad()
            loss.backward()
            self.optimizerBack.step()
            return loss.clone().detach().numpy()

        def train_together(self):
            samples = random.sample(self.replay_buffer, self.batch_size)

            b_t, b_tp, rewards, actions = self._batchwise(samples, normalize=True)
            # npinputs = np.concatenate((b_t, utils.idx2onehot(actions, self.action_space.n)), axis=1)
            npinputs = np.concatenate((b_t, actions), axis=1)
            inputs = torch.tensor(npinputs, dtype=torch.float32)
            pred = self.forward_model(inputs, train=False)
            b_tp = pred.numpy()

            npinputs = np.concatenate([b_tp, b_t], axis=1)  # state i+1, state i
            inputs2 = torch.tensor(npinputs, dtype=torch.float32)
            pred2 = self.backward_model(inputs2, train=False)
            loss = self.criterion2(pred2, actions.flatten())
            # print(torch.argmax(pred2, axis=1))
            # print("true", actions.flatten())
            # self.optimizerBack.zero_grad()
            # loss.backward()
            # self.optimizerBack.step()
            return loss.clone().detach().numpy()

        def train(self):
            '''
            train
            Train the agent, optimize the action generated by agent.
            :param: None
            :return: value of loss
            '''
            samples = random.sample(self.replay_buffer, self.batch_size)
            b_t, b_tp, rewards, actions = self._batchwise(samples)
            out = self.target_model(b_tp, train=False)
            target = rewards + self.gamma * torch.max(out, dim=1)[0]
            target_f = self.model(b_t, train=False)
            for i, action in enumerate(actions):
                target_f[i][action] = target[i]
            loss = self.criterion(self.model(b_t, train=True), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return loss.clone().detach().numpy()

        def update_target_network(self):
            '''
            update_target_network
            Update params of target network.
            :param: None
            :return: None
            '''
            weights = self.model.state_dict()
            self.target_model.load_state_dict(weights)

        def load_model(self, e):
            '''
            load_model
            Load model params of an episode.
            :param e: specified episode
            :return: None
            '''
            model_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                      'model', f'{e}_{self.rank}.pt')
            self.model = self._build_model()
            self.model.load_state_dict(torch.load(model_name))
            self.target_model = self._build_model()
            self.target_model.load_state_dict(torch.load(model_name))

        def save_model(self, e):
            '''
            save_model
            Save model params of an episode.
            :param e: specified episode, used for file name
            :return: None
            '''
            path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
            if not os.path.exists(path):
                os.makedirs(path)
            model_name = os.path.join(path, f'{e}_{self.rank}.pt')
            torch.save(self.target_model.state_dict(), model_name)


    # forward model, needs to take in input from SGAT and switch to libsumo
    class forwardModel(nn.Module):  # (s, a -> s')
        def __init__(self, input_dim, output_dim):
            super(forwardModel, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_dim, 50),
                nn.ReLU(),
                nn.Linear(50, 50),
                nn.ReLU(),
                nn.Linear(50, output_dim),
            )

        def _forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

        def forward(self, x, train=True):
            if train:
                return self._forward(x)
            else:
                with torch.no_grad():
                    return self._forward(x)


    # inverse model, somehow have to switch to traci for this and switch back afterwards
    class backwardModel(nn.Module):  # (s', s -> a)
        def __init__(self, input_dim, output_dim):
            super(backwardModel, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_dim, 50),
                nn.ReLU(),
                nn.Linear(50, 50),
                nn.ReLU(),
                nn.Linear(50, output_dim),
            )

        def _forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            # logits = nn.Softmax()(logits)
            return logits

        def forward(self, x, train=True):
            if train:
                return self._forward(x)
            else:
                with torch.no_grad():
                    return self._forward(x)


    class DQNNet(nn.Module):  # can be replaced with any algorithm that finds the best action at a given state
        def __init__(self, input_dim, output_dim):
            super(DQNNet, self).__init__()
            self.dense_1 = nn.Linear(input_dim, 20)
            self.dense_2 = nn.Linear(20, 20)
            self.dense_3 = nn.Linear(20, output_dim)

        def _forward(self, x):
            x = F.relu(self.dense_1(x))
            x = F.relu(self.dense_2(x))
            x = self.dense_3(x)
            return x

        def forward(self, x, train=True):
            if train:
                return self._forward(x)
            else:
                with torch.no_grad():
                    return self._forward(x)