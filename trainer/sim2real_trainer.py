import datetime
import os
import sys
import time
import pickle as pkl
import numpy as np
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer

from tqdm import tqdm
from common import interface
from common.stat_utils import NN_predictor, UNCERTAINTY_predictor
from agent.utils import idx2onehot
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
import torch.nn.functional as F
from  torch import optim, nn
from prompt_data.ChatGPT.prompt_api import Conversation, split_data
from common.stat_utils import get_weather, prompt_for_info


HISTORY_T = 4
INVERSE = True
Ground_All = True
experiment_mode = 'restart' # 'pretrained'/'restart' # now is the -1

file_path = "../prompt_data/"
save_data =  False # whether to save data or not (from the prompt process)
appendix = "data100.pt" # if to save data, the file will be name as this
pretrain_num = 5
my_key = ''
creat_conv = True

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

@Registry.register_trainer("sim2real")
class SIM2REALTrainer(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''

    def __init__(
            self,
            logger,
            config,
            gpu=0,
            cpu=False,
            name="sim2real"
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.config = config

        self.learning_rate = 0.0001 # set for before sim2real manually
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        # self.learning_rate = Registry.mapping['trainer_mapping']['setting'].param['learning_rate']
        # self.batch_size = Registry.mapping['trainer_mapping']['setting'].param['batch_size']

        # self.optimizer = optim.Adam(params=self.alpha_predictor.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))
        # self.alpha = Registry.mapping['trainer_mapping']['setting'].param['alpha']

        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        # replay file is only valid in cityflow now. 
        # TODO: support SUMO and Openengine later

        # TODO: support other dataset in the future
        self.dataset = Registry.mapping['dataset_mapping'][
            Registry.mapping['command_mapping']['setting'].param['dataset']](
            os.path.join(Registry.mapping['logger_mapping']['path'].path,
                         Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        )
        time_date = str(datetime.now())[:-7].replace(":", "_").replace(" ", "_").replace("-", "_")
        self.writer_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'writer/' + time_date)
        if not os.path.exists(self.writer_path):
            os.makedirs(self.writer_path)
        self.writer = SummaryWriter(log_dir=self.writer_path)

        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']
        # consists of path of output dir + log_dir + file handlers name
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     os.path.basename(self.logger.handlers[-1].baseFilename).rstrip(
                                         '_BRF.log') + '_DTL.log'
                                     )

    def create_world(self):
        '''
        create_world
        Create world, currently support CityFlow World, SUMO World and Citypb World.

        :param: None
        :return: None
        '''
        # traffic setting is in the world mapping
        self.world_sim = Registry.mapping['world_mapping']['cityflow'](
            self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'])

        self.world_real = Registry.mapping['world_mapping']['sumo'](
            self.path.replace('cityflow', 'sumohz'),
            interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def create_metrics(self):
        '''
        create_metrics
        Create metrics to evaluate model performance, currently support reward, queue length, delay(approximate or real) and throughput.

        :param: None
        :return: None
        '''
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards', 'queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
        self.metric_sim = Metrics(lane_metrics, world_metrics, self.world_sim, self.agents_sim)
        self.metric_real = Metrics(lane_metrics, world_metrics, self.world_real, self.agents_real)

    def create_agents(self):
        '''
        create_agents
        Create agents for traffic signal control tasks.

        :param: None
        :return: None
        '''
        self.agents_sim = []
        agent_sim = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
            self.world_sim, 0)

        num_agent = int(len(self.world_sim.intersections) / agent_sim.sub_agents)
        self.agents_sim.append(agent_sim)  # initialized N agents for traffic light control
        for i in range(1, num_agent):
            self.agents_sim.append(
                Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
                    self.world_sim, i))

        self.agents_real = []
        agent_real = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
            self.world_real, 0)

        num_agent = int(len(self.world_real.intersections) / agent_real.sub_agents)
        self.agents_real.append(agent_real)  # initialized N agents for traffic light control
        for i in range(1, num_agent):
            self.agents_real.append(
                Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
                    self.world_real, i))

    def create_env(self):
        '''
        create_env
        Create simulation environment for communication with agents.

        :param: None
        :return: None
        '''
        # TODO: finalized list or non list
        self.env_sim = TSCEnv(self.world_sim, self.agents_sim, self.metric_sim)
        self.env_real = TSCEnv(self.world_real, self.agents_real, self.metric_real)

    def run(self):
        global e
        self.create_world()
        self.create_agents()
        self.create_metrics()
        self.create_env()
        self.root_path = Registry.mapping['logger_mapping']['path'].path
        self.model_path = os.path.join(self.root_path, Registry.mapping['logger_mapping']['setting'].param['model_dir'])
        self.data_path = os.path.join(self.root_path, Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        self.debug_path = os.path.join(self.root_path, 'debug')
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.exists(self.debug_path):
            os.mkdir(self.debug_path)
        # TODO: support multiple intersections

        if INVERSE == True:
            print(self.logger)
            # forward model is used to predict the next state bease one the given (s, a) ==> s_{t+1}
            self.forward_model = NN_predictor(self.logger,
                                            (self.agents_real[0].ob_generator.ob_length + self.agents_real[0].action_space.n) * HISTORY_T,
                                            self.agents_real[0].ob_generator.ob_length, device, self.model_path,
                                            self.data_path + 'real.pkl', history=HISTORY_T)
            
            self.inverse_model = UNCERTAINTY_predictor(self.logger, self.agents_real[0].ob_generator.ob_length * 2,
                                            self.agents_real[0].action_space.n, device, self.model_path,
                                            self.data_path + 'sim.pkl', backward=True)
        else:
            exit()

        
        #'pretrained'/'restart'
        if experiment_mode == 'pretrained': 
            path = 'data/output_data/sim2real/cityflow_dqn/cityflow1x1/report/s_model_collect/-1_0.pt'

        elif experiment_mode =='restart':
            path = self.pretrain(episodes=pretrain_num)
        
        # load knowledge data before hand:

        # pretrain model and load pretrained ones
        best_e = -1
        V = []
        for e in range(self.episodes):
            self.load_pretrained(path)  # load both on the sim and real
            # self.sim_rollout(e-1)
            print("starting rollout...")
            R = self.real_rollout(e - 1)
            self.sim_rollout(e - 1)
            V.append(R)
            self.forward_train(writer=self.writer)
            # get the uncertainty value from inverse training
            uncertainty_list = self.inverse_train(writer=self.writer)
            # mean_uncer = torch.Tensor(np.array(np.mean(uncertainty_list)))
            path = self.sim_tain(episode=20, e=e, writer=self.writer)
            # self.alpha *= 0.95

        self.load_pretrained(path)
        R = self.real_eval(e)
        V.append(R)

    print('-' * 10 + 'finished' + '-' * 10)

    def test_model(self, target_model):
        global e
        self.create_world()
        self.create_agents()
        self.create_metrics()
        self.create_env()
        self.root_path = Registry.mapping['logger_mapping']['path'].path
        self.model_path = os.path.join(self.root_path, Registry.mapping['logger_mapping']['setting'].param['model_dir'])
        self.data_path = os.path.join(self.root_path, Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        self.debug_path = os.path.join(self.root_path, 'debug')
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.exists(self.debug_path):
            os.mkdir(self.debug_path)
        # TODO: support multiple intersections

        if INVERSE == True:
            print(self.logger)
            # forward model is used to predict the next state bease one the given (s, a) ==> s_{t+1}
            self.forward_model = NN_predictor(self.logger,
                                            (self.agents_real[0].ob_generator.ob_length + self.agents_real[0].action_space.n) * HISTORY_T,
                                            self.agents_real[0].ob_generator.ob_length, device, self.model_path,
                                            self.data_path + 'real.pkl', history=HISTORY_T)
            
            self.inverse_model = UNCERTAINTY_predictor(self.logger, self.agents_real[0].ob_generator.ob_length * 2,
                                            self.agents_real[0].action_space.n, device, self.model_path,
                                            self.data_path + 'sim.pkl', backward=True)

        # load the model path to be evaluated
        path = target_model

        # normally only take 1 episode to evaluate
        for e in range(1):
            self.load_test(path)  # load both on the sim and real
            # mean_uncer = torch.Tensor(np.array(np.mean(uncertainty_list)))
            time_start = time.time()
            path = self.sim_test(e=e, writer=self.writer)
            # self.alpha *= 0.95
            print("test duration: "+str(time.time()-time_start))

        # self.load_pretrained(path)

    print('finished testing!')

    def pretrain(self, episodes):
        """
        pretrain a model for in the simulator(cityflows)
        """

        # initialize the Conversation
        if creat_conv:
            conv = Conversation(my_key)
        else: 
            conv = None
        
        record_all = []

        total_decision_num = 0
        pretrained_episode = episodes
        flush = 0
        states_record = []
        action_record = []
        for e in range(pretrained_episode):
            # TODO: check this reset agent
            self.metric_sim.clear()
            last_obs = self.env_sim.reset()  # agent * [sub_agent, feature]
            epo_action_record = []
            epo_states_record = []

            epo_states_record.append(last_obs[0])
            episode_loss = []
            i = 0
            weather_type = conv.get_weather()[0]
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents_sim])  # [agent, intersections]

                    if total_decision_num > self.learning_start:
                        actions = []
                        for idx, ag in enumerate(self.agents_sim):
                            # we could modify the input observation: [origin + prompt knowledge]:
                            take_action = ag.get_action(last_obs[idx], last_phase[idx], test=False)
                            actions.append(take_action)
                            # print(last_obs[idx])
                            # veichle_num = str(20)
                            # capacity_num = str(90)
                            # prompt_info = "Given the knowledge: during snowy weather, the average acceleration of the vehicle is 0.5 meters per second squared, the average deceleration of the vehicle is 1.5 meters per second squared, and the average emergency deceleration of the vehicle is about 2 meters per second squared and the average delay is 0.5 second. Knowing that these indicatiors might vary based on the vehicle number and snowy level, they will have bias based on the standard value. If for a lane with " +veichle_num+ " vehicles while capacity is " + capacity_num +" , and under "+ weather_type +"weather, please assume the four indicators (average acceleration, average deceleration, average emergency deceleration and average delay) in the format of (replace {value } as assumed value, only one value allowed): [average acceleration: {value}], [average deceleration: {value}], [average emergency deceleration: {value}], [average delay: {value}]"

                            # res = conv.get_response(prompt_info=prompt_info, max_token=2000, model="text-davinci-003")

                            # resu = conv.get_response_text(res)
                            # spl = split_data(resu[10:])


                        actions = np.stack(actions)  # [agent, intersections]
                    else:
                        actions = np.stack([ag.sample() for ag in self.agents_sim])

                    actions_prob = []
                    for idx, ag in enumerate(self.agents_sim):
                        actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))

                    rewards_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env_sim.step(actions.flatten())
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                    self.metric_sim.update(rewards)

                    # debug
                    epo_states_record.append(obs[0])

                                        
                    cur_phase = np.stack([ag.get_phase() for ag in self.agents_sim])
                    for idx, ag in enumerate(self.agents_sim):
                        ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx], rewards[idx],
                                    obs[idx], cur_phase[idx], dones[idx], f'{e}_{i // self.action_interval}_{ag.id}')
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
                    total_decision_num += 1

                    # updated the last_obs
                    last_obs = obs
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                    cur_loss_q = np.stack([ag.train() for ag in self.agents_sim])  # TODO: training

                    episode_loss.append(cur_loss_q)
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    [ag.update_target_network() for ag in self.agents_sim]

                if all(dones):
                    break
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0
            self.logger.info("episode:{}/{}, real avg travel time:{}".format(e, pretrained_episode,
                                                                             self.metric_sim.real_average_travel_time()))
            # for j in range(len(self.world_sim.intersections)):
            #     self.logger.debug(
            #         "intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j,
            #                                                                         self.metric_sim.lane_rewards()[j], \
            #                                                                         self.metric_sim.lane_queue()[j]))

            action_record.append(epo_action_record)
            states_record.append(epo_states_record)
            record_all.append({"episode_"+str(e):states_record})
            self.writeLog("TRAIN", e, self.metric_sim.real_average_travel_time(), \
                          mean_loss, self.metric_sim.rewards(), self.metric_sim.queue(), self.metric_sim.delay(),
                          self.metric_sim.throughput())
        # self.test(e)
        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
        model_path = self.agents_sim[-1].save_model(e=-1)
        if save_data:
            torch.save(record_all, file_path+appendix)
            print("successfully saved!")
        return model_path

    def load_pretrained(self, model_path):

        base_path = sys.path[0] + "/"
        real_path = base_path + str(model_path)
        [ag.load_model(e="", customized_path=real_path) for ag in self.agents_sim]
        [ag.load_model(e="", customized_path=real_path) for ag in self.agents_real]
        time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print("timestamp:{} successfully loaded a model with param {}".format(time_stamp, model_path))
    
    def load_test(self, model_path):

        [ag.load_model(e="", customized_path=model_path) for ag in self.agents_sim]
        [ag.load_model(e="", customized_path=model_path) for ag in self.agents_real]
        time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print("timestamp:{} successfully loaded a model with param {}".format(time_stamp, model_path))

    def sim_rollout(self, e):
        # TODO: support multiple intersections

        self.metric_sim.clear()
        obs = self.env_sim.reset()
        action_record = []
        states_record = []
        states_record.append(obs[0])
        for i in range(self.steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents_sim])
                actions = []
                for idx, ag in enumerate(self.agents_sim):
                    action = ag.get_action(obs[idx], phases[idx], test=True)
                    actions.append(action)

                actions = np.stack(actions)
                action_record.append(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env_sim.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric_sim.update(rewards)
                states_record.append(obs[0])
            if all(dones):
                break
        with open(os.path.join(self.data_path, 'sim.pkl'), 'wb') as f:
            pkl.dump([[states_record], [action_record]], f)

        self.logger.info(
            "Epoch %d Sim Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (e,
                                                                                                                self.metric_sim.real_average_travel_time(), \
                                                                                                                self.metric_sim.rewards(),
                                                                                                                self.metric_sim.queue(),
                                                                                                                self.metric_sim.delay(),
                                                                                                                self.metric_sim.throughput()))
        return

    def real_rollout(self, e):
        # TODO: support multiple intersections

        if creat_conv:
            conv = Conversation(my_key)
        else: 
            conv = None

        discount = 0.95
        self.metric_real.clear()
        R = np.array([0 for _ in self.agents_real], dtype=np.float64)
        obs = self.env_real.reset()
        action_record = []
        states_record = []
        for a in self.agents_real:
            a.reset()
        states_record.append(obs[0])
        for i in range(self.steps):
            discount *= discount
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents_real])
                actions = []
                for idx, ag in enumerate(self.agents_real):
                    action = ag.get_action(obs[idx], phases[idx], test=True)
                    actions.append(action)

                actions = np.stack(actions)
                action_record.append(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env_real.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                R += discount * rewards
                self.metric_real.update(rewards)
                states_record.append(obs[0])
            if all(dones):
                break

        with open(os.path.join(self.data_path, 'real.pkl'), 'wb') as f:
            pkl.dump([[states_record], [action_record]], f)

        self.logger.info(
            "Epoch %d Real Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (e,
                                                                                                                 self.metric_real.real_average_travel_time(), \
                                                                                                                 self.metric_real.rewards(),
                                                                                                                 self.metric_real.queue(),
                                                                                                                 self.metric_real.delay(),
                                                                                                                 self.metric_real.throughput()))
        # 
        return R

    def forward_train(self, writer):
        if creat_conv:
            conv = Conversation(my_key)
        else: 
            conv = None
        self.forward_model.load_dataset(conv = conv)
        self.forward_model.train(100, writer=writer, sign='forward')

    def inverse_train(self, writer):
        self.inverse_model.load_dataset()
        # get the uncertainty feedback from inverse training
        uncertainty_list = self.inverse_model.train(100, writer=writer, sign='inverse')
        return uncertainty_list

    def sim_tain(self, episode, e, writer):

        prompt_forward = True

        if prompt_forward:
            pass

        action_diff = []
        action_distribution = []
        action_uncertainty = []

        total_decision_num = 0
        [ag.replay_buffer.clear() for ag in self.agents_sim]

        if Ground_All == True:
            self.alpha = np.zeros(1)
        for ep in range(episode):
            epoch_diff = []
            epoch_distribution = []
            epoch_uncertainty = []
           
            self.metric_sim.clear()
            last_obs = self.env_sim.reset()
            episode_loss = []
            i = 0
            input_size = (16 + ((4 + 1) * 8)) * HISTORY_T
            seq = np.zeros((1, input_size))

            if creat_conv:
                conv = Conversation(my_key)
            else:
                conv = None
            # with tqdm(total=self.steps, desc='Progress') as pbar:
            while i < self.steps:
                # pbar.update(1)
                # print(i)
                if i % self.action_interval == 0:
                    # print(i)
                    last_phase = np.stack([ag.get_phase() for ag in self.agents_sim])  # [agent, intersections]

                    actions = []
                    for idx, ag in enumerate(self.agents_sim):
                        actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))
                    actions = np.stack(actions)  # [agent, intersections]
                    actions_prob = []
                    for idx, ag in enumerate(self.agents_sim):
                        actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx])) 
                        weather_name, weather_value = get_weather()
                        s_prime = []
                        phase_info = idx2onehot(actions[0], 8)
                        path_info = []
                        for k in range(len(last_obs[0][0])):
                            veichle_num = last_obs[0][0][k]
                            phase_value = phase_info[0][k]

                            # GPT responses: 
                            prompt_element = [veichle_num, phase_value, weather_name]
                            path_i = prompt_for_info(prompt_element, conv=conv)

                            path_info.append(path_i)
                        
                        path_info = np.array(path_info).reshape(1, 8, -1)
                        weather_re = np.expand_dims(np.repeat([weather_value], len(last_obs[0][0])), axis=(0, 2)) # (1, 8, 1)
                        ob_re = np.expand_dims(last_obs[0], axis = 2) # (1, 8) => (1, 8, 1)
                        one_hot_re = np.expand_dims(phase_info, axis=2) # (1, 8) => (1, 8, 1)

                        new_feature = np.concatenate((ob_re, one_hot_re, path_info, weather_re), axis=2) # (1, 8, 7) [1, 1, 4, 1]
                        # save_prompt_data(str(new_feature))
                        x = np.reshape(new_feature, (1, -1))

                        # forward_input = np.concatenate((last_obs[0], phase_info), axis=1)
                        forward_input = x
                        seq = np.concatenate((seq, forward_input), axis=1)

                        s_prime.append(self.forward_model.predict(torch.from_numpy(seq[:, -input_size :]).float())[0])
                        grounding_actions = []
                        inverse_input = torch.cat((torch.from_numpy(last_obs[0]).to(device), s_prime[0]), dim=1).float()

                        if INVERSE ==True:
                            logit_distribution, uncertainty = self.inverse_model.predict(inverse_input)
                            distribution = F.softmax(logit_distribution)
                            action_max = np.array(np.argmax(logit_distribution.to(device).cpu().numpy()))
                            # distribution, action_max, uncertainty = self.inverse_model.predict(inverse_input)
                            grounding_actions.append([np.array(action_max)])
                            grounding_actions = np.stack(grounding_actions)
                            epoch_diff.append([actions[0][0], grounding_actions[0][0]])
                            epoch_distribution.append(distribution)
                            epoch_uncertainty.append(uncertainty)


                    # no uncertainty considered (always false)
                    if uncertainty.cpu().detach().numpy()[0] < self.alpha: #self.alpha

                        rewards_list = []
                        for _ in range(self.action_interval):
                            obs, rewards, dones, _ = self.env_sim.step(grounding_actions.flatten())
                            i += 1
                            rewards_list.append(np.stack(rewards))

                        seq[:, -8:] = idx2onehot(grounding_actions[0], 8)
                        rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                        ####### add loss here
                        # loss = (alpha_v/alpha_v) * torch.FloatTensor(- rewards)
                        # loss.backward()
                        # self.optimizer.step()
                        self.metric_sim.update(rewards)
                        cur_phase = np.stack([ag.get_phase() for ag in self.agents_sim])
                        for idx, ag in enumerate(self.agents_sim):
                            ag.remember(last_obs[idx], last_phase[idx], grounding_actions[idx], actions_prob[idx],
                                        rewards[idx],
                                        obs[idx], cur_phase[idx], dones[idx],
                                        f'{e}_{i // self.action_interval}_{ag.id}')

                    else:
                        # take original action
                        epoch_diff.append([actions[0][0], -1])
                        rewards_list = []

                        # haven't edited for the prompt version, because we will not access this condition
                        forward_input = np.concatenate((last_obs[0], idx2onehot(actions[0], 8)), axis=1)
                        seq = np.concatenate((seq, forward_input), axis=1)
                        for _ in range(self.action_interval):
                            obs, rewards, dones, _ = self.env_sim.step(actions.flatten())
                            i += 1
                            rewards_list.append(np.stack(rewards))
                        rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]

                        self.metric_sim.update(rewards)
                        cur_phase = np.stack([ag.get_phase() for ag in self.agents_sim])
                        for idx, ag in enumerate(self.agents_sim):
                            ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx],
                                        rewards[idx],
                                        obs[idx], cur_phase[idx], dones[idx],
                                        f'{e}_{i // self.action_interval}_{ag.id}')

                    total_decision_num += 1
                    last_obs = obs
                
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                    cur_loss_q = np.stack([ag.train() for ag in self.agents_sim])  # TODO: training
                    episode_loss.append(cur_loss_q)
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    [ag.update_target_network() for ag in self.agents_sim]

                if all(dones):
                    break
                    
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0
            action_diff.append(epoch_diff)
            action_distribution.append(epoch_distribution)

            if INVERSE == True:
                temp = np.vstack([e.cpu() for e in epoch_uncertainty]).flatten().mean()
                action_uncertainty.append(temp)

            self.logger.info("episode:{}/{}, real avg travel time:{}".format(ep, episode,
                                                                             self.metric_sim.real_average_travel_time()))
            self.writeLog("TRAIN", ep, self.metric_sim.real_average_travel_time(), \
                          mean_loss, self.metric_sim.rewards(), self.metric_sim.queue(), self.metric_sim.delay(),
                          self.metric_sim.throughput())

            writer.add_scalar("real_a_travel time/simu_train", self.metric_sim.real_average_travel_time(), e)
            writer.add_scalar("real_throughput/simu_train", self.metric_sim.throughput(), e)
            writer.add_scalar("rewards/simu_train", self.metric_sim.rewards(), e)
            writer.add_scalar("queue/simu_train", self.metric_sim.queue(), e)
            writer.add_scalar("delay/simu_train", self.metric_sim.queue(), e)
            writer.add_scalar("mean_loss/simu_train", mean_loss, e)
            # print(epoch_uncertainty)
            
            writer.add_scalar("uncertanty/act_uncertainty_epo", temp, e)

        self.test(e=e)
        writer.flush()
        act_dif = np.concatenate(action_diff)
        # act_distribution = np.vstack(action_distribution)
        if INVERSE == True:
            act_uncertainty = np.vstack(action_uncertainty)
            np.save(os.path.join(self.debug_path, f'act_uncertainty_{e}.npy'), act_uncertainty)
            # dynamically adapt the uncertainty
            self.alpha = np.mean(act_uncertainty)

        np.save(os.path.join(self.debug_path, f'act_diff_{e}.npy'), act_dif)
        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
        model_path = self.agents_sim[-1].save_model(e=e)
        return model_path
    
    
    def sim_test(self, e, writer):
        # used for test
        prompt_forward = True

        if prompt_forward:
            pass
        # [ag.load_model(e='', customized_path=path) for ag in self.agents_sim]
        [ag.replay_buffer.clear() for ag in self.agents_sim]

        self.test(e=e)
        writer.flush()
        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
        return 0

    def real_eval(self, e):
        # TODO: support multiple intersections
        if creat_conv:
            conv = Conversation(my_key)
        else:
            conv = None
        
        discount = 0.95
        self.metric_real.clear()
        R = np.array([0 for _ in self.agents_real])
        obs = self.env_real.reset()
        for a in self.agents_real:
            a.reset()
        for i in range(self.test_steps):
            discount *= discount
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents_real])
                actions = []
                for idx, ag in enumerate(self.agents_real):
                    action = ag.get_action(obs[idx], phases[idx], test=True)
                    actions.append(action)

                actions = np.stack(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env_real.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                R = R + discount * rewards
                self.metric_real.update(rewards)
            if all(dones):
                break

        self.logger.info(
            "Final Real Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (
                self.metric_real.real_average_travel_time(), \
                self.metric_real.rewards(), self.metric_real.queue(), self.metric_real.delay(),
                self.metric_real.throughput()))
        return R

    def test(self, e):
        if creat_conv:
            conv = Conversation(my_key)
        else:
            conv = None
        self.metric_sim.clear()
        obs = self.env_sim.reset()
        act_time = []
        for i in range(self.steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents_sim])
                actions = []
                for idx, ag in enumerate(self.agents_sim):
                    time_now = time.time()
                    action = ag.get_action(obs[idx], phases[idx], test=True)
                    duration = time.time()-time_now
                    act_time.append(duration)
                    print("action time: "+str(duration))
                    actions.append(action)
                

                actions = np.stack(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env_sim.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric_sim.update(rewards)
            if all(dones):
                break
        print("mean_execution time: {}".format(np.mean(np.array(act_time))))
        print("mean_execution std: {}".format(np.std(np.array(act_time))))
        self.logger.info(
            "Eval sim Travel Time Epoch: %d is %.4f, throughput: %d" % (e, self.metric_sim.real_average_travel_time(), self.metric_sim.throughput()))

        return

    def writeLog(self, mode, step, travel_time, loss, cur_rwd, cur_queue, cur_delay, cur_throughput):
        '''
        writeLog
        Write log for record and debug.

        :param mode: "TRAIN" or "TEST"
        :param step: current step in simulation
        :param travel_time: current travel time
        :param loss: current loss
        :param cur_rwd: current reward
        :param cur_queue: current queue length
        :param cur_delay: current delay
        :param cur_throughput: current throughput
        :return: None
        '''
        res = Registry.mapping['model_mapping']['setting'].param['name'] + '\t' + mode + '\t' + str(
            step) + '\t' + "%.1f" % travel_time + '\t' + "%.1f" % loss + "\t" + \
              "%.2f" % cur_rwd + "\t" + "%.2f" % cur_queue + "\t" + "%.2f" % cur_delay + "\t" + "%d" % cur_throughput
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()
