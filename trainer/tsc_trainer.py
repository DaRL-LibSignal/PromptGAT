import os
import sys
import pickle as pkl
import numpy as np
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer
import datetime
from common.stat_utils import log_passing_lane_actinon, write_action_record


@Registry.register_trainer("tsc")
class TSCTrainer(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''

    def __init__(
            self,
            logger,
            gpu=0,
            cpu=False,
            name="tsc"
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        self.test_when_train = Registry.mapping['trainer_mapping']['setting'].param['test_when_train']
        # replay file is only valid in cityflow now. 
        # TODO: support SUMO and Openengine later

        # TODO: support other dataset in the future
        self.create()
        self.dataset = Registry.mapping['dataset_mapping'][
            Registry.mapping['command_mapping']['setting'].param['dataset']](
            os.path.join(Registry.mapping['logger_mapping']['path'].path,
                         Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        )
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
        self.world = Registry.mapping['world_mapping'][Registry.mapping['command_mapping']['setting'].param['world']](
            self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'],
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
        self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        '''
        create_agents
        Create agents for traffic signal control tasks.

        :param: None
        :return: None
        '''
        self.agents = []
        agent = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
            self.world, 0)
        print(agent)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)  # initialized N agents for traffic light control
        for i in range(1, num_agent):
            self.agents.append(
                Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
                    self.world, i))

        # for magd agents should share information 
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'magd':
            for ag in self.agents:
                ag.link_agents(self.agents)

    def create_env(self):
        '''
        create_env
        Create simulation environment for communication with agents.

        :param: None
        :return: None
        '''
        # TODO: finalized list or non list
        self.env = TSCEnv(self.world, self.agents, self.metric)

    def train(self):
        '''
        train
        Train the agent(s).

        :param: None
        :return: None
        '''

        total_decision_num = 0
        flush = 0
        states_record = []
        action_record = []
        for e in range(self.episodes):
            # TODO: check this reset agent
            self.metric.clear()
            last_obs = self.env.reset()  # agent * [sub_agent, feature]
            epo_action_record = []
            epo_states_record = []
            for a in self.agents:
                a.reset()
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay and e % self.save_rate == 0:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{e}.txt"))
                else:
                    self.env.eng.set_save_replay(False)
            # debug
            epo_states_record.append(last_obs[0])
            episode_loss = []
            i = 0
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents])  # [agent, intersections]

                    if total_decision_num > self.learning_start:
                        actions = []
                        for idx, ag in enumerate(self.agents):
                            actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))
                        actions = np.stack(actions)  # [agent, intersections]
                    else:
                        actions = np.stack([ag.sample() for ag in self.agents])
                    # debug
                    epo_action_record.append(actions)

                    actions_prob = []
                    for idx, ag in enumerate(self.agents):
                        actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))

                    rewards_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env.step(actions.flatten())
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                    self.metric.update(rewards)

                    # debug
                    epo_states_record.append(obs[0])
                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])
                    for idx, ag in enumerate(self.agents):
                        ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx], rewards[idx],
                                    obs[idx], cur_phase[idx], dones[idx], f'{e}_{i // self.action_interval}_{ag.id}')
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
                    total_decision_num += 1
                    last_obs = obs
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                    cur_loss_q = np.stack([ag.train() for ag in self.agents])  # TODO: training

                    episode_loss.append(cur_loss_q)
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    [ag.update_target_network() for ag in self.agents]

                if all(dones):
                    break
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0

            self.writeLog("TRAIN", e, self.metric.real_average_travel_time(), \
                          mean_loss, self.metric.rewards(), self.metric.queue(), self.metric.delay(),
                          self.metric.throughput())
            self.logger.info(
                "step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(i, self.steps, \
                                                                                              mean_loss,
                                                                                              self.metric.rewards(),
                                                                                              self.metric.queue(),
                                                                                              self.metric.delay(),
                                                                                              int(self.metric.throughput())))
            if e % self.save_rate == 0:
                [ag.save_model(e=e) for ag in self.agents]
            self.logger.info("episode:{}/{}, real avg travel time:{}".format(e, self.episodes,
                                                                             self.metric.real_average_travel_time()))
            for j in range(len(self.world.intersections)):
                self.logger.debug(
                    "intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, self.metric.lane_rewards()[j], \
                                                                                    self.metric.lane_queue()[j]))
            if self.test_when_train:
                self.train_test(e)

            action_record.append(epo_action_record)
            states_record.append(epo_states_record)
        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
        [ag.save_model(e=self.episodes) for ag in self.agents]

        path = 'collected'
        if not os.path.exists(path):
            os.mkdir(path)
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            file = os.path.join(path, 'train.pkl')
            with open(file, 'wb') as f:
                pkl.dump([states_record, action_record], f)
        elif Registry.mapping['command_mapping']['setting'].param['world'] == 'sumo':
            file = os.path.join(path, 'test.pkl')
            with open(file, 'wb') as f:
                pkl.dump([states_record, action_record], f)



    def train_test(self, e):
        '''
        train_test
        Evaluate model performance after each episode training process.

        :param e: number of episode
        :return self.metric.real_average_travel_time: travel time of vehicles
        '''
        obs = self.env.reset()
        self.metric.clear()
        for a in self.agents:
            a.reset()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())  # make sure action is [intersection]
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
            if all(dones):
                break
        self.logger.info("Test step:{}/{}, travel time :{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format( \
            e, self.episodes, self.metric.real_average_travel_time(), self.metric.rewards(), \
            self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))
        self.writeLog("TEST", e, self.metric.real_average_travel_time(), \
                      100, self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput())
        return self.metric.real_average_travel_time()

    def test(self, drop_load=True):
        '''
        test
        Test process. Evaluate model performance.

        :param drop_load: decide whether to load pretrained model's parameters
        :return self.metric: including queue length, throughput, delay and travel time
        '''
        cityflow_trained_save = '../data/output_data/tsc/sim2real_paper/compare/cityflow_train_dqn_pickout/cityflow1x1/test/model/'
        cityflow_sub = 'cityflow_dqn/cityflow1x1'
        sumo_sub = 'sumo_dqn/sumohz1x1'

        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            path_sub = cityflow_sub
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        else:
            path_sub = sumo_sub
            
        self.metric.clear()
        # load_path = 'data/output_data/tsc/'+path_sub+'/test/model/'
        load_path = cityflow_trained_save

        if not drop_load:
            print(".......not droping loading, generating random agents.......")
            [ag.load_model(self.episodes) for ag in self.agents]

        else:
            # loaded_agent_list = []
            import re

            base_path = sys.path[0] + Registry.mapping['logger_mapping']['path'].path + '/model/'
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            print(base_path)
            history_dir_path = sys.path[0] + "/history_save/" + Registry.mapping['command_mapping']['setting'].param['world'] + datetime.datetime.now().strftime('%Y-%m-%d:%H-%M-%S')
            os.makedirs(history_dir_path)
            history_save_path = history_dir_path + "/history.txt"
            pass_save_path = history_dir_path + "/action_pass.txt"
            num = 200
            candidate_list = []
            
            for files in os.listdir(load_path):  
                print("files", files)
                if files.startswith(str(num)):
                    candidate_list.append(files)
                    # print(files)
            candidate_list.sort(key=lambda l: int(re.findall('\d+', l[3:])[0]))
            print(candidate_list)
            for i in range(len(candidate_list)):
                # [ag.load_model(e="", customized_path=base_path + candidate_list[i]) for ag in self.agents]
                [ag.load_model(e="", customized_path=load_path + str(num) + "_" + str(ag.rank) + ".pt") for ag in
                 self.agents]

        attention_mat_list = []
        obs = self.env.reset()
        for a in self.agents:
            a.reset()
        print("-------self.test_steps-------")
        print(self.test_steps)
        # my record files:

        history_record = []
        struc = []
        for a in self.agents:
            a_struc = []
            for ls in a.ob_generator.lanes:
                for l in ls:
                    a_struc.append(l)
            struc.append(a_struc)
        history_record.append(str(struc))

        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, ag in enumerate(self.agents):
                    temp = str([str(int(i)).ljust(5, ' ') for i in obs[idx][0]])
                    temp_action = ag.get_action(obs[idx], phases[idx], test=True)
                    # 3 placeholder to align output state
                    actions.append(temp_action)

                    temp +=  ":" + str(temp_action)
                    temp = temp.replace(',', '').replace("'", "")
                    history_record.append(temp)

                actions = np.stack(actions)
                rewards_list = []
                for j in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())

                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
            if all(dones):
                break

        # if Registry.mapping['command_mapping']['setting'].param['debug']:
        with open(file=load_path+"/record.txt", mode='a+', encoding='utf-8') as wf:
            for line in history_record:
                net_info =Registry.mapping['command_mapping']['setting'].param['network']
                wf.writelines(net_info + ":   " +"Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (
                    self.metric.real_average_travel_time(), \
                    self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput()
                    ) + "\n")
            # calculate existing vehicles in each phase (fixedtime only)
            traj = self.env.world.vehicle_trajectory
            path_record = log_passing_lane_actinon(traj, self.world.intersections[0].startlanes)
            write_action_record(pass_save_path, path_record, a_struc)
            

        self.logger.info("Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (
            self.metric.real_average_travel_time(), \
            self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput()))

        return self.metric

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
