import os.path
import sys

print(sys.path)

import task
import trainer
import agent
import dataset
from common import interface
from common.registry import Registry

from common.utils import *
from utils.logger import *
import time
from datetime import datetime
import argparse


# parseargs
parser = argparse.ArgumentParser(description='Run Experiment')
parser.add_argument('--thread_num', type=int, default=8, help='number of threads')  # used in cityflow
parser.add_argument('--ngpu', type=str, default="1", help='gpu to be used')  # choose gpu card
parser.add_argument('--prefix', type=str, default='report', help="the number of prefix in this running process")
parser.add_argument('--seed', type=int, default=3047, help="seed for pytorch backend")
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--interface', type=str, default="libsumo", choices=['libsumo', 'traci'],
                    help="interface type")  # libsumo(fast) or traci(slow)
parser.add_argument('--delay_type', type=str, default="apx", choices=['apx', 'real'],
                    help="method of calculating delay")  # apx(approximate) or real
parser.add_argument('--task_id', type=str, default='',  help='the task id for save report')

parser.add_argument('-t', '--task', type=str, default="sim2real", help="task type to run")
parser.add_argument('-a', '--agent', type=str, default="dqn", help="agent type of agents in RL environment")
parser.add_argument('-n', '--network', type=str, default="cityflow1x1", help="network name")
parser.add_argument('-d', '--dataset', type=str, default='onfly', help='type of dataset in training process')

# sumohz4x4

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

logging_level = logging.INFO
if args.debug:
    logging_level = logging.DEBUG


class Runner:
    def __init__(self, pArgs):
        """
        instantiate runner object with processed config and register config into Registry class
        """
        self.config, self.duplicate_config = build_config(pArgs)
        self.task_id = args.task_id
        self.config_registry()

    def config_registry(self):
        """
        Register config into Registry class
        """

        interface.Command_Setting_Interface(self.config)
        interface.Logger_param_Interface(self.config)  # register logger path

        if self.config['model'].get('graphic', False):

            self.config['command']['world'] = 'sumo'
            interface.World_param_Interface(self.config, task_id=self.task_id)

            self.config['command']['world'] = 'cityflow'
            interface.World_param_Interface(self.config, task_id=self.task_id)

        else:
            raise ValueError

            # interface.Graph_World_Interface(roadnet_path)  # register graphic parameters in Registry class
        interface.Logger_path_Interface(self.config, task_id=self.task_id)
        print('saving logging to ', Registry.mapping['logger_mapping']['path'].path)
        # make output dir if not exist
        if not os.path.exists(Registry.mapping['logger_mapping']['path'].path):
            os.makedirs(Registry.mapping['logger_mapping']['path'].path)
        interface.Trainer_param_Interface(self.config)
        interface.ModelAgent_param_Interface(self.config)

    def run(self):
        logger = setup_logging(logging_level)
        self.trainer = Registry.mapping['trainer_mapping'] \
            [Registry.mapping['command_mapping']['setting'].param['task']](logger, self.config)
        self.task = Registry.mapping['task_mapping'] \
            [Registry.mapping['command_mapping']['setting'].param['task']](self.trainer)
        start_time = time.time()
        self.task.run()
        logger.info(f"Total time taken: {time.time() - start_time}")


if __name__ == '__main__':
    test = Runner(args)

    test.run()
