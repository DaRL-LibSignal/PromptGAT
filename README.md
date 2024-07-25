
# Prompt learning for Sim2Real under reinforcement learning

## 🚀 🚀 🚀
## We have created a docker image for your convenience 
## <span style="color:red">(Start sim-to-real for TSC by single line)!</span>


This docker code base is from a sister paper of sim2real tsc, you can simply replace the code with current repository: 

> Uncertainty-aware Grounded Action Transformation towards Sim-to-Real Transfer for Traffic Signal Control (https://github.com/darl-libsignal/ugat)

`docker pull danielda1/ugat:latest`

`docker run -it --name ugat_case danielda1/ugat:latest`

`cd /DaRL/UGAT_Docker/`

`python sim2real.py`

## At the same time of using this Docker Image, you have the the readily prepared LibSignal
This is a multi-simulator supported framework, provide easy-to-configure settings for sim-to-sim simulated sim-to-real training and testing.
For details, please visit: https://darl-libsignal.github.io/



## Description: 
 This repo is the code implementation of AAAI 2024 paper: 
 `Prompt to Transfer: Sim-to-Real Transfer for Traffic Signal Control with Prompt Learning`  
 Or another version: 
 `LLM Powered Sim-to-real Transfer for Traffic Signal Control`
 Before publish proceedings complete, You can find on an arXiv version here:
 https://arxiv.org/pdf/2308.14284.pdf
 
 An illustration of our method (PromptGAT) compared to Vanilla GAT:    
 ![Illustration](/assets/image2.png "Demonstration of our method compared to Vanilla GAT")
 
 Detailed Structure of PromptGAT (for more introduction, please refer to our paper above):  
 ![Illustration](/assets/demo_image.png "Detailed Structure of PromptGAT")
 
 This Project is built opon the code of paper: "Uncertainty-aware Grounded Action Transformation towards Sim-to-Real Transfer for Traffic Signal Control" But annotated the uncertainty module to not working.

 The overall project is built based on LibSignal: https://darl-libsignal.github.io/

 ## Instruction:
 Please fill your Open-AI api-key in the file `sim2real_trainer.py` to make sure the conversation to language agent is successfully connected!

 Please make sure install the `requirements.txt` file before execution.

 To execute the code, please execute the `sim2real.py` directly in the root folder.


 ## Citation
If you find this work helpful, please cite us:
```
@inproceedings{da2024prompt,
  title={Prompt to Transfer: Sim-to-Real Transfer for Traffic Signal Control with Prompt Learning},
  author={Da, Longchao and Gao, Minquan and Mei, Hao and Wei, Hua},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={82--90},
  year={2024}
}
```
