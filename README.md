
# Prompt learning for Sim2Real under reinforcement learning

## Description: 
 This repo is the code implementation of AAAI 2024 paper: 
 `Prompt to Transfer: Sim-to-Real Transfer for Traffic Signal Control with Prompt Learning`  
 Or another version: 
 `LLM Powered Sim-to-real Transfer for Traffic Signal Control`
 Before publish proceedings complete, You can find on an arXiv version here:
 https://arxiv.org/pdf/2308.14284.pdf


 This Project is built opon the code of paper: "Uncertainty-aware Grounded Action Transformation towards Sim-to-Real Transfer for Traffic Signal Control" But annotated the uncertainty module to not working.

 The overall project is built based on LibSignal: https://darl-libsignal.github.io/

 ## Instruction:
 Please fill your Open-AI api-key in the file `sim2real_trainer.py` to make sure the conversation to language agent is successfully connected!

 Please make sure install the `requirements.txt` file before execution.

 To execute the code, please execute the `sim2real.py` directly in the root folder.


 ## Citation
If you find this work helpful, please cite us:
```
@article{da2023llm,
  title={Llm powered sim-to-real transfer for traffic signal control},
  author={Da, Longchao and Gao, Minchiuan and Mei, Hao and Wei, Hua},
  journal={arXiv preprint arXiv:2308.14284},
  year={2023}
}
```
