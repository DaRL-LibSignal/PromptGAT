
# Prompt learning for Sim2Real under reinforcement learning

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
