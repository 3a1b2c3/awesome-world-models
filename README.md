# awesome-world-models 2024
[[_TOC_]]

## Yann LeCun's defintion is a bit techical
Lots of confusion about what a world model is. Here is my definition: 
Given:
- an observation x(t)
- a previous estimate of the state of the world s(t)
- an action proposal a(t)
- a latent variable proposal z(t)

A world model computes:
- representation: h(t) = Enc(x(t))
- prediction: s(t+1) = Pred( h(t), s(t), z(t), a(t) )
Where
- Enc() is an encoder (a trainable deterministic function, e.g. a neural net)
- Pred() is a hidden state predictor (also a trainable deterministic function).
- the latent variable z(t) represents the unknown information that would allow us to predict exactly what happens. It must be sampled from a distribution or or varied over a set. It parameterizes the set (or distribution) of plausible predictions.

The trick is to train the entire thing from observation triplets (x(t),a(t),x(t+1)) while preventing the Encoder from collapsing to a trivial solution on which it ignores the input.
Auto-regressive generative models (such as LLMs) are a simplified special case in which
1. the Encoder is the identity function: h(t) = x(t),
2. the state is a window of past inputs 
3. there is no action variable a(t)
4. x(t) is discrete
5. the Predictor computes a distribution over outcomes for x(t+1) and uses the latent z(t) to select one value from that distribution.
The equations reduce to:
s(t) = [x(t),x(t-1),...x(t-k)]
x(t+1) = Pred( s(t), z(t) )
There is no collapse issue in that case.

## Commercial world models, no code or papers
* odyssey https://odyssey.systems/introducing-explorer#waitlist
* worldlabs https://www.worldlabs.ai/blog
* google genie-2 https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/* 
* google veo-2 https://deepmind.google/technologies/veo/veo-2/
* openai sora https://openai.com/sora/ Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond (from GigaAI)

## Playable demos 
* https://next.journee.ai/xyz-diamond
* https://oasis.decart.ai/welcome

## Papers and code
### OpenSora
* *Project:* https://hpcaitech.github.io/Open-Sora/ no windows, 24 GB RAM+
Open-Sora’s methodology revolves around a comprehensive training pipeline incorporating video compression, denoising, and decoding stages to process and generate video content efficiently. Using a video compression network, the model compresses videos into sequences of spatial-temporal patches in latent space, then refined through a Diffusion Transformer for denoising, followed by decoding to produce the final video output. This innovative approach allows for handling various sizes and complexities of videos with improved efficiency and reduced computational demands.
* *Project:* https://github.com/PKU-YuanGroup/Open-Sora-Plan
  
## Navigation World Models
Amir BarGaoyue ZhouDanny TranTrevor DarrellYann LeCun meta
Navigation is a fundamental skill of agents with visual-motor capabilities. We introduce a Navigation World Model (NWM), a controllable video generation model that predicts future visual observations…
* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals.

* *Paper:* https://arxiv.org/pdf/2411.04983

## Learning Generative Interactive Environments by Trained Agent Exploration
Learning Generative Interactive Environments by Trained Agent Exploration

* *Project:* https://github.com/insait-institute/GenieRedux
* *Paper:* https://arxiv.org/pdf/2409.06445

## The Matrix: Infinite-Horizon World Generation with Real-Time Moving Control
Infinite-Horizon World Generation with Real-Time Interaction

* *Project:* https://thematrix1999.github.io/
* *Paper:* https://thematrix1999.github.io/article/the_matrix.pdf
  
## GameGen-X: Interactive Open-world Game Video Generation
We introduce GameGen-X, the first diffusion transformer model specifically designed for both generating and interactively controlling open-world game videos.

* *Paper:* https://arxiv.org/abs/2411.00769

## Diffusion for World Modeling: Visual Details Matter in Atari  DIAMOND
DIAMOND, a diffusion world model, to train sample-efficient RL agents on Atari 100k.
* *Project and code:* https://github.com/eloialonso/diamond
* *Demo:* https://next.journee.ai/xyz-diamond

## Efficient World Models with Context-Aware Tokenization
* *Project:* https://github.com/vmicheli/delta-iris
Scaling up deep Reinforcement Learning (RL) methods presents a significant challenge. Following developments in generative modelling, model-based RL positions itself as a strong contender. 

## AVID: Adapting Video Diffusion Models to World Models
* *Paper:* https://arxiv.org/pdf/2410.12822

Large-scale generative models have achieved remarkable success in a number of domains. However, for sequential decision-making problems, such as robotics, action-labelled data is often scarce and…

## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774
paper demo code

## Genesis is a physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications. It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
* *Project:* https://github.com/Genesis-Embodied-AI/Genesis

## World models 2023 and before
* See https://github.com/PatrickHua/Awesome-World-Models
* https://github.com/GigaAI-research/General-World-Models-Survey

## Basic 
### ViT: Transformers for Image Recognition [Paper] [Blog] [Video]
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
* *Paper:* https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
* *Paper:*  https://arxiv.org/abs/2106.10270
Masked Autoencoders: A PyTorch Implementation https://github.com/facebookresearch/mae/blob/main/models_vit.py


### Dit "Scalable Diffusion Models with Transformers" 
* *Project:* https://github.com/facebookresearch/DiT
* *Paper:* https://arxiv.org/pdf/2212.09748
OpenDiT: An acceleration for DiT training. We adopt valuable acceleration strategies for training progress from OpenDiT.

