  * [Yann LeCun's defintion (a bit techical)](#yann-lecun-s-defintion--a-bit-techical-)
  * [Commercial world models: no code or papers](#commercial-world-models--no-code-or-papers)
  * [Playable demos](#playable-demos)
  * [Papers and code](#papers-and-code)
  * [World models 2023 and before](#world-models-2023-and-before)
  * [Background](#background)


## Yann LeCun's defintion (a bit techical)
For a more accessible definition see "World Models: the AI technology that could displace 3D as we know it
https://substack.com/home/post/p-153106976

Yann LeCun's definition. Given:
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

## Commercial world models: no code or papers
* odyssey https://odyssey.systems/introducing-explorer#waitlist
* worldlabs https://www.worldlabs.ai/blog
* google genie-2 https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/* 
* google veo-2 https://deepmind.google/technologies/veo/veo-2/
* Oasis https://www.decart.ai/articles/oasis-interactive-ai-video-game-model
* openai sora https://openai.com/sora/ Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond (from GigaAI)
  * Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models, **Paper:** https://arxiv.org/pdf/2402.17177

## Playable demos 
* https://next.journee.ai/xyz-diamond Counter strike
* https://oasis.decart.ai/welcome Minecraft

## Papers and code
### DIFFUSION MODELS ARE REAL-TIME GAME ENGINES 
GameNGen, the first game engine powered entirely by a neural model that enables real-time interaction with a complex environment over long trajectories at high quality. 
*Project:*  https://gamengen.github.io

### Hunyuan, OpenSora and Open-Sora-Plan (see also Dit)
Open-Sora’s methodology revolves around a comprehensive training pipeline incorporating video compression, denoising, and decoding stages to process and generate video content efficiently. Using a video compression network, the model compresses videos into sequences of spatial-temporal patches in latent space, then refined through a Diffusion Transformer for denoising, followed by decoding to produce the final video output. This innovative approach allows for handling various sizes and complexities of videos with improved efficiency and reduced computational demands.

* *Project:* https://github.com/PKU-YuanGroup/Open-Sora-Plan
* *Project:* https://hpcaitech.github.io/Open-Sora/ no windows, 24 GB RAM+
* Hunyuan-DiT and video: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding
  *Report:* https://aivideo.hunyuan.tencent.com/hunyuanvideo.pdf
  *Code:* https://github.com/Tencent/HunyuanDiT 
  *Code:* https://github.com/Tencent/HunyuanVideo

## Learning Generative Interactive Environments by Trained Agent Exploration
Proposes to improve the model by employing reinforcement learning based agents for data generation.

* *Project:* https://github.com/insait-institute/GenieRedux
* *Paper:* https://arxiv.org/pdf/2409.06445

## The Matrix: Infinite-Horizon World Generation with Real-Time Moving Control
Infinite-Horizon World Generation with Real-Time Interaction
We present The Matrix, the first foundational realistic world simulator capable of generating continuous 720p high-fidelity real-scene video streams with real-time, responsive control in both first- and third-person perspectives, enabling immersive exploration of richly dynamic environments.

* *Project:* https://thematrix1999.github.io/
* *Paper:* https://thematrix1999.github.io/article/the_matrix.pdf
  
## GameGen-X: Interactive Open-world Game Video Generation
We introduce GameGen-X, the first diffusion transformer model specifically designed for both generating and interactively controlling open-world game videos

* *Paper:* https://arxiv.org/abs/2411.00769

## Diffusion for World Modeling: Visual Details Matter in Atari  DIAMOND
DIAMOND, a diffusion world model, to train sample-efficient RL agents on Atari 100k.
* Minecraft, CSGO

* *Project and code:* https://github.com/eloialonso/diamond
* *Demo:* https://next.journee.ai/xyz-diamond

## Efficient World Models with Context-Aware Tokenization
Δ-IRIS is a reinforcement learning agent trained in the imagination of its world model.
 
* *Project and code:* https://github.com/vmicheli/delta-iris
* *Paper:* https://arxiv.org/abs/2406.19320
  
## AVID: Adapting Video Diffusion Models to World Models
The key idea behind AVID is to adapt the video diffusion model to better understand and represent the real world.

* *Paper:* https://arxiv.org/pdf/2410.12822

## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.


* *Paper:* https://arxiv.org/pdf/2410.10774
## Navigation World Models
Navigation is a fundamental skill of agents with visual-motor capabilities. We introduce a Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983
  
## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## World models 2023 and before
See list below for papers

* https://github.com/PatrickHua/Awesome-World-Models
* https://github.com/GigaAI-research/General-World-Models-Survey

## Background 
### ViT: Transformers for Image Recognition [Paper] [Blog] [Video]
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
* *Paper:* https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
* *Paper:*  https://arxiv.org/abs/2106.10270
Masked Autoencoders: A PyTorch Implementation https://github.com/facebookresearch/mae/blob/main/models_vit.py


### Dit "Scalable Diffusion Models with Transformers" 
* *Project:* https://github.com/facebookresearch/DiT
* *Paper:* https://arxiv.org/pdf/2212.09748
* OpenDiT: An acceleration for DiT training. Acceleration strategies for training progress from OpenDiT.
 *Project and code:* https://oahzxl.github.io/PAB/   
* PixArt: An open-source DiT-based text-to-image model. *Code:* https://github.com/PixArt-alpha/PixArt-alpha
* Latte: An attempt to efficiently train DiT for video. *Code:* [[https://github.com/PixArt-alpha/PixArt-alph](https://github.com/Vchitect/Latte)a](https://github.com/Vchitect/Latte)
