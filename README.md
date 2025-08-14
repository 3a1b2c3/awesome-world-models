# World Models Overview

## Table of Contents
* [Yann LeCun's Definition (a bit technical)](#yann-lecuns-definition-a-bit-technical)
  * [Commercial World Models: No Code or Papers](#commercial-world-models-no-code-or-papers)
  * [Open Source World Models](#open-source-world-models)
  * [Playable Demos](#playable-demos)
  * [Papers and Code](#papers-and-code)
  * [3D World Models](#3d-world-models)
  * [Camera Control](#camera-control)
  * [Humans](#humans)
  * [Mapping](#mapping)
  * [Physics](#physics)
  * [Speed and Benchmarks](#benchmarks)
* [World Models 2023 and Before](#world-models-2023-and-before)
* [Background](#background)

<img src="https://github.com/user-attachments/assets/c65aa56c-4f4e-45b9-a5a6-5be630ed6a7e" alt="World Models Illustration" style="width:400px;"/>

Source: [Hugging Face Video Generation Blog](https://huggingface.co/blog/video_gen)

## Definition  
World Models blend generative AI with reinforcement learning. If a language model compresses all the text on the internet, a world model does the equivalent for video data—compressing the visual world (e.g., YouTube).

For a more accessible explanation, see this [YouTube video](https://www.youtube.com/watch?v=iv-5mZ_9CPY).

### Yann LeCun's Definition (a bit technical)
Given:
- an observation x(t)
- a previous state estimate s(t)
- an action proposal a(t)
- a latent proposal z(t)

A world model computes:
- **Representation**: h(t) = Enc(x(t))
- **Prediction**: s(t+1) = Pred(h(t), s(t), z(t), a(t))

Where:
- **Enc()** is a trainable deterministic encoder (e.g. a neural net).
- **Pred()** is a deterministic state predictor.
- **z(t)** represents unknown factors that allow exact future prediction; it is sampled from a distribution.

The entire system is trained using triplets (x(t), a(t), x(t+1)) while avoiding trivial encoder collapse.

Auto-regressive generative models (like LLMs) simplify this:
1. The encoder is the identity: h(t) = x(t).
2. The state is a window of past inputs.
3. No action a(t) is used.
4. x(t) is discrete.
5. The predictor computes a distribution over x(t+1) and selects a value via z(t).

## Commercial World Models: No Code or Papers
- **Odyssey** – [Explorer](https://odyssey.systems/introducing-explorer#waitlist)
- **WorldLabs** – [Blog](https://www.worldlabs.ai/blog)
- **Google Genie-3** – [DeepMind Blog](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)
- **Google Video Poet** – [Fliki Blog](https://fliki.ai/blog/google-videopoet)
- **Oasis** – [Decart’s Article](https://www.decart.ai/articles/oasis-interactive-ai-video-game-model)
- **OpenAI Sora** – [OpenAI Website](https://openai.com/sora/)  
  - *Survey Paper:* [Sora Review](https://arxiv.org/pdf/2402.17177)

**Provider | Model | Open/Closed | License**  
---|---|---|---  
Google | **Genie 3** | Closed | Proprietary  
NVIDIA | **Cosmoas** | Open | Apache
YUME | **YUME** | Open | Apache 2.0  
Tencent | **Hunyuan Video/World/3d** | Open | Custom  


Source: [Hugging Face Video Generation Blog](https://huggingface.co/blog/video_gen)

## Open Source World Models
- **YUME** – [GitHub Repository](https://github.com/stdstu12/YUME)

## Game/Interactive Models

### Interactive Game Models: Playable Demos
- **Hunyuan-GameCraft** – Interactive game video generation with hybrid history conditioning. [Demo](https://hunyuan-gamecraft.github.io/) | [Paper](https://arxiv.org/abs/2506.17201)
- **AIGame Engine** – AI-Native UGC engine powered by real-time world models. [Blog](https://blog.dynamicslab.ai/)
- **Odyssey World** – Research preview of interactive video generation in real time. [Interactive Video](https://odyssey.world/introducing-interactive-video)
- **XYZ Diamond** – Counter-Strike–like interactive game demo. [Demo](https://next.journee.ai/xyz-diamond)
- **Promptable Game Models** – Text-Guided Game Simulation via Masked Diffusion Models. [Project](https://snap-research.github.io/promptable-game-models/) | [Paper](https://arxiv.org/pdf/2303.13472)
- **Creating New Games with Generative Interactive Videos** – [Paper](https://arxiv.org/pdf/2501.08325)

### Papers and Code

#### Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model
- *Project*: https://matrix-game-v2.github.io/ MIT License

#### GameFactory: Creating New Games with Generative Interactive Videos
- *Project*: https://github.com/KwaiVGI/GameFactory

####  Seaweed APT2** Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation
- *Paper*: https://seaweed-apt.com/2

####  Yan Foundational Interactive Video Generation
- *Paper*:[ https://seaweed-apt.com/2](https://greatx3.github.io/Yan/)
  

#### From Virtual Games to Real-World Play
- [Website](https://wenqsun.github.io/RealPlay/)  
- *Paper*: [https://arxiv.org/abs/2506.18901](https://arxiv.org/abs/2506.18901)

#### Muse for Gameplay Ideation
- *Blog*: [Introducing Muse](https://www.microsoft.com/en-us/research/blog/introducing-muse-our-first-generative-ai-model-designed-for-gameplay-ideation/)

#### GameFactory
- *Paper*: [https://arxiv.org/abs/2501.08325](https://arxiv.org/abs/2501.08325)  
- *Project*: [3D Generalist](https://3d-generalist.pages.dev/)

#### GameNGen
- *Demo*: [Doom Demo](https://gamengen.github.io)

#### Additional References
- **Cosmos and Genie-3** (Google)
- **DynamiCrafter**: Open-domain video diffusion.
- **Open-Sora**: Survey and open-source training pipelines.
- **Hunyuan-DiT / HunyuanVideo**: Advanced models with 3D capabilities.

## Camera Control
- **Cavia**: Camera-controllable multi-view video diffusion with view-integrated attention.  
  *Paper*: [https://arxiv.org/pdf/2410.10774](https://arxiv.org/pdf/2410.10774)
- **ShotAdapter**: Text-to-multi-shot video generation with diffusion models.  
  [Demo](https://shotadapter.github.io/)

## 3D/4D World Creation
- **DimensionX**: Generate any 3D/4D scene from a single image using controllable video diffusion.  
  *Code*: [GitHub](https://github.com/wenqsun/DimensionX) | *Paper*: [https://arxiv.org/pdf/2411.04928](https://arxiv.org/pdf/2411.04928)
- **HunyuanWorld 3D**: Playable 3D worlds from text or images.  
  *Code*: [GitHub](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) | *Paper*: [https://arxiv.org/pdf/2507.21809](https://arxiv.org/pdf/2507.21809)
- **WorldGen**: Rapid generation of any 3D scene.  
  *Code*: [GitHub](https://github.com/ZiYang-xie/)
- **WonderJourney**: Generate connected 3D scene journeys from user input.  
  *Paper*: [https://arxiv.org/pdf/2312.03884](https://arxiv.org/pdf/2312.03884)
- **3D Generalist**: Vision-language-action models for 3D world creation.  
  [Project](https://3d-generalist.pages.dev/)
- **World-Consistent Video Diffusion**: With explicit 3D modeling.  
  [Project](https://zqh0253.github.io/wvd/)
- **Matrix3D**: All-in-one large photogrammetry model.  
  *Paper*: [https://arxiv.org/pdf/2502.07685](https://arxiv.org/pdf/2502.07685)
- **Domain-Free Generation of 3D Gaussian Splatting Scenes**  
  *Code*: [Website](https://luciddreamer-cvlab.github.io/) | *Paper*: [https://arxiv.org/abs/2311.13384](https://arxiv.org/abs/2311.13384)

## Humans
- **HumanDiT**  
  [Project](https://agnjason.github.io/HumanDiT-page/)

## Mapping
- **Mars**: Controllable video synthesis with accurate 3D reconstruction.  
  [Demo](https://marsgenai.github.io/)
- **GPS as a Control Signal for Image Generation**  
  [Demo](https://cfeng16.github.io/gps-gen/)
- **Streetscapes**: Consistent street view generation using autoregressive video diffusion.  
  *Paper*: [https://arxiv.org/abs/2407.13759](https://arxiv.org/abs/2407.13759)

## Physics
- **V-JEPA 2**: World model and physical reasoning benchmark.  
  [Blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- **Genesis**: Universal physics simulation platform for embodied/physical AI.  
  [Project & Code](https://github.com/Genesis-Embodied-AI/Genesis)

## Speed and benchmarks
### Faster and cheaper and world models
#### Objects matter: object-centric world models improve reinforcement learning in visually complex environments
  *Paper*: https://arxiv.org/pdf/2501.16443

#### AVID: Adapting Video Diffusion Models to World Models
  *Paper*: https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_64.pdf


### Benchmarks
- **PBench**: Physical AI benchmark for world models.  
  [Details](https://research.nvidia.com/labs/dir/pbench/)
- **E3D-Bench**: Benchmark for 3D geometric foundation models.  
  [Website](https://e3dbench.github.io/)
- **VBench-2.9**: Video generation benchmark suite for measuring intrinsic faithfulness.
- **Physics IQ Benchmark**: Assesses physical reasoning in generated videos.  
  *Paper*: [Google Drive](https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view) | [Project](https://physics-iq.github.io/)


## Background 
### World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

* https://github.com/PatrickHua/Awesome-World-Models
* https://github.com/GigaAI-research/General-World-Models-Survey


### Temporal consistency
- **FramePack-P1** – [FramePack-P1]
 * *Project:*  (https://lllyasviel.github.io/frame_pack_gitpage/p1/)
Planned Anti‑Drifting
Instead of generating video sections strictly in chronological order, FramePack‑P1 predicts distant sections first, then fills in the in-between. This “planning ahead” approach mitigates drift that tends to occur between generated segments, maintaining consistency across the video 

History Discretization
FramePack‑P1 transforms past frames into discrete token representations—using methods like K-means clustering—to create a more uniform, stable historical context. This helps reduce drift over the planned endpoints and ensures consistency between training and inference contexts


- **Context as Memory: Scene-Consistent Interactive Long Video**
Generation with Memory Retrieval
 * *Paper:*  https://arxiv.org/pdf/2506.03141


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
* *Paper:* https://arxiv.org/pdf/2410.12822
