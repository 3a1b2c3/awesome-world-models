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
  * [Benchmarks](#benchmarks)
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
- **Google Veo** – [Veo 2](https://deepmind.google/models/veo/)  
  - [Veo-3 Report](https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf)
- **Pika** – [Pika Website](https://pika.art/)
- **Luma AI** – [Dream Machine](https://lumalabs.ai/dream-machine)
- **Oasis** – [Decart’s Article](https://www.decart.ai/articles/oasis-interactive-ai-video-game-model)
- **OpenAI Sora** – [OpenAI Website](https://openai.com/sora/)  
  - *Survey Paper:* [Sora Review](https://arxiv.org/pdf/2402.17177)

**Provider | Model | Open/Closed | License**  
---|---|---|---  
Meta | **MovieGen** | Closed | Proprietary  
OpenAI | **Sora** | Closed | Proprietary  
Google | **Veo 2** | Closed | Proprietary  
RunwayML | **Gen 3 (Alpha)** | Closed | Proprietary  
Pika Labs | **Pika 2.0** | Closed | Proprietary  
KlingAI | **Kling** | Closed | Proprietary  
Haliluo | **MiniMax** | Closed | Proprietary  
THUDM | **CogVideoX** | Open | Custom  
Genmo | **Mochi-1** | Open | Apache 2.0  
RhymesAI | **Allegro** | Open | Apache 2.0  
Lightricks | **LTX Video** | Open | Custom  
Tencent | **Hunyuan Video** | Open | Custom  

Source: [Hugging Face Video Generation Blog](https://huggingface.co/blog/video_gen)

## Open Source World Models
- **YUME** – [GitHub Repository](https://github.com/stdstu12/YUME)

## World Models 2023 and Before
*Overview of earlier models and their evolution.*


### World Models in Games
- **Hunyuan-GameCraft**: Interactive game video generation with hybrid history conditioning.  
  [Demo](https://hunyuan-gamecraft.github.io/) | [Paper](https://arxiv.org/abs/2506.17201)
- **AIGame Engine**: AI-Native UGC engine powered by real-time world models.  
  [Blog](https://blog.dynamicslab.ai/)
- **Odyssey World**: A research preview of interactive video generation in real time.  
  [Interactive Video](https://odyssey.world/introducing-interactive-video)
- **XYZ Diamond**: Counter-Strike-like interactive game demo.  
  [Demo](https://next.journee.ai/xyz-diamond)

### Papers and Code
#### LTX Video
- *Paper*: [https://arxiv.org/abs/2501.00103](https://arxiv.org/abs/2501.00103)  
- *Code*: [GitHub Repository](https://github.com/Lightricks/LTX-Video)

#### Hunyuan-GameCraft 
- [Demo](https://hunyuan-gamecraft.github.io/)  
- *Paper*: [https://arxiv.org/abs/2506.17201](https://arxiv.org/abs/2506.17201)

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

## Benchmarks
- **PBench**: Physical AI benchmark for world models.  
  [Details](https://research.nvidia.com/labs/dir/pbench/)
- **E3D-Bench**: Benchmark for 3D geometric foundation models.  
  [Website](https://e3dbench.github.io/)
- **VBench-2.9**: Video generation benchmark suite for measuring intrinsic faithfulness.
- **Physics IQ Benchmark**: Assesses physical reasoning in generated videos.  
  *Paper*: [Google Drive](https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view) | [Project](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

* https://github.com/PatrickHua/Awesome-World-Models
* https://github.com/GigaAI-research/General-World-Models-Survey




## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location (by a text description or an
image), and generate a journey through a long sequence of
diverse yet coherently connected 3D scene

* *Paper:*  https://arxiv.org/pdf/2312.03884
  
## 3d Generalist: Vision-Language-Action Models for Crafting 3D Worlds
*Project:*  https://3d-generalist.pages.dev/

## 3World-consistent Video Diffusion with Explicit 3D Modeling 
*Project:* https://zqh0253.github.io/wvd/

## Matrix3D: Large Photogrammetry Model All-in-One
* *Paper:* https://arxiv.org/pdf/2502.07685

## Domain-free generation of 3d gaussian splatting scenes
* *Code:*  https://luciddreamer-cvlab.github.io/
* *Paper:* https://arxiv.org/abs/2311.13384

# Humans
## HumanDiT
* *Project:* https://agnjason.github.io/HumanDiT-page/

# Mapping
## Mars
Martian World Models : Controllable Video Synthesis with Physically Accurate 3D Reconstructions
https://marsgenai.github.io/

## GPS as a Control Signal for Image Generation
https://cfeng16.github.io/gps-gen/

## Streetscapes: Large-scale Consistent Street View Generation using Autoregressive Video Diffusion
underlying map/layout hosting the desired trajectory.
* *Paper:* https://arxiv.org/abs/2407.13759


# Physics
## v-jepa-2
V-JEPA 2 world model and new benchmarks for physical reasoning
https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## Benchmarks:
### PBench: A Physical AI Benchmark for World Models
We measure two scores in PBench: Domain Score and Quality Score. The Domain Score measures the domain-specific capabilities of the world model via the QA pairs, and the Quality Score measures the quality of the generated video. Given a generated video from a world model and the corresponding QA pairs in PBench, we employ a VLM (Qwen2.5-VL-72B-Instruct) as a judge to measure the generated video by calculating the accuracy as the domain score on all the QA pairs for each sample.
* https://research.nvidia.com/labs/dir/pbench/

### E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
https://e3dbench.github.io/

### VBench-2.9
Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness:
https://huggingface.co/papers/2503.21755

### Physics IQ Benchmark:
Do generative video models learn physical principles from watching videos?
* *Paper:*  https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view
* *Project and code:* [[https://github.com/Genesis-Embodied-AI/Genesis](https://physics-iq.github.io/)](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

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
* *Paper:* https://arxiv.org/pdf/2410.12822


## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location (by a text description or an
image), and generate a journey through a long sequence of
diverse yet coherently connected 3D scene

* *Paper:*  https://arxiv.org/pdf/2312.03884
  
## 3d Generalist: Vision-Language-Action Models for Crafting 3D Worlds
*Project:*  https://3d-generalist.pages.dev/

## 3World-consistent Video Diffusion with Explicit 3D Modeling 
*Project:* https://zqh0253.github.io/wvd/

## Matrix3D: Large Photogrammetry Model All-in-One
* *Paper:* https://arxiv.org/pdf/2502.07685

## Domain-free generation of 3d gaussian splatting scenes
* *Code:*  https://luciddreamer-cvlab.github.io/
* *Paper:* https://arxiv.org/abs/2311.13384

# Humans
## HumanDiT
* *Project:* https://agnjason.github.io/HumanDiT-page/

# Mapping
## Mars
Martian World Models : Controllable Video Synthesis with Physically Accurate 3D Reconstructions
https://marsgenai.github.io/

## GPS as a Control Signal for Image Generation
https://cfeng16.github.io/gps-gen/

## Streetscapes: Large-scale Consistent Street View Generation using Autoregressive Video Diffusion
underlying map/layout hosting the desired trajectory.
* *Paper:* https://arxiv.org/abs/2407.13759


# Physics
## v-jepa-2
V-JEPA 2 world model and new benchmarks for physical reasoning
https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## Benchmarks:
### PBench: A Physical AI Benchmark for World Models
We measure two scores in PBench: Domain Score and Quality Score. The Domain Score measures the domain-specific capabilities of the world model via the QA pairs, and the Quality Score measures the quality of the generated video. Given a generated video from a world model and the corresponding QA pairs in PBench, we employ a VLM (Qwen2.5-VL-72B-Instruct) as a judge to measure the generated video by calculating the accuracy as the domain score on all the QA pairs for each sample.
* https://research.nvidia.com/labs/dir/pbench/

### E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
https://e3dbench.github.io/

### VBench-2.9
Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness:
https://huggingface.co/papers/2503.21755

### Physics IQ Benchmark:
Do generative video models learn physical principles from watching videos?
* *Paper:*  https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view
* *Project and code:* [[https://github.com/Genesis-Embodied-AI/Genesis](https://physics-iq.github.io/)](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

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
* *Paper:* https://arxiv.org/pdf/2410.12822


## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location (by a text description or an
image), and generate a journey through a long sequence of
diverse yet coherently connected 3D scene

* *Paper:*  https://arxiv.org/pdf/2312.03884
  
## 3d Generalist: Vision-Language-Action Models for Crafting 3D Worlds
*Project:*  https://3d-generalist.pages.dev/

## 3World-consistent Video Diffusion with Explicit 3D Modeling 
*Project:* https://zqh0253.github.io/wvd/

## Matrix3D: Large Photogrammetry Model All-in-One
* *Paper:* https://arxiv.org/pdf/2502.07685

## Domain-free generation of 3d gaussian splatting scenes
* *Code:*  https://luciddreamer-cvlab.github.io/
* *Paper:* https://arxiv.org/abs/2311.13384

# Humans
## HumanDiT
* *Project:* https://agnjason.github.io/HumanDiT-page/

# Mapping
## Mars
Martian World Models : Controllable Video Synthesis with Physically Accurate 3D Reconstructions
https://marsgenai.github.io/

## GPS as a Control Signal for Image Generation
https://cfeng16.github.io/gps-gen/

## Streetscapes: Large-scale Consistent Street View Generation using Autoregressive Video Diffusion
underlying map/layout hosting the desired trajectory.
* *Paper:* https://arxiv.org/abs/2407.13759


# Physics
## v-jepa-2
V-JEPA 2 world model and new benchmarks for physical reasoning
https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## Benchmarks:
### PBench: A Physical AI Benchmark for World Models
We measure two scores in PBench: Domain Score and Quality Score. The Domain Score measures the domain-specific capabilities of the world model via the QA pairs, and the Quality Score measures the quality of the generated video. Given a generated video from a world model and the corresponding QA pairs in PBench, we employ a VLM (Qwen2.5-VL-72B-Instruct) as a judge to measure the generated video by calculating the accuracy as the domain score on all the QA pairs for each sample.
* https://research.nvidia.com/labs/dir/pbench/

### E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
https://e3dbench.github.io/

### VBench-2.9
Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness:
https://huggingface.co/papers/2503.21755

### Physics IQ Benchmark:
Do generative video models learn physical principles from watching videos?
* *Paper:*  https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view
* *Project and code:* [[https://github.com/Genesis-Embodied-AI/Genesis](https://physics-iq.github.io/)](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

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
* *Paper:* https://arxiv.org/pdf/2410.12822


## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location (by a text description or an
image), and generate a journey through a long sequence of
diverse yet coherently connected 3D scene

* *Paper:*  https://arxiv.org/pdf/2312.03884
  
## 3d Generalist: Vision-Language-Action Models for Crafting 3D Worlds
*Project:*  https://3d-generalist.pages.dev/

## 3World-consistent Video Diffusion with Explicit 3D Modeling 
*Project:* https://zqh0253.github.io/wvd/

## Matrix3D: Large Photogrammetry Model All-in-One
* *Paper:* https://arxiv.org/pdf/2502.07685

## Domain-free generation of 3d gaussian splatting scenes
* *Code:*  https://luciddreamer-cvlab.github.io/
* *Paper:* https://arxiv.org/abs/2311.13384

# Humans
## HumanDiT
* *Project:* https://agnjason.github.io/HumanDiT-page/

# Mapping
## Mars
Martian World Models : Controllable Video Synthesis with Physically Accurate 3D Reconstructions
https://marsgenai.github.io/

## GPS as a Control Signal for Image Generation
https://cfeng16.github.io/gps-gen/

## Streetscapes: Large-scale Consistent Street View Generation using Autoregressive Video Diffusion
underlying map/layout hosting the desired trajectory.
* *Paper:* https://arxiv.org/abs/2407.13759


# Physics
## v-jepa-2
V-JEPA 2 world model and new benchmarks for physical reasoning
https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## Benchmarks:
### PBench: A Physical AI Benchmark for World Models
We measure two scores in PBench: Domain Score and Quality Score. The Domain Score measures the domain-specific capabilities of the world model via the QA pairs, and the Quality Score measures the quality of the generated video. Given a generated video from a world model and the corresponding QA pairs in PBench, we employ a VLM (Qwen2.5-VL-72B-Instruct) as a judge to measure the generated video by calculating the accuracy as the domain score on all the QA pairs for each sample.
* https://research.nvidia.com/labs/dir/pbench/

### E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
https://e3dbench.github.io/

### VBench-2.9
Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness:
https://huggingface.co/papers/2503.21755

### Physics IQ Benchmark:
Do generative video models learn physical principles from watching videos?
* *Paper:*  https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view
* *Project and code:* [[https://github.com/Genesis-Embodied-AI/Genesis](https://physics-iq.github.io/)](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

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
* *Paper:* https://arxiv.org/pdf/2410.12822


## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location (by a text description or an
image), and generate a journey through a long sequence of
diverse yet coherently connected 3D scene

* *Paper:*  https://arxiv.org/pdf/2312.03884
  
## 3d Generalist: Vision-Language-Action Models for Crafting 3D Worlds
*Project:*  https://3d-generalist.pages.dev/

## 3World-consistent Video Diffusion with Explicit 3D Modeling 
*Project:* https://zqh0253.github.io/wvd/

## Matrix3D: Large Photogrammetry Model All-in-One
* *Paper:* https://arxiv.org/pdf/2502.07685

## Domain-free generation of 3d gaussian splatting scenes
* *Code:*  https://luciddreamer-cvlab.github.io/
* *Paper:* https://arxiv.org/abs/2311.13384

# Humans
## HumanDiT
* *Project:* https://agnjason.github.io/HumanDiT-page/

# Mapping
## Mars
Martian World Models : Controllable Video Synthesis with Physically Accurate 3D Reconstructions
https://marsgenai.github.io/

## GPS as a Control Signal for Image Generation
https://cfeng16.github.io/gps-gen/

## Streetscapes: Large-scale Consistent Street View Generation using Autoregressive Video Diffusion
underlying map/layout hosting the desired trajectory.
* *Paper:* https://arxiv.org/abs/2407.13759


# Physics
## v-jepa-2
V-JEPA 2 world model and new benchmarks for physical reasoning
https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## Benchmarks:
### PBench: A Physical AI Benchmark for World Models
We measure two scores in PBench: Domain Score and Quality Score. The Domain Score measures the domain-specific capabilities of the world model via the QA pairs, and the Quality Score measures the quality of the generated video. Given a generated video from a world model and the corresponding QA pairs in PBench, we employ a VLM (Qwen2.5-VL-72B-Instruct) as a judge to measure the generated video by calculating the accuracy as the domain score on all the QA pairs for each sample.
* https://research.nvidia.com/labs/dir/pbench/

### E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
https://e3dbench.github.io/

### VBench-2.9
Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness:
https://huggingface.co/papers/2503.21755

### Physics IQ Benchmark:
Do generative video models learn physical principles from watching videos?
* *Paper:*  https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view
* *Project and code:* [[https://github.com/Genesis-Embodied-AI/Genesis](https://physics-iq.github.io/)](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

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
* *Paper:* https://arxiv.org/pdf/2410.12822


## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location (by a text description or an
image), and generate a journey through a long sequence of
diverse yet coherently connected 3D scene

* *Paper:*  https://arxiv.org/pdf/2312.03884
  
## 3d Generalist: Vision-Language-Action Models for Crafting 3D Worlds
*Project:*  https://3d-generalist.pages.dev/

## 3World-consistent Video Diffusion with Explicit 3D Modeling 
*Project:* https://zqh0253.github.io/wvd/

## Matrix3D: Large Photogrammetry Model All-in-One
* *Paper:* https://arxiv.org/pdf/2502.07685

## Domain-free generation of 3d gaussian splatting scenes
* *Code:*  https://luciddreamer-cvlab.github.io/
* *Paper:* https://arxiv.org/abs/2311.13384

# Humans
## HumanDiT
* *Project:* https://agnjason.github.io/HumanDiT-page/

# Mapping
## Mars
Martian World Models : Controllable Video Synthesis with Physically Accurate 3D Reconstructions
https://marsgenai.github.io/

## GPS as a Control Signal for Image Generation
https://cfeng16.github.io/gps-gen/

## Streetscapes: Large-scale Consistent Street View Generation using Autoregressive Video Diffusion
underlying map/layout hosting the desired trajectory.
* *Paper:* https://arxiv.org/abs/2407.13759


# Physics
## v-jepa-2
V-JEPA 2 world model and new benchmarks for physical reasoning
https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## Benchmarks:
### PBench: A Physical AI Benchmark for World Models
We measure two scores in PBench: Domain Score and Quality Score. The Domain Score measures the domain-specific capabilities of the world model via the QA pairs, and the Quality Score measures the quality of the generated video. Given a generated video from a world model and the corresponding QA pairs in PBench, we employ a VLM (Qwen2.5-VL-72B-Instruct) as a judge to measure the generated video by calculating the accuracy as the domain score on all the QA pairs for each sample.
* https://research.nvidia.com/labs/dir/pbench/

### E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
https://e3dbench.github.io/

### VBench-2.9
Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness:
https://huggingface.co/papers/2503.21755

### Physics IQ Benchmark:
Do generative video models learn physical principles from watching videos?
* *Paper:*  https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view
* *Project and code:* [[https://github.com/Genesis-Embodied-AI/Genesis](https://physics-iq.github.io/)](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

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
* *Paper:* https://arxiv.org/pdf/2410.12822


## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location (by a text description or an
image), and generate a journey through a long sequence of
diverse yet coherently connected 3D scene

* *Paper:*  https://arxiv.org/pdf/2312.03884
  
## 3d Generalist: Vision-Language-Action Models for Crafting 3D Worlds
*Project:*  https://3d-generalist.pages.dev/

## 3World-consistent Video Diffusion with Explicit 3D Modeling 
*Project:* https://zqh0253.github.io/wvd/

## Matrix3D: Large Photogrammetry Model All-in-One
* *Paper:* https://arxiv.org/pdf/2502.07685

## Domain-free generation of 3d gaussian splatting scenes
* *Code:*  https://luciddreamer-cvlab.github.io/
* *Paper:* https://arxiv.org/abs/2311.13384

# Humans
## HumanDiT
* *Project:* https://agnjason.github.io/HumanDiT-page/

# Mapping
## Mars
Martian World Models : Controllable Video Synthesis with Physically Accurate 3D Reconstructions
https://marsgenai.github.io/

## GPS as a Control Signal for Image Generation
https://cfeng16.github.io/gps-gen/

## Streetscapes: Large-scale Consistent Street View Generation using Autoregressive Video Diffusion
underlying map/layout hosting the desired trajectory.
* *Paper:* https://arxiv.org/abs/2407.13759


# Physics
## v-jepa-2
V-JEPA 2 world model and new benchmarks for physical reasoning
https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## Benchmarks:
### PBench: A Physical AI Benchmark for World Models
We measure two scores in PBench: Domain Score and Quality Score. The Domain Score measures the domain-specific capabilities of the world model via the QA pairs, and the Quality Score measures the quality of the generated video. Given a generated video from a world model and the corresponding QA pairs in PBench, we employ a VLM (Qwen2.5-VL-72B-Instruct) as a judge to measure the generated video by calculating the accuracy as the domain score on all the QA pairs for each sample.
* https://research.nvidia.com/labs/dir/pbench/

### E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
https://e3dbench.github.io/

### VBench-2.9
Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness:
https://huggingface.co/papers/2503.21755

### Physics IQ Benchmark:
Do generative video models learn physical principles from watching videos?
* *Paper:*  https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view
* *Project and code:* [[https://github.com/Genesis-Embodied-AI/Genesis](https://physics-iq.github.io/)](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

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
* *Paper:* https://arxiv.org/pdf/2410.12822


## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location (by a text description or an
image), and generate a journey through a long sequence of
diverse yet coherently connected 3D scene

* *Paper:*  https://arxiv.org/pdf/2312.03884
  
## 3d Generalist: Vision-Language-Action Models for Crafting 3D Worlds
*Project:*  https://3d-generalist.pages.dev/

## 3World-consistent Video Diffusion with Explicit 3D Modeling 
*Project:* https://zqh0253.github.io/wvd/

## Matrix3D: Large Photogrammetry Model All-in-One
* *Paper:* https://arxiv.org/pdf/2502.07685

## Domain-free generation of 3d gaussian splatting scenes
* *Code:*  https://luciddreamer-cvlab.github.io/
* *Paper:* https://arxiv.org/abs/2311.13384

# Humans
## HumanDiT
* *Project:* https://agnjason.github.io/HumanDiT-page/

# Mapping
## Mars
Martian World Models : Controllable Video Synthesis with Physically Accurate 3D Reconstructions
https://marsgenai.github.io/

## GPS as a Control Signal for Image Generation
https://cfeng16.github.io/gps-gen/

## Streetscapes: Large-scale Consistent Street View Generation using Autoregressive Video Diffusion
underlying map/layout hosting the desired trajectory.
* *Paper:* https://arxiv.org/abs/2407.13759


# Physics
## v-jepa-2
V-JEPA 2 world model and new benchmarks for physical reasoning
https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## Benchmarks:
### PBench: A Physical AI Benchmark for World Models
We measure two scores in PBench: Domain Score and Quality Score. The Domain Score measures the domain-specific capabilities of the world model via the QA pairs, and the Quality Score measures the quality of the generated video. Given a generated video from a world model and the corresponding QA pairs in PBench, we employ a VLM (Qwen2.5-VL-72B-Instruct) as a judge to measure the generated video by calculating the accuracy as the domain score on all the QA pairs for each sample.
* https://research.nvidia.com/labs/dir/pbench/

### E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
https://e3dbench.github.io/

### VBench-2.9
Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness:
https://huggingface.co/papers/2503.21755

### Physics IQ Benchmark:
Do generative video models learn physical principles from watching videos?
* *Paper:*  https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view
* *Project and code:* [[https://github.com/Genesis-Embodied-AI/Genesis](https://physics-iq.github.io/)](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

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
* *Paper:* https://arxiv.org/pdf/2410.12822


## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location (by a text description or an
image), and generate a journey through a long sequence of
diverse yet coherently connected 3D scene

* *Paper:*  https://arxiv.org/pdf/2312.03884
  
## 3d Generalist: Vision-Language-Action Models for Crafting 3D Worlds
*Project:*  https://3d-generalist.pages.dev/

## 3World-consistent Video Diffusion with Explicit 3D Modeling 
*Project:* https://zqh0253.github.io/wvd/

## Matrix3D: Large Photogrammetry Model All-in-One
* *Paper:* https://arxiv.org/pdf/2502.07685

## Domain-free generation of 3d gaussian splatting scenes
* *Code:*  https://luciddreamer-cvlab.github.io/
* *Paper:* https://arxiv.org/abs/2311.13384

# Humans
## HumanDiT
* *Project:* https://agnjason.github.io/HumanDiT-page/

# Mapping
## Mars
Martian World Models : Controllable Video Synthesis with Physically Accurate 3D Reconstructions
https://marsgenai.github.io/

## GPS as a Control Signal for Image Generation
https://cfeng16.github.io/gps-gen/

## Streetscapes: Large-scale Consistent Street View Generation using Autoregressive Video Diffusion
underlying map/layout hosting the desired trajectory.
* *Paper:* https://arxiv.org/abs/2407.13759


# Physics
## v-jepa-2
V-JEPA 2 world model and new benchmarks for physical reasoning
https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## Benchmarks:
### PBench: A Physical AI Benchmark for World Models
We measure two scores in PBench: Domain Score and Quality Score. The Domain Score measures the domain-specific capabilities of the world model via the QA pairs, and the Quality Score measures the quality of the generated video. Given a generated video from a world model and the corresponding QA pairs in PBench, we employ a VLM (Qwen2.5-VL-72B-Instruct) as a judge to measure the generated video by calculating the accuracy as the domain score on all the QA pairs for each sample.
* https://research.nvidia.com/labs/dir/pbench/

### E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
https://e3dbench.github.io/

### VBench-2.9
Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness:
https://huggingface.co/papers/2503.21755

### Physics IQ Benchmark:
Do generative video models learn physical principles from watching videos?
* *Paper:*  https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view
* *Project and code:* [[https://github.com/Genesis-Embodied-AI/Genesis](https://physics-iq.github.io/)](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

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
* *Paper:* https://arxiv.org/pdf/2410.12822


## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location (by a text description or an
image), and generate a journey through a long sequence of
diverse yet coherently connected 3D scene

* *Paper:*  https://arxiv.org/pdf/2312.03884
  
## 3d Generalist: Vision-Language-Action Models for Crafting 3D Worlds
*Project:*  https://3d-generalist.pages.dev/

## 3World-consistent Video Diffusion with Explicit 3D Modeling 
*Project:* https://zqh0253.github.io/wvd/

## Matrix3D: Large Photogrammetry Model All-in-One
* *Paper:* https://arxiv.org/pdf/2502.07685

## Domain-free generation of 3d gaussian splatting scenes
* *Code:*  https://luciddreamer-cvlab.github.io/
* *Paper:* https://arxiv.org/abs/2311.13384

# Humans
## HumanDiT
* *Project:* https://agnjason.github.io/HumanDiT-page/

# Mapping
## Mars
Martian World Models : Controllable Video Synthesis with Physically Accurate 3D Reconstructions
https://marsgenai.github.io/

## GPS as a Control Signal for Image Generation
https://cfeng16.github.io/gps-gen/

## Streetscapes: Large-scale Consistent Street View Generation using Autoregressive Video Diffusion
underlying map/layout hosting the desired trajectory.
* *Paper:* https://arxiv.org/abs/2407.13759


# Physics
## v-jepa-2
V-JEPA 2 world model and new benchmarks for physical reasoning
https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## Benchmarks:
### PBench: A Physical AI Benchmark for World Models
We measure two scores in PBench: Domain Score and Quality Score. The Domain Score measures the domain-specific capabilities of the world model via the QA pairs, and the Quality Score measures the quality of the generated video. Given a generated video from a world model and the corresponding QA pairs in PBench, we employ a VLM (Qwen2.5-VL-72B-Instruct) as a judge to measure the generated video by calculating the accuracy as the domain score on all the QA pairs for each sample.
* https://research.nvidia.com/labs/dir/pbench/

### E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
https://e3dbench.github.io/

### VBench-2.9
Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness:
https://huggingface.co/papers/2503.21755

### Physics IQ Benchmark:
Do generative video models learn physical principles from watching videos?
* *Paper:*  https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view
* *Project and code:* [[https://github.com/Genesis-Embodied-AI/Genesis](https://physics-iq.github.io/)](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

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
* *Paper:* https://arxiv.org/pdf/2410.12822


## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location (by a text description or an
image), and generate a journey through a long sequence of
diverse yet coherently connected 3D scene

* *Paper:*  https://arxiv.org/pdf/2312.03884
  
## 3d Generalist: Vision-Language-Action Models for Crafting 3D Worlds
*Project:*  https://3d-generalist.pages.dev/

## 3World-consistent Video Diffusion with Explicit 3D Modeling 
*Project:* https://zqh0253.github.io/wvd/

## Matrix3D: Large Photogrammetry Model All-in-One
* *Paper:* https://arxiv.org/pdf/2502.07685

## Domain-free generation of 3d gaussian splatting scenes
* *Code:*  https://luciddreamer-cvlab.github.io/
* *Paper:* https://arxiv.org/abs/2311.13384

# Humans
## HumanDiT
* *Project:* https://agnjason.github.io/HumanDiT-page/

# Mapping
## Mars
Martian World Models : Controllable Video Synthesis with Physically Accurate 3D Reconstructions
https://marsgenai.github.io/

## GPS as a Control Signal for Image Generation
https://cfeng16.github.io/gps-gen/

## Streetscapes: Large-scale Consistent Street View Generation using Autoregressive Video Diffusion
underlying map/layout hosting the desired trajectory.
* *Paper:* https://arxiv.org/abs/2407.13759


# Physics
## v-jepa-2
V-JEPA 2 world model and new benchmarks for physical reasoning
https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## Benchmarks:
### PBench: A Physical AI Benchmark for World Models
We measure two scores in PBench: Domain Score and Quality Score. The Domain Score measures the domain-specific capabilities of the world model via the QA pairs, and the Quality Score measures the quality of the generated video. Given a generated video from a world model and the corresponding QA pairs in PBench, we employ a VLM (Qwen2.5-VL-72B-Instruct) as a judge to measure the generated video by calculating the accuracy as the domain score on all the QA pairs for each sample.
* https://research.nvidia.com/labs/dir/pbench/

### E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
https://e3dbench.github.io/

### VBench-2.9
Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness:
https://huggingface.co/papers/2503.21755

### Physics IQ Benchmark:
Do generative video models learn physical principles from watching videos?
* *Paper:*  https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view
* *Project and code:* [[https://github.com/Genesis-Embodied-AI/Genesis](https://physics-iq.github.io/)](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

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
* *Paper:* https://arxiv.org/pdf/2410.12822


## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location (by a text description or an
image), and generate a journey through a long sequence of
diverse yet coherently connected 3D scene

* *Paper:*  https://arxiv.org/pdf/2312.03884
  
## 3d Generalist: Vision-Language-Action Models for Crafting 3D Worlds
*Project:*  https://3d-generalist.pages.dev/

## 3World-consistent Video Diffusion with Explicit 3D Modeling 
*Project:* https://zqh0253.github.io/wvd/

## Matrix3D: Large Photogrammetry Model All-in-One
* *Paper:* https://arxiv.org/pdf/2502.07685

## Domain-free generation of 3d gaussian splatting scenes
* *Code:*  https://luciddreamer-cvlab.github.io/
* *Paper:* https://arxiv.org/abs/2311.13384

# Humans
## HumanDiT
* *Project:* https://agnjason.github.io/HumanDiT-page/

# Mapping
## Mars
Martian World Models : Controllable Video Synthesis with Physically Accurate 3D Reconstructions
https://marsgenai.github.io/

## GPS as a Control Signal for Image Generation
https://cfeng16.github.io/gps-gen/

## Streetscapes: Large-scale Consistent Street View Generation using Autoregressive Video Diffusion
underlying map/layout hosting the desired trajectory.
* *Paper:* https://arxiv.org/abs/2407.13759


# Physics
## v-jepa-2
V-JEPA 2 world model and new benchmarks for physical reasoning
https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## Benchmarks:
### PBench: A Physical AI Benchmark for World Models
We measure two scores in PBench: Domain Score and Quality Score. The Domain Score measures the domain-specific capabilities of the world model via the QA pairs, and the Quality Score measures the quality of the generated video. Given a generated video from a world model and the corresponding QA pairs in PBench, we employ a VLM (Qwen2.5-VL-72B-Instruct) as a judge to measure the generated video by calculating the accuracy as the domain score on all the QA pairs for each sample.
* https://research.nvidia.com/labs/dir/pbench/

### E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
https://e3dbench.github.io/

### VBench-2.9
Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness:
https://huggingface.co/papers/2503.21755

### Physics IQ Benchmark:
Do generative video models learn physical principles from watching videos?
* *Paper:*  https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view
* *Project and code:* [[https://github.com/Genesis-Embodied-AI/Genesis](https://physics-iq.github.io/)](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

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
* *Paper:* https://arxiv.org/pdf/2410.12822


## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location (by a text description or an
image), and generate a journey through a long sequence of
diverse yet coherently connected 3D scene

* *Paper:*  https://arxiv.org/pdf/2312.03884
  
## 3d Generalist: Vision-Language-Action Models for Crafting 3D Worlds
*Project:*  https://3d-generalist.pages.dev/

## 3World-consistent Video Diffusion with Explicit 3D Modeling 
*Project:* https://zqh0253.github.io/wvd/

## Matrix3D: Large Photogrammetry Model All-in-One
* *Paper:* https://arxiv.org/pdf/2502.07685

## Domain-free generation of 3d gaussian splatting scenes
* *Code:*  https://luciddreamer-cvlab.github.io/
* *Paper:* https://arxiv.org/abs/2311.13384

# Humans
## HumanDiT
* *Project:* https://agnjason.github.io/HumanDiT-page/

# Mapping
## Mars
Martian World Models : Controllable Video Synthesis with Physically Accurate 3D Reconstructions
https://marsgenai.github.io/

## GPS as a Control Signal for Image Generation
https://cfeng16.github.io/gps-gen/

## Streetscapes: Large-scale Consistent Street View Generation using Autoregressive Video Diffusion
underlying map/layout hosting the desired trajectory.
* *Paper:* https://arxiv.org/abs/2407.13759


# Physics
## v-jepa-2
V-JEPA 2 world model and new benchmarks for physical reasoning
https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## Benchmarks:
### PBench: A Physical AI Benchmark for World Models
We measure two scores in PBench: Domain Score and Quality Score. The Domain Score measures the domain-specific capabilities of the world model via the QA pairs, and the Quality Score measures the quality of the generated video. Given a generated video from a world model and the corresponding QA pairs in PBench, we employ a VLM (Qwen2.5-VL-72B-Instruct) as a judge to measure the generated video by calculating the accuracy as the domain score on all the QA pairs for each sample.
* https://research.nvidia.com/labs/dir/pbench/

### E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
https://e3dbench.github.io/

### VBench-2.9
Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness:
https://huggingface.co/papers/2503.21755

### Physics IQ Benchmark:
Do generative video models learn physical principles from watching videos?
* *Paper:*  https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view
* *Project and code:* [[https://github.com/Genesis-Embodied-AI/Genesis](https://physics-iq.github.io/)](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

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
* *Paper:* https://arxiv.org/pdf/2410.12822


## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location (by a text description or an
image), and generate a journey through a long sequence of
diverse yet coherently connected 3D scene

* *Paper:*  https://arxiv.org/pdf/2312.03884
  
## 3d Generalist: Vision-Language-Action Models for Crafting 3D Worlds
*Project:*  https://3d-generalist.pages.dev/

## 3World-consistent Video Diffusion with Explicit 3D Modeling 
*Project:* https://zqh0253.github.io/wvd/

## Matrix3D: Large Photogrammetry Model All-in-One
* *Paper:* https://arxiv.org/pdf/2502.07685

## Domain-free generation of 3d gaussian splatting scenes
* *Code:*  https://luciddreamer-cvlab.github.io/
* *Paper:* https://arxiv.org/abs/2311.13384

# Humans
## HumanDiT
* *Project:* https://agnjason.github.io/HumanDiT-page/

# Mapping
## Mars
Martian World Models : Controllable Video Synthesis with Physically Accurate 3D Reconstructions
https://marsgenai.github.io/

## GPS as a Control Signal for Image Generation
https://cfeng16.github.io/gps-gen/

## Streetscapes: Large-scale Consistent Street View Generation using Autoregressive Video Diffusion
underlying map/layout hosting the desired trajectory.
* *Paper:* https://arxiv.org/abs/2407.13759


# Physics
## v-jepa-2
V-JEPA 2 world model and new benchmarks for physical reasoning
https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## Benchmarks:
### PBench: A Physical AI Benchmark for World Models
We measure two scores in PBench: Domain Score and Quality Score. The Domain Score measures the domain-specific capabilities of the world model via the QA pairs, and the Quality Score measures the quality of the generated video. Given a generated video from a world model and the corresponding QA pairs in PBench, we employ a VLM (Qwen2.5-VL-72B-Instruct) as a judge to measure the generated video by calculating the accuracy as the domain score on all the QA pairs for each sample.
* https://research.nvidia.com/labs/dir/pbench/

### E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
https://e3dbench.github.io/

### VBench-2.9
Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness:
https://huggingface.co/papers/2503.21755

### Physics IQ Benchmark:
Do generative video models learn physical principles from watching videos?
* *Paper:*  https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view
* *Project and code:* [[https://github.com/Genesis-Embodied-AI/Genesis](https://physics-iq.github.io/)](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

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
* *Paper:* https://arxiv.org/pdf/2410.12822


## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location (by a text description or an
image), and generate a journey through a long sequence of
diverse yet coherently connected 3D scene

* *Paper:*  https://arxiv.org/pdf/2312.03884
  
## 3d Generalist: Vision-Language-Action Models for Crafting 3D Worlds
*Project:*  https://3d-generalist.pages.dev/

## 3World-consistent Video Diffusion with Explicit 3D Modeling 
*Project:* https://zqh0253.github.io/wvd/

## Matrix3D: Large Photogrammetry Model All-in-One
* *Paper:* https://arxiv.org/pdf/2502.07685

## Domain-free generation of 3d gaussian splatting scenes
* *Code:*  https://luciddreamer-cvlab.github.io/
* *Paper:* https://arxiv.org/abs/2311.13384

# Humans
## HumanDiT
* *Project:* https://agnjason.github.io/HumanDiT-page/

# Mapping
## Mars
Martian World Models : Controllable Video Synthesis with Physically Accurate 3D Reconstructions
https://marsgenai.github.io/

## GPS as a Control Signal for Image Generation
https://cfeng16.github.io/gps-gen/

## Streetscapes: Large-scale Consistent Street View Generation using Autoregressive Video Diffusion
underlying map/layout hosting the desired trajectory.
* *Paper:* https://arxiv.org/abs/2407.13759


# Physics
## v-jepa-2
V-JEPA 2 world model and new benchmarks for physical reasoning
https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

## Genesis: physics platform designed for general purpose Robotics/Embodied AI/Physical AI applications
It is simultaneously multiple things:
    A universal physics engine re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
    A lightweight, ultra-fast, pythonic, and user-friendly robotics simulation platform.
    A powerful and fast photo-realistic rendering system.
    A generative data engine that transforms user-prompted natural language description into various modalities of data.
    
* *Project and code:* https://github.com/Genesis-Embodied-AI/Genesis

## Benchmarks:
### PBench: A Physical AI Benchmark for World Models
We measure two scores in PBench: Domain Score and Quality Score. The Domain Score measures the domain-specific capabilities of the world model via the QA pairs, and the Quality Score measures the quality of the generated video. Given a generated video from a world model and the corresponding QA pairs in PBench, we employ a VLM (Qwen2.5-VL-72B-Instruct) as a judge to measure the generated video by calculating the accuracy as the domain score on all the QA pairs for each sample.
* https://research.nvidia.com/labs/dir/pbench/

### E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
https://e3dbench.github.io/

### VBench-2.9
Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness:
https://huggingface.co/papers/2503.21755

### Physics IQ Benchmark:
Do generative video models learn physical principles from watching videos?
* *Paper:*  https://drive.google.com/file/d/1ac93646QoMlFtcO_6GlcZXex7lPzRWaE/view
* *Project and code:* [[https://github.com/Genesis-Embodied-AI/Genesis](https://physics-iq.github.io/)](https://physics-iq.github.io/)

# World models history: 2013 - 2023 
See list below for papers
* 2013  Decoding “World Models” by David Ha and Jürgen Schmidhuber: A Milestone in AI Research
 * *Project:*  https://worldmodels.github.io/

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
* *Paper:* https://arxiv.org/pdf/2410.12822


## Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models
Proposes Promptable Game Models (PGMs), controllable models of games that are learned from annotated videos.
* *Project:* https://snap-research.github.io/promptable-game-models/
* *Paper:* https://arxiv.org/pdf/2303.13472

## Navigation World Models and self-driving cars

## Epona: Autoregressive Diffusion World Model for Autonomous Driving 
* *Paper:* https://arxiv.org/abs/2506.24113

## Navigation is a fundamental skill of agents with visual-motor capabilities. 

 Navigation World Model (NWM), a controllable video generation model that predicts future visual observation

* *Paper:* https://arxiv.org/pdf/2410.10774
* *Project:* https://www.amirbar.net/nwm/

## DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
We introduce a simple method for constructing versatile world models with pre-trained DINOv2 that generalize to complex environment dynamics, which enables zero-shot solutions at test time for arbitrary goals

* *Paper:* https://arxiv.org/pdf/2411.04983

##  Unified Video Action Mode
Joint Video-Action Optimization -- learns a unified latent space for both video and action generation.
Decoupled Video-Action Decoding -- speeds up policy inference by skipping video generation.
Masked Training -- enables a single model to handle diverse tasks while reducing overfitting.
* *Project:* https://unified-video-action-model.github.io/
* *Code:* https://github.com/ShuangLI59/unified_video_action

## Temporally-Controlled Multi-Event Video Generation
MinT is the first text-to-video model capable of generating sequential events and controlling their timestamps.
* *Project:* https://mint-video.github.io/
https://arxiv.org/abs/2412.05263

## Creating New Games with Generative Interactive Videos
* *Paper:* https://arxiv.org/pdf/2501.08325
* 
# Camera control
## Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
We present Cavia, the first framework that enables users to generate multiple videos of the same scene with precise control over camera motion, while simultaneously preserving object motion.
* *Paper:* https://arxiv.org/pdf/2410.10774

## ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
https://shotadapter.github.io/

##### GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, using Cosmos
https://research.nvidia.com/labs/toronto-ai/GEN3C/

# 3d/4d World creation
## DimensionX
DimensionX: Create Any 3D and 4D Scenes from a Single Image with
Controllable Video Diffusion
* *Code:* https://github.com/wenqsun/DimensionX
* *Paper:* https://arxiv.org/pdf/2411.04928
## HunyuanWorld 3d
immersive and playable 3D worlds from texts or images 
* *Code:*  https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
* *Paper:* https://arxiv.org/pdf/2507.21809

## WorldGen, 2025
Worldgen: Generate any 3d scene in seconds. 

* *Code:* https://github.com/ZiYang-xie/

## WonderJourney: Going from Anywhere to Everywhere
Start at any user-provided location

## Background 
### Temporal consistency
- **FramePack-P1** – [FramePack-P1](https://lllyasviel.github.io/frame_pack_gitpage/p1/)

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
