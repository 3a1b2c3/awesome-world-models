# awesome-world-models
#TOC

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

## 101: Start here
World Models: the AI technology that could displace 3D as we know it

## Commercial, no code or papers
* https://odyssey.systems/introducing-explorer#waitlist
* https://www.worldlabs.ai/blog
* https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/* 
* https://deepmind.google/technologies/veo/veo-2/
* https://openai.com/sora/ Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond (from GigaAI)

## Playable demos 
https://next.journee.ai/xyz-diamond
https://oasis.decart.ai/welcome

## Papers and code
###OpenSora
https://hpcaitech.github.io/Open-Sora/ no windows, 24 GB RAM+
https://github.com/PKU-YuanGroup/Open-Sora-Plan
by the Colossal-AI team with the development of Open-Sora, a replication architecture solution for the Sora model, marks a significant advancement in the field. This solution mirrors the capabilities of the Sora model in video generation and brings forth a remarkable reduction in training costs by 46%. Additionally, it extends the length of the model training input sequence to 819K patches, pushing the boundaries of what’s possible in AI-driven video generation.
🧵🧵 [Download] Evaluation of Large Language Model Vulnerabilities Report (Promoted)

Open-Sora’s methodology revolves around a comprehensive training pipeline incorporating video compression, denoising, and decoding stages to process and generate video content efficiently. Using a video compression network, the model compresses videos into sequences of spatial-temporal patches in latent space, then refined through a Diffusion Transformer for denoising, followed by decoding to produce the final video output. This innovative approach allows for handling various sizes and complexities of videos with improved efficiency and reduced computational demands.
The performance of Open-Sora is noteworthy, showcasing over a 40% improvement in efficiency and cost reduction compared to baseline solutions. Furthermore, it enables the training of longer sequences, up to 819K+ patches, while maintaining or even enhancing training speeds. This performance leap demonstrates the solution’s capability to address the challenges of computational cost and resource efficiency in AI video generation. It also reassures the audience about its practicality and value, making high-quality video production more accessible to a wider range of users.


## Navigation World Models
Amir BarGaoyue ZhouDanny TranTrevor DarrellYann LeCun meta
Navigation is a fundamental skill of agents with visual-motor capabilities. We introduce a Navigation World Model (NWM), a controllable video generation model that predicts future visual observations…
* https://arxiv.org/pdf/2410.10774
* https://www.amirbar.net/nwm/






DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (from NYU)
https://arxiv.org/pdf/2411.04983








From Slow Bidirectional to Fast Causal Video Generators





Learning Generative Interactive Environments by Trained Agent Exploration
https://github.com/insait-institute/GenieRedux
https://arxiv.org/pdf/2409.06445


𝐆𝐞𝐧𝐢𝐞𝐑𝐞𝐝𝐮𝐱 learns to simulate interactive environments simply by observing interactions during training.




The Matrix: Infinite-Horizon World Generation with Real-Time Moving Control
No code
https://thematrix1999.github.io/











GameGen-X: Interactive Open-world Game Video Generation
Haoxuan CheXuanhua HeQuande LiuCheng JinHao Chen
Computer Science
arXiv.org1 November 2024
We introduce GameGen-X, the first diffusion transformer model specifically designed for both generating and interactively controlling open-world game videos. This model facilitates high-quality,…

Diffusion for World Modeling: Visual Details Matter in Atari  DIAMOND, code
Eloi AlonsoAdam Jelley
https://github.com/eloialonso/diamond







Efficient World Models with Context-Aware Tokenization
https://github.com/vmicheli/delta-iris







Vincent MicheliEloi AlonsoFranccois Fleuret
Scaling up deep Reinforcement Learning (RL) methods presents a significant challenge. Following developments in generative modelling, model-based RL positions itself as a strong contender. 

AVID: Adapting Video Diffusion Models to World Models
Marc RigterTarun GuptaAgrin HilmkilChao Ma
https://arxiv.org/pdf/2410.12822

Large-scale generative models have achieved remarkable success in a number of domains. However, for sequential decision-making problems, such as robotics, action-labelled data is often scarce and…







Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
No code






Basic Dit "Scalable Diffusion Models with Transformers" 
https://github.com/facebookresearch/DiT
https://arxiv.org/pdf/2212.09748
OpenDiT: An acceleration for DiT training. We adopt valuable acceleration strategies for training progress from OpenDiT.
PixArt: An open-source DiT-based text-to-image model.
Latte: An attempt to efficiently train DiT for video
https://lilianweng.github.io/posts/2024-04-12-diffusion-video/
https://encord.com/blog/vision-transformers/
The Original model Oasis and other architectures are based on
-embeddings and masking
Vit and transformer
ViT: Transformers for Image Recognition [Paper] [Blog] [Video]
https://lilianweng.github.io/posts/2024-04-12-diffusion-video/
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
- https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
- https://arxiv.org/abs/2106.10270
Masked Autoencoders: A PyTorch Implementation https://github.com/facebookresearch/mae/blob/main/models_vit.py



Each patch will make a token of length 400.

Vit tokens

Vit
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py



Dit

