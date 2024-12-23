# awesome-world-models
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
https://odyssey.systems/introducing-explorer#waitlist
https://www.worldlabs.ai/blog
https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
https://deepmind.google/technologies/veo/veo-2/
https://openai.com/sora/
Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond (from GigaAI)

## Playable demos 
https://next.journee.ai/xyz-diamond
https://oasis.decart.ai/welcome

## Papers
