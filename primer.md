# A Gentle Primer to Data Synthesis and Generation Architectures

## Introduction to Data Synthesis and Generation

### Motivation

In biomedical and healthcare fields, it's often hard to get enough high-quality data. Biosignals like **BCG**, **ECG**, **PPG**, **EEG**, and respiration data can be challenging to work with because:

* **Class imbalance**: Unusual events like arrhythmias or sleep apnea are rare, so the data is skewed.
* **Privacy concerns**: Strict patient confidentiality rules limit access to real datasets.
* **High cost of data collection**: Recording and labeling biosignals typically requires expensive equipment, trained personnel, and clinical settings.
* **Noise and variation**: Biosignals can differ widely between individuals and often include noise from movement or sensor issues.

To address these problems, we use data synthesis techniques to create realistic, high-quality artificial data. This can help:

* Enrich small datasets and improve model performance.
* Simulate rare or extreme conditions.
* Test and validate algorithms more effectively.
* Support privacy-friendly data sharing and research.

### What is Data Synthesis

Data synthesis means generating artificial data that resembles real data in structure and behavior. It can be used to:

* **Augment** real datasets by adding new variations.
* **Impute** missing or noisy parts of data.
* **Generate** entirely new data samples that still follow real-world patterns.
* **Simulate** data using physical models or statistical assumptions.

Synthesis methods fall into two main types:

* **Deterministic**: Based on rules or simulations (e.g., a mathematical model of heartbeats).
* **Probabilistic**: Based on learned distributions or models that generate data *stochastically*.

In biosignals, synthesis could be as simple as slicing real signals into new segments or as advanced as training a deep learning model to simulate disease progression.

### Framing the Landscape

There are several broad categories of data synthesis methods, each suited to different needs:

1. **Classical Statistical Methods**:

   * These include bootstrapping, time-series models like ARIMA, Gaussian processes, signal simulators, and resampling-based techniques like SMOTE (Synthetic Minority Over-sampling Technique) for addressing class imbalance.
   * They are usually easy to interpret and good for small datasets.

2. **Machine Learning-Based Methods**:

   * These learn patterns directly from data to create new examples.
   * Key approaches include:

     * **Autoencoders (AE)** and **Variational Autoencoders (VAE)**: Learn compact signal representations that can be used to reconstruct or generate new data.
     * **Generative Adversarial Networks (GANs)**: Use two competing models to produce realistic synthetic samples.
     * **Transformers**: Sequence models that are well-suited to generating complex time-dependent biosignal data.

Each approach has its own pros and cons in terms of ease of use, data needs, computational cost, and realism. Throughout this session, we’ll look at these methods with a focus on biosignal applications and help you understand how to choose the right technique for your use case.

## Classical Statistical Methods

### Signal Modeling Foundations

Signal modeling is the starting point of many classical synthesis techniques. For ballistocardiography (BCG), this often includes:

* **Autoregressive models (AR, ARMA, ARIMA)** to capture repeating cardiac-related oscillations over time.
* **Fourier or Wavelet-based decompositions** to analyze frequency components related to heartbeat, respiration, and motion artifacts.
* **Gaussian Processes** to model smooth, quasi-periodic components in the BCG waveform with uncertainty quantification.
  These models help capture temporal structures such as the repeating I-J-K complex in BCG signals, which reflect mechanical movements associated with the cardiac cycle.

### Bootstrapping and Resampling

Bootstrapping refers to generating new datasets by resampling with replacement from the original data. In the case of BCG:

* **Window slicing**: Extracting overlapping or randomly shifted windows around cardiac cycles (e.g., I-J-K complexes).
* **Permutation**: Shuffling or jittering sub-windows to introduce variability in timing or amplitude while maintaining physiological structure.
* **SMOTE/ADASYN**: Used on extracted BCG features (e.g., peak-to-peak intervals, amplitudes) to synthesize minority-class examples in classification tasks.
  These techniques are useful in creating balanced datasets for sleep staging, arrhythmia detection, or posture classification from BCG.

### Parametric Modelling

Parametric models assume BCG signals or their derived features follow a known distribution. For example:

* **Peak amplitudes** (I, J, K) or inter-beat intervals could be modeled using Gaussian, log-normal, or gamma distributions.
* **Respiratory modulations in the BCG envelope** could follow sinusoidal or exponential decay patterns.
* **Additive noise** may be modeled as Gaussian or uniform, depending on the sensing setup (e.g., bed sensor vs wearable).
  Once fitted, these distributions can be sampled to generate realistic waveforms or features.

### Fit Distributions

This involves fitting statistical distributions to empirical BCG measurements:

* **GMMs** on amplitude distributions of J-peaks across multiple users.
* **Kernel Density Estimation (KDE)** for cycle-to-cycle timing variability.
* **Histograms** to capture the distribution of signal-derived metrics like heart rate variability or BCG beat widths.
  These fits enable generation of synthetic cycles or segments with statistical fidelity to real data.

### Copula Models

Copulas allow modelling complex dependencies between BCG-derived variables:

* Joint modeling of **I-J-K amplitudes**, **inter-beat intervals**, and **respiratory modulations**.
* Can preserve nonlinear relationships between metrics that might be lost in simpler parametric modeling.
  This is valuable for generating multi-feature synthetic datasets that reflect the joint statistical structure of BCG recordings.

### Simulators

Physiological simulators for BCG are less mature than for ECG but still feasible:

* Use **biomechanical models** of cardiac-induced body movement to simulate the BCG waveform.
* Approximate cardiac motion as a sum of Gaussian or sinusoidal components representing the I, J, and K peaks.
* Introduce variation via parameter tuning: heart rate, stroke volume, respiration rate, posture.
  These simulations are useful for algorithm prototyping, training, and robustness testing.

### Strengths

* Can work with limited labeled data.
* Offer interpretable control over signal properties.
* Enable fast, lightweight prototyping.
* Provide a foundation for physiologically inspired synthesis.

### Limitations of Classical Statistical Methods

* May oversimplify complex waveforms, especially under pathological or noisy conditions.
* Often rely on assumptions of linearity or stationarity.
* Hard to extend to multi-sensor or multivariate BCG setups.
* May miss subtle morphological changes (e.g., those caused by aging, disease, or motion).

## Autoencoders (AEs)

### Concept and Architecture

Autoencoders are a class of unsupervised neural networks trained to reconstruct their input. The central idea is to learn a low-dimensional encoding that captures the essential structure of the data. An autoencoder is composed of two main components:

* **Encoder** $f_\theta$: Maps the input $x$ to a latent representation $z = f_\theta(x)$.
* **Decoder** $g_\phi$: Attempts to reconstruct the input $\hat{x} = g_\phi(z)$ from the latent code.

The model is trained to minimize a reconstruction loss, typically:

$\mathcal{L}_{\text{AE}} = \| x - \hat{x} \|^2$

This encourages the encoder-decoder pair to compress and decompress the signal in a way that preserves its important features.

Autoencoders learn a **nonlinear projection** of the input space onto a **latent manifold**, which ideally captures key modes of variation in the data. For time series signals like BCG, this means preserving cycle morphology, timing dynamics, and person-specific signal signatures.

### Application to BCG Data

In the context of BCG, autoencoders are valuable for both analysis and synthesis:

* **Denoising**: Autoencoders can be trained to reconstruct clean signals from noisy inputs, learning to suppress motion artifacts.
* **Anomaly detection**: Abnormal beats that deviate from learned latent patterns will reconstruct poorly, serving as a basis for unsupervised detection.
* **Latent representation learning**: The encoder compresses each beat or segment into a low-dimensional vector that summarizes heartbeat morphology and rhythm.
* **Synthetic signal generation**: Although basic AEs are not inherently generative, one can interpolate in the latent space between real BCG encodings to synthesize novel, realistic-like signals.

This latent space becomes a **proxy for the structure of the data manifold**. For example, smooth transitions in latent space often correspond to smooth variations in heart rate, amplitude, or posture in the reconstructed BCG signals.

### Technical Implementation

Autoencoders for BCG typically operate on 1D waveform segments of fixed length (e.g., 3750 samples per window : **15s @ 250Hz**). Common architectural choices include:

* **1D Convolutional Autoencoders**: Encoders use stacked `Conv1D` layers to reduce temporal resolution while increasing feature abstraction; decoders use `ConvTranspose1D` to reconstruct the original waveform shape.
* **Fully Connected (FC) Autoencoders**: More common for fixed-size windows when signal shape is simpler or flattened.
* **Recurrent AEs**: Less common, but potentially useful for preserving sequential dependencies across longer time spans.

Design choices include:

* **Latent dimensionality**: Affects how compressed the information is; smaller latent spaces enforce more abstraction.
* **Activation functions**: `ReLU`, `LeakyReLU`, or `Tanh` depending on whether negative values are meaningful in the signal.
* **Loss functions**: L2 is common, but others (e.g., L1, dynamic time warping loss) may better capture morphological fidelity.

### Limitations for Data Synthesis

While AEs are great for compression and anomaly detection, they have several limitations for full generative synthesis:

* **No structured prior**: The learned latent space is arbitrary and may have discontinuities, making it difficult to sample new points that decode into realistic signals.
* **Deterministic mapping**: Given an input, the encoder always outputs the same latent code—there’s no built-in stochasticity.
* **Poor interpolation/extrapolation**: Latent vectors may not lie on a meaningful manifold, so decoded interpolations can be unrealistic.

Despite these limitations, AEs form a crucial building block in the synthesis pipeline. They help us:

* Understand the intrinsic structure of BCG data.
* Learn features that can be used downstream in more powerful generative models (e.g., VAEs or GANs).
* Enable simple forms of synthetic signal augmentation by manipulating the latent space.

## Variational Autoencoders (VAEs)

### Motivation and Concept

Variational Autoencoders (VAEs) extend traditional autoencoders by framing the latent space as a **probabilistic distribution** rather than a fixed point. This allows for structured, interpretable, and generative latent representations—making VAEs ideal for data synthesis.

The core idea is to model the latent space $z$ as a distribution $q_\theta(z|x)$, typically Gaussian, and to encourage it to approximate a simple prior $p(z) \sim \mathcal{N}(0, I)$. Instead of mapping each input to a point in latent space, the encoder outputs parameters of a probability distribution:

* $\mu = f_\theta^{(\mu)}(x)$, the mean vector
* $\sigma = f_\theta^{(\sigma)}(x)$, the standard deviation vector

A sample from the latent space is drawn via the **reparameterization trick**:

$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$

The decoder then generates the reconstructed signal from the sampled latent code: $\hat{x} = g_\phi(z)$.

### Loss Function

VAEs are trained by optimizing the **evidence lower bound (ELBO)**:

$ \mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\theta(z|x)}[\log p_\phi(x|z)] - D_{\text{KL}}(q_\theta(z|x) \| p(z))$

* The first term is a reconstruction loss (similar to traditional AEs).
* The second term is the KL divergence, which regularizes the latent space to be close to a standard normal distribution.

This balance ensures that:

* Samples in latent space are drawn from a smooth, continuous, and meaningful manifold.
* New samples can be generated simply by sampling $z \sim \mathcal{N}(0, I)$ and passing them through the decoder.
* **Generate diverse variants** of similar signals by sampling different latent codes around a known signal’s encoding.
* **Model uncertainty**: Because the encoder returns a distribution, VAEs can capture variability within repeated measurements or among subjects.

### Practical Considerations

* **Architecture**: Similar to AEs (1D Conv, FC, or hybrid), but with two heads in the encoder to output $\mu$ and $\sigma$, in the case of the *latent space* being Gaussian.
* **Latent space dimension**: Should be large enough to capture variability but small enough to remain interpretable.
* **Training dynamics**: KL loss annealing (gradually increasing its weight) is often used to avoid posterior collapse.
* **Sampling and decoding**: Sampling multiple $z$ values for a given input can help visualize variation around the same signal.

### Advantages for Synthesis

* **Principled generative process**: Sampling from a known prior ensures high-quality, realistic samples.
* **Smooth latent space**: Encourages continuity—meaning that interpolations and perturbations produce coherent outputs.
* **Better control**: You can condition or constrain the latent space to generate signals with desired characteristics.

### Limitations

* May underfit sharp signal features (e.g., steep transitions in BCG) due to Gaussian assumptions.
* KL loss may dominate during training, leading to **posterior collapse** (decoder ignores latent code).
* Sampling quality is sensitive to the choice of latent space size and architecture.

### Summary

VAEs offer a natural bridge between statistical modeling and deep learning. Their probabilistic foundation makes them one of the most well-principled techniques for synthetic signal generation, especially for biosignals like BCG where controlled variation and physiological realism are key.

## Generative Adversarial Networks (GANs)

### Quick Overview

Generative Adversarial Networks (GANs) are a class of deep generative models that learn to generate new data by playing a game between two neural networks:

* **Generator (G)**: Learns to map random noise vectors $z \sim \mathcal{N}(0, I)$ to realistic data samples $G(z) $.
* **Discriminator (D)**: Tries to distinguish between real data and fake samples produced by the generator.

The two networks are trained in an adversarial manner:

* The generator tries to fool the discriminator.
* The discriminator tries to detect whether a sample is real or fake.

The objective is a minimax game:

$ \min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p(z)} [\log (1 - D(G(z)))]$

When trained successfully, the generator learns to produce data that is indistinguishable from the real data distribution.

### Applications

In the context of BCG and biosignal synthesis, GANs have been explored for:

* **Realistic signal generation**: Learning to generate plausible BCG segments with natural variation.
* **Data augmentation**: Enhancing minority classes (e.g., rare sleep stages, abnormal beats) to balance datasets.
* **Modality translation**: Generating one signal type from another (e.g., PPG $\rightarrow$ BCG).
* **Style transfer**: Learning to impose subject-specific patterns or device-specific noise characteristics.

Conditional GANs (cGANs) extend this framework by adding a conditioning variable (e.g., heart rate, sleep stage, or user ID) to guide generation.

### Limitations of Generative Adversarial Networks

Despite their promise, GANs come with several significant challenges:

* **Training instability**: GANs are notoriously difficult to train; the generator and discriminator may not converge.
* **Mode collapse**: The generator may produce limited diversity (e.g., same waveform repeated with minor changes).
* **Lack of inference**: Unlike VAEs, GANs don’t provide a natural way to encode existing signals into a latent space.
* **Evaluation difficulty**: There are no universal metrics for measuring the quality or diversity of generated signals, especially for biosignals.

In biosignal contexts like BCG, where realism, physiological plausibility, and fine-scale variability are important, these issues can limit their practical utility.

### How GANs Differ from Diffusion Models

While GANs rely on an adversarial setup with two networks competing, diffusion models—such as Stable Diffusion—take a fundamentally different approach based on gradual denoising of noise.

* **GANs** learn a direct mapping from random noise to realistic samples via a generator, trained against a discriminator.
* **Diffusion models** learn to reverse a noising process. They start with pure noise and gradually denoise it using a learned probabilistic model (often a U-Net architecture).

Key differences:

* **Training stability**: Diffusion models are typically more stable than GANs and do not suffer from mode collapse.
* **Sample diversity**: Diffusion models tend to generate more diverse outputs because they cover the data distribution more completely.
* **Sampling speed**: GANs are fast at inference; diffusion models are typically slower because they require many steps to denoise.
* **Latent control**: Diffusion models can be easily conditioned (e.g., text, label, signal class), often with better control over output than GANs.

While diffusion models are currently more popular in image domains, their adaptation to biosignal synthesis is ongoing and promising, particularly due to their robustness and quality.

### Status

GANs remain an active area of research in the biosignal space, but adoption in production systems is limited. Most work is still exploratory, focusing on improving training stability, tailoring architectures (e.g., using 1D convolutions), or combining GANs with other methods (e.g., VAE-GAN hybrids).

In summary, GANs offer exciting potential for highly realistic signal generation, but they come with caveats that make them more challenging to apply effectively—especially in safety-critical or medically regulated applications like BCG monitoring.

## Transformers

### Why Transformers?

Transformers were originally developed for natural language processing but have since proven highly effective for any data with temporal or sequential structure—including biosignals like BCG. Unlike RNNs or CNNs, transformers use self-attention mechanisms that:

* Model long-range dependencies across sequences.
* Scale well with data.
* Operate in parallel, improving efficiency and performance.

This makes them particularly well-suited for modeling BCG signals, which exhibit complex, multi-scale patterns over time.

### Auto-Regressive Generation

Transformers can be used in an **auto-regressive** fashion to generate signals step-by-step:

* Each output is predicted conditioned on previous outputs (i.e., $x_t \sim p(x_t | x_1, ..., x_{t-1})$).
* During training, the model learns to predict the next sample given all past samples.
* At inference, the model can generate new synthetic sequences one step at a time.

Variants like **Transformers with causal masking** or **GPT-style decoders** are commonly used for this setup.

Alternatively, encoder-decoder transformers can be used for **sequence-to-sequence** generation—useful for translating one modality to another or simulating a signal under different conditions (e.g., BCG under stress vs rest).

### Examples

Some use cases of transformer-based synthesis in biosignals include:

* **BCG waveform generation**: Learning the dynamics of multiple heartbeat cycles from raw signals.
* **Multi-modal synthesis**: Predicting PPG or accelerometer signals from BCG input.
* **Context-aware augmentation**: Generating synthetic signals conditioned on metadata such as age, posture, or device placement.
* **Imputation**: Filling in missing regions of a signal with plausible completions.

In particular, transformers excel in scenarios where dependencies span large time intervals (e.g., respiration-modulated BCG), or where fine control is needed over generation.

### Benefits

* **Captures long-range structure**: Ideal for full-length waveform modeling.
* **Flexible architecture**: Can be adapted for univariate, multivariate, or even hybrid signal representations.
* **Scalable**: More data typically improves performance.
* **Amenable to conditioning**: Easy to incorporate auxiliary inputs like labels or context.

### Challenges

* **High memory and compute requirements**: Especially for long sequences like raw BCG.
* **Training complexity**: Requires large datasets and careful tuning.
* **Overfitting risks**: Without regularization or proper augmentation, transformers may memorize instead of generalize.
* **Data representation**: Raw waveform vs feature-based input choices can affect results significantly.

### Use Cases

Transformers are gaining traction in biosignal research and show strong promise for:

* **Synthetic dataset creation** for sleep staging, cardiovascular monitoring, or posture recognition.
* **Multimodal generation** in wearable or contactless sensing setups.
* **Model-based simulation** for algorithm stress-testing (e.g., how a heart failure condition might look across devices).

Their flexibility, generative power, and interpretability (via attention maps) make transformers one of the most forward-looking tools for BCG and broader biosignal synthesis.

## Comparitive Analysis and Selection Guide

| Method        | Pros                  | Cons                          | Use Case                          |
|---------------|-----------------------|-------------------------------|-----------------------------------|
| Statistical   | Fast, interpretable   | Limited complexity            | Baselines, simulators             |
| AE/VAE        | Compressive, flexible | Lower fidelity                | Anomaly detection, interpolation  |
| Transformer   | Powerful, contextual  | Compute heavy, data hungry    | Long signals, multimodal          |
| GAN           | High quality samples  | Instability                   | BCG/ECG artifacts, imputation     |

### Key Takeaways

* Use statistical methods for rapid prototyping or interpretable signal generation where simplicity is valued.

* Use autoencoders/VAEs when you need compression, reconstruction, or latent representations for interpolation.

* Use transformers when signal complexity, long-range dependencies, or multimodal context are central.

* Use GANs where sharp detail is required (e.g., simulating signal noise or artifacts), but be cautious of instability.

* Use diffusion models when generation quality and diversity matter most, and you're not constrained by compute time.

Ultimately, the best method depends on:

* Data availability and quality

* Desired output fidelity

* Interpretability vs realism

* Computational budget

> Note: In practice, hybrid approaches (e.g., VAE+GAN, AE+Transformer) are increasingly popular, combining the strengths of multiple techniques.

## Practical Examples and Implementation

### For Biosignals

Biosignal synthesis techniques need to preserve physiological validity, morphological diversity, and temporal dynamics. In the case of BCG, the goal is often to generate realistic heartbeats (I-J-K complexes), rhythm modulations (respiration), or noise-corrupted variants for robust training.

**Common biosignal preprocessing steps:**

* Filtering and normalization
* Cycle segmentation (e.g., beat detection)
* Feature extraction (e.g., peak amplitudes, timing intervals)

**Use cases:**

* Generating synthetic BCG segments for underrepresented physiological states (e.g., tachycardia)
* Expanding training sets for sleep staging or posture classification
* Creating test signals with controlled noise, posture, or respiration parameters

### Machine Learning Based Synthesis

**Autoencoder Example:**

* Train a 1D convolutional autoencoder on 3-second BCG windows
* Latent space exploration to interpolate between beats
* Add perturbations in latent space to generate new variations

**VAE Example:**

* Train a VAE on beat-aligned cycles
* Sample from $ \mathcal{N}(0, I)$ to generate new BCG beats
* Use conditional VAE for generating class-specific beats (e.g., normal vs abnormal)

**Transformer Example:**

* Train an autoregressive transformer on full-length BCG sequences (e.g., 30s windows)
* Use causal masking to generate signals sample-by-sample
* Condition on metadata (e.g., user ID, posture) to produce realistic variations

**GAN Example:**

* Use a 1D GAN to generate realistic I-J-K complexes
* Discriminator trained to distinguish real vs synthetic beats
* Use output to augment training for heartbeat detection

### Evaluation Strategies

**Visual inspection:**

* Overlay real vs synthetic signals
* Inspect beat morphology, rhythm, and noise characteristics

**Statistical similarity:**

* Compare histograms of key metrics (peak amplitudes, interbeat intervals)
* Use tests like KS-test, Fréchet distance, or Earth Mover’s Distance

**Model-based evaluation:**

* Train a classifier or detector on synthetic data; evaluate on real data
* Use synthetic samples for data augmentation and measure downstream task performance

**Human-in-the-loop evaluation:**

* Ask clinicians or annotators to assess realism or label synthetic beats

Combining these strategies provides both qualitative and quantitative insight into the usefulness and realism of generated biosignals. In practice, iterative synthesis and evaluation cycles are essential for refining generative models.

## Future Direction and Considerations

### Ethics & Fairness

As data synthesis becomes more powerful, it raises critical questions around ethics and fairness:

* **Data privacy**: Synthetic signals should not unintentionally leak identifiable characteristics from real users.
* **Bias amplification**: Generative models trained on imbalanced datasets may reinforce existing biases (e.g., age, sex, device placement).
* **Transparency**: Users of synthesized data should be informed whether it’s real or generated—especially in clinical or regulatory contexts.
* **Overtrust**: Realistic-looking synthetic signals may mislead users if their limitations aren't clearly communicated.

Building fair and responsible synthesis pipelines requires careful curation, validation, and governance.

### Hybrid Models

Future progress will likely come from hybrid approaches that combine the strengths of multiple models:

* **VAE-GAN**: Blends VAE’s stable latent space with GAN’s sharp outputs.
* **Transformer-AE**: Uses an autoencoder to compress input before feeding to a transformer for efficient long-sequence modeling.
* **Physio-informed priors**: Embedding physiological knowledge (e.g., timing constraints, heart rate variability) into generative models.

These designs can improve sample quality, controllability, and interpretability—especially important in biosignal domains like BCG.

### Final Note

Data synthesis for biosignals is an emerging and rapidly evolving space. Whether for augmentation, simulation, or anonymization, these techniques offer powerful tools to overcome the limitations of real-world data. But their use must be guided by domain understanding, responsible design, and rigorous validation.

As we move forward, interdisciplinary collaboration—between data scientists, clinicians, and engineers—will be essential to ensure that synthetic data supports both innovation and trust in real-world applications.
