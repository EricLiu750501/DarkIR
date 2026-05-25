Summary:
Problem Definition :
Photography in extreme low-light environments often suffers from a combination of degradations, including severe noise, low visibility, and motion blur caused by long exposure times. Existing methods typically treat low-light image enhancement (LLIE) and deblurring as separate tasks, failing to provide a robust, unified solution for real-world scenarios.
Proposed Solution :
DarkIR introduces an "All-in-One" restoration framework that handles denoising, deblurring, and enhancement simultaneously within a single model. It employs an asymmetric processing strategy: addressing illumination correction at low resolution in the encoder, followed by upsampling and deblurring in the decoder.
Architecture and Method :
The model follows a lightweight Metaformer structure with specialized components:
Encoder (EBlock): Focuses on light enhancement by manipulating image amplitudes in the Fourier domain through a Frequency MLP (Fre-MLP) and Spatial Attention Module (SpAM).
Decoder (DBlock): Focuses on detail recovery and sharpening using a Dilated-Spatial Attention Module (Di-SpAM), which provides a large receptive field to effectively remove motion blur.
Loss Functions: A multi-term objective combines pixel, perceptual, edge, and architecture-guiding losses to ensure high fidelity and structural consistency.
Key Advantages:
Efficiency: DarkIR is significantly more compact than current SOTA models, using 55% fewer parameters than LEDNet and 88% fewer than Restormer. Its computational cost (MACs) is reduced by up to 20 times compared to leading transformers.
Superior Performance: It achieves state-of-the-art results across major benchmarks like LOL-Blur and Real-LOLBlur.
Robustness: The model generalizes exceptionally well to real-world night scenes, showing resilience against handheld camera shake and sensor noise.
Goal and objective: 
Because there is already quite a comprehensive github code. The goal trying is to add a temporal extension to DarkIR. Compare adding a temporal for better noise suppression capability, Detail recovery effect.
Expected Steps:
extends the original DarkIR single-image input into a short temporal window using three consecutive frames
Add a lightweight Temporal Fusion Module is then applied before feeding the fused representation into the original DarkIR network for restoration.
Add a temporal consistency constraint to limit abrupt changes between neighboring output frames
dataset: use BVI-Lowlight: Fully registered datasets for low-light image and video enhancement


Expected outcome:
Reduce random noise in low-light environments.
Improve detail restoration capability.
Improve temporal consistency across frames.
Reduce flickering during video playback.
The proposed method will also be compared against the original single-frame DarkIR model.


