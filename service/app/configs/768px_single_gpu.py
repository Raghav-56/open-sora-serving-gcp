# Open-Sora v2 Single GPU Configuration 768px (No ShardFormer)
# High quality mode for Vertex AI deployment with single A100 GPU

save_dir = "samples"
seed = 42
batch_size = 1
dtype = "bf16"

cond_type = "t2v"  # text-to-video

dataset = dict(type="text")

sampling_option = dict(
    resolution="768px",
    aspect_ratio="16:9",
    num_frames=49,  # Default, can be overridden
    num_steps=50,
    shift=True,
    temporal_reduction=4,
    is_causal_vae=True,
    guidance=7.5,
    guidance_img=3.0,
    text_osci=True,
    image_osci=True,
    scale_temporal_osci=True,
    method="i2v",
    seed=None,
)

motion_score = "4"
fps_save = 24

# Model configuration
model = dict(
    type="flux",
    from_pretrained="./ckpts/Open_Sora_v2.safetensors",
    guidance_embed=False,
    fused_qkv=False,
    use_liger_rope=True,
    in_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    cond_embed=True,
)

ae = dict(
    type="hunyuan_vae",
    from_pretrained="./ckpts/hunyuan_vae.safetensors",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,
    use_temporal_tiling=False,
)

# T5 without shardformer for single GPU stability
t5 = dict(
    type="text_embedder",
    from_pretrained="./ckpts/google/t5-v1_1-xxl",
    max_length=512,
    shardformer=False,  # Disabled for single GPU
)

clip = dict(
    type="text_embedder",
    from_pretrained="./ckpts/openai/clip-vit-large-patch14",
    max_length=77,
)
