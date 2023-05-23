import os
import tempfile


def Root():
    device = "cuda"
    models_path = "models/other"
    half_precision = True
    mask_preset_names = ['everywhere', 'video_mask']
    p = None
    frames_cache = []
    raw_batch_name = None
    raw_seed = None
    initial_seed = None
    initial_info = None
    first_frame = None
    outpath_samples = ""
    animation_prompts = None
    color_corrections = None
    initial_clipskip = None
    subseed = -1
    subseed_strength = 0
    init_sample = None
    #current_user_os = get_os()
    tmp_deforum_run_duplicated_folder = os.path.join(tempfile.gettempdir(), 'tmp_run_deforum')
    return locals()


def DeforumAnimArgs():
    animation_mode = '3D'  # ['None', '2D', '3D', 'Video Input', 'Interpolation']
    max_frames = 120
    border = 'replicate'  # ['wrap', 'replicate']
    angle = "0:(0)"
    zoom = "0:(1.0025+0.002*sin(1.25*3.14*t/30))"
    translation_x = "0:(0)"
    translation_y = "0:(0)"
    translation_z = "0:(1.75)"
    transform_center_x = "0:(0.5)"
    transform_center_y = "0:(0.5)"
    rotation_3d_x = "0:(0)"
    rotation_3d_y = "0:(0)"
    rotation_3d_z = "0:(0)"
    enable_perspective_flip = False
    perspective_flip_theta = "0:(0)"
    perspective_flip_phi = "0:(0)"
    perspective_flip_gamma = "0:(0)"
    perspective_flip_fv = "0:(53)"
    noise_schedule = "0: (0.065)"
    strength_schedule = "0: (0.65)"
    contrast_schedule = "0: (1.0)"
    cfg_scale_schedule = "0: (7)"
    enable_steps_scheduling = False
    steps_schedule = "0: (25)"
    fov_schedule = "0: (70)"
    aspect_ratio_schedule = "0: (1)"
    aspect_ratio_use_old_formula = False
    near_schedule = "0: (200)"
    far_schedule = "0: (10000)"
    seed_schedule = '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)'
    pix2pix_img_cfg_scale = "1.5"
    pix2pix_img_cfg_scale_schedule = "0:(1.5)"
    enable_subseed_scheduling = False
    subseed_schedule = "0:(1)"
    subseed_strength_schedule = "0:(0)"
    # Sampler Scheduling
    enable_sampler_scheduling = False
    sampler_schedule = '0: ("Euler a")'
    # Composable mask scheduling
    use_noise_mask = False
    mask_schedule = '0: ("{video_mask}")'
    noise_mask_schedule = '0: ("{video_mask}")'
    # Checkpoint Scheduling
    enable_checkpoint_scheduling = False
    checkpoint_schedule = '0: ("model1.ckpt"), 100: ("model2.safetensors")'
    # CLIP skip Scheduling
    enable_clipskip_scheduling = False
    clipskip_schedule = '0: (2)'
    # Noise Multiplier Scheduling
    enable_noise_multiplier_scheduling = True
    noise_multiplier_schedule = '0: (1.05)'
    # Anti-blur
    amount_schedule = "0: (0.1)"
    kernel_schedule = "0: (5)"
    sigma_schedule = "0: (1.0)"
    threshold_schedule = "0: (0.0)"
    # Hybrid video
    hybrid_comp_alpha_schedule = "0:(0.5)"
    hybrid_comp_mask_blend_alpha_schedule = "0:(0.5)"
    hybrid_comp_mask_contrast_schedule = "0:(1)"
    hybrid_comp_mask_auto_contrast_cutoff_high_schedule = "0:(100)"
    hybrid_comp_mask_auto_contrast_cutoff_low_schedule = "0:(0)"
    hybrid_flow_factor_schedule = "0:(1)"
    # Coherence
    color_coherence = 'LAB'  # ['None', 'HSV', 'LAB', 'RGB', 'Video Input', 'Image']
    color_coherence_image_path = ""
    color_coherence_video_every_N_frames = 1
    color_force_grayscale = False
    legacy_colormatch = False
    diffusion_cadence = '2'  # ['1','2','3','4','5','6','7','8']
    optical_flow_cadence = 'None'  # ['None', 'RAFT','DIS Medium', 'DIS Fine', 'Farneback']
    cadence_flow_factor_schedule = "0: (1)"
    optical_flow_redo_generation = 'None'  # ['None', 'RAFT', 'DIS Medium', 'DIS Fine', 'Farneback']
    redo_flow_factor_schedule = "0: (1)"
    diffusion_redo = '0'
    # **Noise settings:**
    noise_type = 'perlin'  # ['uniform', 'perlin']
    # Perlin params
    perlin_w = 8
    perlin_h = 8
    perlin_octaves = 4
    perlin_persistence = 0.5
    # **3D Depth Warping:**
    use_depth_warping = True
    depth_algorithm = 'Zoe' #  'Midas-3-Hybrid'  # ['Midas+AdaBins (old)','Zoe+AdaBins (old)', 'Midas-3-Hybrid','Midas-3.1-BeitLarge', 'AdaBins', 'Zoe', 'Leres'] Midas-3.1-BeitLarge is temporarily removed 04-05-23 until fixed
    midas_weight = 0.2  # midas/ zoe weight - only relevant in old/ legacy depth_algorithm modes. see above ^
    padding_mode = 'border'  # ['border', 'reflection', 'zeros']
    sampling_mode = 'bicubic'  # ['bicubic', 'bilinear', 'nearest']
    save_depth_maps = False
    # **Video Input:**
    video_init_path = 'https://deforum.github.io/a1/V1.mp4'
    extract_nth_frame = 1
    extract_from_frame = 0
    extract_to_frame = -1  # minus 1 for unlimited frames
    overwrite_extracted_frames = True
    use_mask_video = False
    video_mask_path = 'https://deforum.github.io/a1/VM1.mp4'
    # **Hybrid Video for 2D/3D Animation Mode:**
    hybrid_generate_inputframes = False
    hybrid_generate_human_masks = "None"  # ['None','PNGs','Video', 'Both']
    hybrid_use_first_frame_as_init_image = True
    hybrid_motion = "None"  # ['None','Optical Flow','Perspective','Affine']
    hybrid_motion_use_prev_img = False
    hybrid_flow_consistency = False
    hybrid_consistency_blur = 2
    hybrid_flow_method = "RAFT"  # ['RAFT', 'DIS Medium', 'DIS Fine', 'Farneback']
    hybrid_composite = 'None'  # ['None', 'Normal', 'Before Motion', 'After Generation']
    hybrid_use_init_image = False
    hybrid_comp_mask_type = "None"  # ['None', 'Depth', 'Video Depth', 'Blend', 'Difference']
    hybrid_comp_mask_inverse = False
    hybrid_comp_mask_equalize = "None"  # ['None','Before','After','Both']
    hybrid_comp_mask_auto_contrast = False
    hybrid_comp_save_extra_frames = False
    # **Resume Animation:**
    resume_from_timestring = False
    resume_timestring = "20230129210106"
    enable_ddim_eta_scheduling = False
    ddim_eta_schedule = "0:(0)"
    enable_ancestral_eta_scheduling = False
    ancestral_eta_schedule = "0:(1)"
    use_controlnet = False

    return locals()


def DeforumAnimPrompts():
    return r"""{
    "0": "tiny cute swamp bunny, highly detailed, intricate, ultra hd, sharp photo, crepuscular rays, in focus, by tomasz alen kopera",
    "30": "anthropomorphic clean cat, surrounded by fractals, epic angle and pose, symmetrical, 3d, depth of field, ruan jia and fenghua zhong",
    "60": "a beautiful coconut --neg photo, realistic",
    "90": "a beautiful durian, trending on Artstation"
}
    """


def DeforumArgs():
    # **Image Settings**
    W = 512  #
    H = 512  #
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64

    show_info_on_ui = True

    # **Webui stuff**
    tiling = False
    restore_faces = False
    seed_enable_extras = False
    seed_resize_from_w = 0
    seed_resize_from_h = 0

    # **Sampling Settings**
    seed = -1  #
    sampler = 'euler_ancestral'  # ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
    steps = 25  #
    scale = 7  #

    dynamic_threshold = None
    static_threshold = None

    # **Save & Display Settings**
    save_settings = True
    save_sample_per_step = False

    # **Prompt Settings**
    prompt_weighting = False
    normalize_prompt_weights = True
    log_weighted_subprompts = False

    # **Batch Settings**
    n_batch = 1  #
    batch_name = "Deforum_{timestring}"
    seed_behavior = "iter"  # ["iter","fixed","random","ladder","alternate","schedule"]
    seed_iter_N = 1
    outdir = "output/deforum"

    # **Init Settings**
    use_init = False
    strength = 0.8
    strength_0_no_init = True  # Set the strength to 0 automatically when no init image is used
    init_image = "https://deforum.github.io/a1/I1.png"
    # Whiter areas of the mask are areas that change more
    use_mask = False
    use_alpha_as_mask = False  # use the alpha channel of the init image as the mask
    mask_file = "https://deforum.github.io/a1/M1.jpg"
    invert_mask = False
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_contrast_adjust = 1.0
    mask_brightness_adjust = 1.0
    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = True
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = 4

    fill = 1  # MASKARGSEXPANSION Todo : Rename and convert to same formatting as used in img2img masked content
    full_res_mask = True
    full_res_mask_padding = 4
    reroll_blank_frames = 'reroll'  # reroll, interrupt, or ignore
    reroll_patience = 10

    n_samples = 1  # doesnt do anything
    precision = 'autocast'

    prompt = ""
    timestring = ""
    init_sample = None
    mask_image = None
    noise_mask = None
    seed_internal = 0

    return locals()


def keyframeExamples():
    return '''{
    "0": "https://deforum.github.io/a1/Gi1.png",
    "max_f/4-5": "https://deforum.github.io/a1/Gi2.png",
    "max_f/2-10": "https://deforum.github.io/a1/Gi3.png",
    "3*max_f/4-15": "https://deforum.github.io/a1/Gi4.jpg",
    "max_f-20": "https://deforum.github.io/a1/Gi1.png"
}'''


def LoopArgs():
    use_looper = False
    init_images = keyframeExamples()
    image_strength_schedule = "0:(0.75)"
    blendFactorMax = "0:(0.35)"
    blendFactorSlope = "0:(0.25)"
    tweening_frames_schedule = "0:(20)"
    color_correction_factor = "0:(0.075)"
    return locals()


def ParseqArgs():
    parseq_manifest = None
    parseq_use_deltas = True
    return locals()


def DeforumOutputArgs():
    skip_video_creation = False
    fps = 15
    make_gif = False
    delete_imgs = False  # True will delete all imgs after a successful mp4 creation
    image_path = "C:/SD/20230124234916_%09d.png"
    mp4_path = "testvidmanualsettings.mp4"
    add_soundtrack = 'None'  # ["File","Init Video"]
    soundtrack_path = "https://deforum.github.io/a1/A1.mp3"
    # End-Run upscaling
    r_upscale_video = False
    r_upscale_factor = 'x2'  # ['2x', 'x3', 'x4']
    r_upscale_model = 'realesr-animevideov3'  # 'realesr-animevideov3' (default of realesrgan engine, does 2-4x), the rest do only 4x: 'realesrgan-x4plus', 'realesrgan-x4plus-anime'
    r_upscale_keep_imgs = True

    store_frames_in_ram = False
    # **Interpolate Video Settings**
    frame_interpolation_engine = "None"  # ["None", "RIFE v4.6", "FILM"]
    frame_interpolation_x_amount = 2  # [2 to 1000 depends on the engine]
    frame_interpolation_slow_mo_enabled = False
    frame_interpolation_slow_mo_amount = 2  # [2 to 10]
    frame_interpolation_keep_imgs = False
    return locals()
