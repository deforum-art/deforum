import os
import sys

sys.path.extend([os.path.join(os.getcwd(), "deforum", "exttools")])

import pandas as pd
import cv2
import numpy as np
import numexpr
import gc
import random
import PIL

from PIL import Image, ImageOps

from deforum.animation.animation import anim_frame_warp
from deforum.animation.animation_key_frames import DeformAnimKeys, LooperAnimKeys
from deforum.avfunctions.colors.colors import maintain_colors
from deforum.exttools.depth import DepthModel
from deforum.avfunctions.hybridvideo.hybrid_video import hybrid_generation, get_flow_from_images, \
    image_transform_optical_flow, get_matrix_for_hybrid_motion_prev, image_transform_ransac, \
    get_matrix_for_hybrid_motion, get_flow_for_hybrid_motion_prev, get_flow_for_hybrid_motion, abs_flow_to_rel_flow, \
    rel_flow_to_abs_flow, hybrid_composite
from deforum.avfunctions.image.image_sharpening import unsharp_mask
from deforum.avfunctions.image.load_images import load_img, get_mask_from_file, get_mask, load_image
from deforum.avfunctions.image.save_images import save_image
from deforum.avfunctions.interpolation.RAFT import RAFT
from deforum.avfunctions.masks.composable_masks import compose_mask_with_check
from deforum.avfunctions.masks.masks import do_overlay_mask
from deforum.avfunctions.noise.noise import add_noise
from deforum.avfunctions.video_audio_utilities import get_next_frame, get_frame_name
from deforum.datafunctions.parseq_adapter import ParseqAnimKeys
from deforum.datafunctions.prompt import prepare_prompt
from deforum.datafunctions.resume import get_resume_vars
from deforum.datafunctions.seed import next_seed
from deforum.datafunctions.subtitle_handler import format_animation_params, write_frame_subtitle, init_srt_file
from deforum.general_utils import isJson



class Deforum:
    
    def __init__(self, args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root, opts=None, state=None):
        
        super().__init__()
        
        self.args = args
        self.anim_args = anim_args
        self.video_args = video_args
        self.parseq_args = parseq_args
        self.loop_args = loop_args
        self.controlnet_args = controlnet_args
        self.root = root
        self.opts = opts
        self.state = state
        
    def __call__(self, *args, **kwargs):
        if self.opts is not None:
            if self.opts.data.get("deforum_save_gen_info_as_srt",
                             False):  # create .srt file and set timeframe mechanism using FPS
                srt_filename = os.path.join(self.args.outdir, f"{self.root.timestring}.srt")
                srt_frame_duration = init_srt_file(srt_filename, self.video_args.fps)

        if self.anim_args.animation_mode in ['2D', '3D']:
            # handle hybrid video generation
            if self.anim_args.hybrid_composite != 'None' or self.anim_args.hybrid_motion in ['Affine', 'Perspective',
                                                                                   'Optical Flow']:
                self.args, self.anim_args, inputfiles = hybrid_generation(self.args, self.anim_args, self.root)
                # path required by hybrid functions, even if hybrid_comp_save_extra_frames is False
                hybrid_frame_path = os.path.join(self.args.outdir, 'hybridframes')
            # initialize prev_flow
            if self.anim_args.hybrid_motion == 'Optical Flow':
                prev_flow = None

            if self.loop_args.use_looper:
                print(
                    "Using Guided Images mode: seed_behavior will be set to 'schedule' and 'strength_0_no_init' to False")
                if self.args.strength == 0:
                    raise RuntimeError("Strength needs to be greater than 0 in Init tab")
                self.args.strength_0_no_init = False
                self.args.seed_behavior = "schedule"
                if not isJson(self.loop_args.init_images):
                    raise RuntimeError("The images set for use with keyframe-guidance are not in a proper JSON format")

        # handle controlnet video input frames generation
        self.handle_controlnet(self.args, self.anim_args, self.controlnet_args)
        #if is_controlnet_enabled(self.controlnet_args):
        #    unpack_controlnet_vids(self.args, self.anim_args, self.controlnet_args)

        # use parseq if manifest is provided
        if self.parseq_args is not None:
            use_parseq = self.parseq_args.parseq_manifest is not None and self.parseq_args.parseq_manifest.strip()
        else:
            use_parseq = False
        # expand key frame strings to values
        keys = DeformAnimKeys(self.anim_args, self.args.seed) if not use_parseq else ParseqAnimKeys(self.parseq_args, self.anim_args,
                                                                                          self.video_args)
        loopSchedulesAndData = LooperAnimKeys(self.loop_args, self.anim_args, self.args.seed)

        # create output folder for the batch
        os.makedirs(self.args.outdir, exist_ok=True)
        print(f"Saving animation frames to:\n{self.args.outdir}")

        # save settings.txt file for the current run
        self.save_settings_from_animation_run(self.args, self.anim_args, self.parseq_args, self.loop_args, self.controlnet_args, self.video_args, self.root)

        # resume from timestring
        if self.anim_args.resume_from_timestring:
            self.root.timestring = self.anim_args.resume_timestring

        # Always enable pseudo-3d with parseq. No need for an extra toggle:
        # Whether it's used or not in practice is defined by the schedules
        if use_parseq:
            self.anim_args.flip_2d_perspective = True

            # expand prompts out to per-frame
        if use_parseq and keys.manages_prompts():
            prompt_series = keys.prompts
        else:
            prompt_series = pd.Series([np.nan for a in range(self.anim_args.max_frames)])
            for i, prompt in self.root.animation_prompts.items():
                if str(i).isdigit():
                    prompt_series[int(i)] = prompt
                else:
                    prompt_series[int(numexpr.evaluate(i))] = prompt
            prompt_series = prompt_series.ffill().bfill()

        # check for video inits
        using_vid_init = self.anim_args.animation_mode == 'Video Input'

        # load depth model for 3D
        predict_depths = (self.anim_args.animation_mode == '3D' and self.anim_args.use_depth_warping) or self.anim_args.save_depth_maps
        predict_depths = predict_depths or (
                    self.anim_args.hybrid_composite and self.anim_args.hybrid_comp_mask_type in ['Depth', 'Video Depth'])
        if predict_depths:
            if self.opts is not None:
                self.keep_in_vram = self.opts.data.get("deforum_keep_3d_models_in_vram")
            else:
                self.keep_in_vram = True

            #device = ('cpu' if cmd_opts.lowvram or cmd_opts.medvram else self.root.device)
            #TODO Set device in root in webui
            device = self.root.device
            depth_model = DepthModel(self.root.models_path, device, self.root.half_precision, keep_in_vram=self.keep_in_vram,
                                     depth_algorithm=self.anim_args.depth_algorithm, Width=self.args.W, Height=self.args.H,
                                     midas_weight=self.anim_args.midas_weight)

            # depth-based hybrid composite mask requires saved depth maps
            if self.anim_args.hybrid_composite != 'None' and self.anim_args.hybrid_comp_mask_type == 'Depth':
                self.anim_args.save_depth_maps = True
        else:
            depth_model = None
            self.anim_args.save_depth_maps = False

        raft_model = None
        load_raft = (self.anim_args.optical_flow_cadence == "RAFT" and int(self.anim_args.diffusion_cadence) > 1) or \
                    (self.anim_args.hybrid_motion == "Optical Flow" and self.anim_args.hybrid_flow_method == "RAFT") or \
                    (self.anim_args.optical_flow_redo_generation == "RAFT")
        if load_raft:
            print("Loading RAFT model...")
            raft_model = RAFT()

        # state for interpolating between diffusion steps
        turbo_steps = 1 if using_vid_init else int(self.anim_args.diffusion_cadence)
        turbo_prev_image, turbo_prev_frame_idx = None, 0
        turbo_next_image, turbo_next_frame_idx = None, 0

        # initialize vars
        prev_img = None
        color_match_sample = None
        start_frame = 0

        # resume animation (requires at least two frames - see function)
        if self.anim_args.resume_from_timestring:
            # determine last frame and frame to start on
            prev_frame, next_frame, prev_img, next_img = get_resume_vars(
                folder=self.args.outdir,
                timestring=self.anim_args.resume_timestring,
                cadence=turbo_steps
            )

            # set up turbo step vars
            if turbo_steps > 1:
                turbo_prev_image, turbo_prev_frame_idx = prev_img, prev_frame
                turbo_next_image, turbo_next_frame_idx = next_img, next_frame

            # advance start_frame to next frame
            start_frame = next_frame + 1

        frame_idx = start_frame

        # reset the mask vals as they are overwritten in the compose_mask algorithm
        mask_vals = {}
        noise_mask_vals = {}

        mask_vals['everywhere'] = Image.new('1', (self.args.W, self.args.H), 1)
        noise_mask_vals['everywhere'] = Image.new('1', (self.args.W, self.args.H), 1)

        mask_image = None

        if self.args.use_init and self.args.init_image != None and self.args.init_image != '':
            _, mask_image = load_img(self.args.init_image,
                                     shape=(self.args.W, self.args.H),
                                     use_alpha_as_mask=self.args.use_alpha_as_mask)
            mask_vals['video_mask'] = mask_image
            noise_mask_vals['video_mask'] = mask_image

        # Grab the first frame masks since they wont be provided until next frame
        # Video mask overrides the init image mask, also, won't be searching for init_mask if use_mask_video is set
        # Made to solve https://github.com/deforum-art/deforum-for-automatic1111-webui/issues/386
        if self.anim_args.use_mask_video:

            self.args.mask_file = get_mask_from_file(get_next_frame(self.args.outdir, self.anim_args.video_mask_path, frame_idx, True),
                                                self.args)
            self.root.noise_mask = get_mask_from_file(
                get_next_frame(self.args.outdir, self.anim_args.video_mask_path, frame_idx, True), self.args)

            mask_vals['video_mask'] = get_mask_from_file(
                get_next_frame(self.args.outdir, self.anim_args.video_mask_path, frame_idx, True), self.args)
            noise_mask_vals['video_mask'] = get_mask_from_file(
                get_next_frame(self.args.outdir, self.anim_args.video_mask_path, frame_idx, True), self.args)
        elif mask_image is None and self.args.use_mask:
            mask_vals['video_mask'] = get_mask(self.args)
            noise_mask_vals['video_mask'] = get_mask(self.args)  # TODO?: add a different default noisc mask

        # get color match for 'Image' color coherence only once, before loop
        if self.anim_args.color_coherence == 'Image':
            color_match_sample = load_image(self.anim_args.color_coherence_image_path)
            color_match_sample = color_match_sample.resize((self.args.W, self.args.H), PIL.Image.LANCZOS)
            color_match_sample = cv2.cvtColor(np.array(color_match_sample), cv2.COLOR_RGB2BGR)

        # Webui
        done = self.datacallback({"max_frames":self.anim_args.max_frames})

        #state.job_count = self.anim_args.max_frames

        while frame_idx < self.anim_args.max_frames:
            # Webui

            done = self.datacallback({"job":f"frame {frame_idx + 1}/{self.anim_args.max_frames}",
                               "job_no":frame_idx + 1})
            #state.job = f"frame {frame_idx + 1}/{self.anim_args.max_frames}"
            #state.job_no = frame_idx + 1

            #if state.skipped:
            #    print("\n** PAUSED **")
            #    state.skipped = False
            #    while not state.skipped:
            #        time.sleep(0.1)
            #    print("** RESUMING **")

            print(f"\033[36mAnimation frame: \033[0m{frame_idx}/{self.anim_args.max_frames}  ")

            noise = keys.noise_schedule_series[frame_idx]
            strength = keys.strength_schedule_series[frame_idx]
            scale = keys.cfg_scale_schedule_series[frame_idx]
            contrast = keys.contrast_schedule_series[frame_idx]
            kernel = int(keys.kernel_schedule_series[frame_idx])
            sigma = keys.sigma_schedule_series[frame_idx]
            amount = keys.amount_schedule_series[frame_idx]
            threshold = keys.threshold_schedule_series[frame_idx]
            cadence_flow_factor = keys.cadence_flow_factor_schedule_series[frame_idx]
            redo_flow_factor = keys.redo_flow_factor_schedule_series[frame_idx]
            hybrid_comp_schedules = {
                "alpha": keys.hybrid_comp_alpha_schedule_series[frame_idx],
                "mask_blend_alpha": keys.hybrid_comp_mask_blend_alpha_schedule_series[frame_idx],
                "mask_contrast": keys.hybrid_comp_mask_contrast_schedule_series[frame_idx],
                "mask_auto_contrast_cutoff_low": int(
                    keys.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series[frame_idx]),
                "mask_auto_contrast_cutoff_high": int(
                    keys.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series[frame_idx]),
                "flow_factor": keys.hybrid_flow_factor_schedule_series[frame_idx]
            }
            scheduled_sampler_name = None
            scheduled_clipskip = None
            scheduled_noise_multiplier = None
            scheduled_ddim_eta = None
            scheduled_ancestral_eta = None

            mask_seq = None
            noise_mask_seq = None
            if self.anim_args.enable_steps_scheduling and keys.steps_schedule_series[frame_idx] is not None:
                self.args.steps = int(keys.steps_schedule_series[frame_idx])
            if self.anim_args.enable_sampler_scheduling and keys.sampler_schedule_series[frame_idx] is not None:
                scheduled_sampler_name = keys.sampler_schedule_series[frame_idx].casefold()
            if self.anim_args.enable_clipskip_scheduling and keys.clipskip_schedule_series[frame_idx] is not None:
                scheduled_clipskip = int(keys.clipskip_schedule_series[frame_idx])
            if self.anim_args.enable_noise_multiplier_scheduling and keys.noise_multiplier_schedule_series[
                frame_idx] is not None:
                scheduled_noise_multiplier = float(keys.noise_multiplier_schedule_series[frame_idx])
            if self.anim_args.enable_ddim_eta_scheduling and keys.ddim_eta_schedule_series[frame_idx] is not None:
                scheduled_ddim_eta = float(keys.ddim_eta_schedule_series[frame_idx])
            if self.anim_args.enable_ancestral_eta_scheduling and keys.ancestral_eta_schedule_series[frame_idx] is not None:
                scheduled_ancestral_eta = float(keys.ancestral_eta_schedule_series[frame_idx])
            if self.args.use_mask and keys.mask_schedule_series[frame_idx] is not None:
                mask_seq = keys.mask_schedule_series[frame_idx]
            if self.anim_args.use_noise_mask and keys.noise_mask_schedule_series[frame_idx] is not None:
                noise_mask_seq = keys.noise_mask_schedule_series[frame_idx]

            if self.args.use_mask and not self.anim_args.use_noise_mask:
                noise_mask_seq = mask_seq

            depth = None
            done = self.datacallback({"webui":"sd_to_cpu"})
            #if self.anim_args.animation_mode == '3D' and (cmd_opts.lowvram or cmd_opts.medvram):
            #    # Unload the main checkpoint and load the depth model
            #    lowvram.send_everything_to_cpu()
            #    sd_hijack.model_hijack.undo_hijack(sd_model)
            #    devices.torch_gc()
            #
            #    if predict_depths: depth_model.to(self.root.device)
            if self.anim_args.animation_mode == '3D':
                if predict_depths: depth_model.to(self.root.device)
            if self.opts is not None:
                if turbo_steps == 1 and self.opts.data.get("deforum_save_gen_info_as_srt"):
                    params_string = format_animation_params(keys, prompt_series, frame_idx)
                    write_frame_subtitle(srt_filename, frame_idx, srt_frame_duration,
                                         f"F#: {frame_idx}; Cadence: false; Seed: {self.args.seed}; {params_string}")
                    params_string = None

            # emit in-between frames
            if turbo_steps > 1:
                tween_frame_start_idx = max(start_frame, frame_idx - turbo_steps)
                cadence_flow = None
                for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                    # update progress during cadence
                    done = self.datacallback({"job": f"frame {tween_frame_idx + 1}/{self.anim_args.max_frames}",
                                       "job_no": tween_frame_idx + 1})
                    #state.job = f"frame {tween_frame_idx + 1}/{self.anim_args.max_frames}"
                    #state.job_no = tween_frame_idx + 1
                    # cadence vars
                    tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(
                        frame_idx - tween_frame_start_idx)
                    advance_prev = turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx
                    advance_next = tween_frame_idx > turbo_next_frame_idx

                    # optical flow cadence setup before animation warping
                    if self.anim_args.animation_mode in ['2D', '3D'] and self.anim_args.optical_flow_cadence != 'None':
                        if keys.strength_schedule_series[tween_frame_start_idx] > 0:
                            if cadence_flow is None and turbo_prev_image is not None and turbo_next_image is not None:
                                cadence_flow = get_flow_from_images(turbo_prev_image, turbo_next_image,
                                                                    self.anim_args.optical_flow_cadence, raft_model) / 2
                                turbo_next_image = image_transform_optical_flow(turbo_next_image, -cadence_flow, 1)
                    if self.opts is not None:
                        if self.opts.data.get("deforum_save_gen_info_as_srt"):
                            params_string = format_animation_params(keys, prompt_series, tween_frame_idx)
                            write_frame_subtitle(srt_filename, tween_frame_idx, srt_frame_duration,
                                                 f"F#: {tween_frame_idx}; Cadence: {tween < 1.0}; Seed: {self.args.seed}; {params_string}")
                            params_string = None

                    print(
                        f"Creating in-between {'' if cadence_flow is None else self.anim_args.optical_flow_cadence + ' optical flow '}cadence frame: {tween_frame_idx}; tween:{tween:0.2f};")

                    if depth_model is not None:
                        assert (turbo_next_image is not None)
                        depth = depth_model.predict(turbo_next_image, self.anim_args.midas_weight, self.root.half_precision)

                    if advance_prev:
                        turbo_prev_image, _ = anim_frame_warp(turbo_prev_image, self.args, self.anim_args, keys, tween_frame_idx,
                                                              depth_model, depth=depth, device=self.root.device,
                                                              half_precision=self.root.half_precision)
                    if advance_next:
                        turbo_next_image, _ = anim_frame_warp(turbo_next_image, self.args, self.anim_args, keys, tween_frame_idx,
                                                              depth_model, depth=depth, device=self.root.device,
                                                              half_precision=self.root.half_precision)

                    # hybrid video motion - warps turbo_prev_image or turbo_next_image to match motion
                    if tween_frame_idx > 0:
                        if self.anim_args.hybrid_motion in ['Affine', 'Perspective']:
                            if self.anim_args.hybrid_motion_use_prev_img:
                                matrix = get_matrix_for_hybrid_motion_prev(tween_frame_idx - 1, (self.args.W, self.args.H),
                                                                           inputfiles, prev_img,
                                                                           self.anim_args.hybrid_motion)
                                if advance_prev:
                                    turbo_prev_image = image_transform_ransac(turbo_prev_image, matrix,
                                                                              self.anim_args.hybrid_motion)
                                if advance_next:
                                    turbo_next_image = image_transform_ransac(turbo_next_image, matrix,
                                                                              self.anim_args.hybrid_motion)
                            else:
                                matrix = get_matrix_for_hybrid_motion(tween_frame_idx - 1, (self.args.W, self.args.H), inputfiles,
                                                                      self.anim_args.hybrid_motion)
                                if advance_prev:
                                    turbo_prev_image = image_transform_ransac(turbo_prev_image, matrix,
                                                                              self.anim_args.hybrid_motion)
                                if advance_next:
                                    turbo_next_image = image_transform_ransac(turbo_next_image, matrix,
                                                                              self.anim_args.hybrid_motion)
                        if self.anim_args.hybrid_motion in ['Optical Flow']:
                            if self.anim_args.hybrid_motion_use_prev_img:
                                flow = get_flow_for_hybrid_motion_prev(tween_frame_idx - 1, (self.args.W, self.args.H),
                                                                       inputfiles, hybrid_frame_path, prev_flow,
                                                                       prev_img, self.anim_args.hybrid_flow_method,
                                                                       raft_model,
                                                                       self.anim_args.hybrid_flow_consistency,
                                                                       self.anim_args.hybrid_consistency_blur,
                                                                       self.anim_args.hybrid_comp_save_extra_frames)
                                if advance_prev:
                                    turbo_prev_image = image_transform_optical_flow(turbo_prev_image, flow,
                                                                                    hybrid_comp_schedules[
                                                                                        'flow_factor'])
                                if advance_next:
                                    turbo_next_image = image_transform_optical_flow(turbo_next_image, flow,
                                                                                    hybrid_comp_schedules[
                                                                                        'flow_factor'])
                                prev_flow = flow
                            else:
                                flow = get_flow_for_hybrid_motion(tween_frame_idx - 1, (self.args.W, self.args.H), inputfiles,
                                                                  hybrid_frame_path, prev_flow,
                                                                  self.anim_args.hybrid_flow_method, raft_model,
                                                                  self.anim_args.hybrid_flow_consistency,
                                                                  self.anim_args.hybrid_consistency_blur,
                                                                  self.anim_args.hybrid_comp_save_extra_frames)
                                if advance_prev:
                                    turbo_prev_image = image_transform_optical_flow(turbo_prev_image, flow,
                                                                                    hybrid_comp_schedules[
                                                                                        'flow_factor'])
                                if advance_next:
                                    turbo_next_image = image_transform_optical_flow(turbo_next_image, flow,
                                                                                    hybrid_comp_schedules[
                                                                                        'flow_factor'])
                                prev_flow = flow

                    # do optical flow cadence after animation warping
                    if cadence_flow is not None:
                        cadence_flow = abs_flow_to_rel_flow(cadence_flow, self.args.W, self.args.H)
                        cadence_flow, _ = anim_frame_warp(cadence_flow, self.args, self.anim_args, keys, tween_frame_idx,
                                                          depth_model, depth=depth, device=self.root.device,
                                                          half_precision=self.root.half_precision)
                        cadence_flow_inc = rel_flow_to_abs_flow(cadence_flow, self.args.W, self.args.H) * tween
                        if advance_prev:
                            turbo_prev_image = image_transform_optical_flow(turbo_prev_image, cadence_flow_inc,
                                                                            cadence_flow_factor)
                        if advance_next:
                            turbo_next_image = image_transform_optical_flow(turbo_next_image, cadence_flow_inc,
                                                                            cadence_flow_factor)

                    turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx

                    if turbo_prev_image is not None and tween < 1.0:
                        img = turbo_prev_image * (1.0 - tween) + turbo_next_image * tween
                    else:
                        img = turbo_next_image

                    # intercept and override to grayscale
                    if self.anim_args.color_force_grayscale:
                        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                        # overlay mask
                    if self.args.overlay_mask and (self.anim_args.use_mask_video or self.args.use_mask):
                        img = do_overlay_mask(self.args, self.anim_args, img, tween_frame_idx, True)

                    # get prev_img during cadence
                    prev_img = img

                    # current image update for cadence frames (left commented because it doesn't currently update the preview)
                    # state.current_image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))

                    # saving cadence frames
                    filename = f"{self.root.timestring}_{tween_frame_idx:09}.png"
                    im = img.copy()
                    self.datacallback({"cadence_frame":Image.fromarray(cv2.cvtColor(im.astype("uint8"), cv2.COLOR_BGR2RGB))})

                    cv2.imwrite(os.path.join(self.args.outdir, filename), img)
                    if self.anim_args.save_depth_maps:
                        depth_model.save(os.path.join(self.args.outdir, f"{self.root.timestring}_depth_{tween_frame_idx:09}.png"),
                                         depth)

            # get color match for video outside of prev_img conditional
            hybrid_available = self.anim_args.hybrid_composite != 'None' or self.anim_args.hybrid_motion in ['Optical Flow',
                                                                                                   'Affine',
                                                                                                   'Perspective']
            if self.anim_args.color_coherence == 'Video Input' and hybrid_available:
                if int(frame_idx) % int(self.anim_args.color_coherence_video_every_N_frames) == 0:
                    prev_vid_img = Image.open(os.path.join(self.args.outdir, 'inputframes', get_frame_name(
                        self.anim_args.video_init_path) + f"{frame_idx:09}.jpg"))
                    prev_vid_img = prev_vid_img.resize((self.args.W, self.args.H), PIL.Image.LANCZOS)
                    color_match_sample = np.asarray(prev_vid_img)
                    color_match_sample = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2BGR)

            # after 1st frame, prev_img exists
            if prev_img is not None:
                # apply transforms to previous frame
                prev_img, depth = anim_frame_warp(prev_img, self.args, self.anim_args, keys, frame_idx, depth_model, depth=None,
                                                  device=self.root.device, half_precision=self.root.half_precision)

                # do hybrid compositing before motion
                if self.anim_args.hybrid_composite == 'Before Motion':
                    self.args, prev_img = hybrid_composite(self.args, self.anim_args, frame_idx, prev_img, depth_model,
                                                      hybrid_comp_schedules, self.root)

                # hybrid video motion - warps prev_img to match motion, usually to prepare for compositing
                if self.anim_args.hybrid_motion in ['Affine', 'Perspective']:
                    if self.anim_args.hybrid_motion_use_prev_img:
                        matrix = get_matrix_for_hybrid_motion_prev(frame_idx - 1, (self.args.W, self.args.H), inputfiles,
                                                                   prev_img, self.anim_args.hybrid_motion)
                    else:
                        matrix = get_matrix_for_hybrid_motion(frame_idx - 1, (self.args.W, self.args.H), inputfiles,
                                                              self.anim_args.hybrid_motion)
                    prev_img = image_transform_ransac(prev_img, matrix, self.anim_args.hybrid_motion)
                if self.anim_args.hybrid_motion in ['Optical Flow']:
                    if self.anim_args.hybrid_motion_use_prev_img:
                        flow = get_flow_for_hybrid_motion_prev(frame_idx - 1, (self.args.W, self.args.H), inputfiles,
                                                               hybrid_frame_path, prev_flow, prev_img,
                                                               self.anim_args.hybrid_flow_method, raft_model,
                                                               self.anim_args.hybrid_flow_consistency,
                                                               self.anim_args.hybrid_consistency_blur,
                                                               self.anim_args.hybrid_comp_save_extra_frames)
                    else:
                        flow = get_flow_for_hybrid_motion(frame_idx - 1, (self.args.W, self.args.H), inputfiles,
                                                          hybrid_frame_path, prev_flow, self.anim_args.hybrid_flow_method,
                                                          raft_model,
                                                          self.anim_args.hybrid_flow_consistency,
                                                          self.anim_args.hybrid_consistency_blur,
                                                          self.anim_args.hybrid_comp_save_extra_frames)
                    prev_img = image_transform_optical_flow(prev_img, flow, hybrid_comp_schedules['flow_factor'])
                    prev_flow = flow

                # do hybrid compositing after motion (normal)
                if self.anim_args.hybrid_composite == 'Normal':
                    self.args, prev_img = hybrid_composite(self.args, self.anim_args, frame_idx, prev_img, depth_model,
                                                      hybrid_comp_schedules, self.root)

                # apply color matching
                if self.anim_args.color_coherence != 'None':
                    if color_match_sample is None:
                        color_match_sample = prev_img.copy()
                    else:
                        prev_img = maintain_colors(prev_img, color_match_sample, self.anim_args.color_coherence)

                # intercept and override to grayscale
                if self.anim_args.color_force_grayscale:
                    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
                    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_GRAY2BGR)

                # apply scaling
                contrast_image = (prev_img * contrast).round().astype(np.uint8)
                # anti-blur
                if amount > 0:
                    contrast_image = unsharp_mask(contrast_image, (kernel, kernel), sigma, amount, threshold,
                                                  mask_image if self.args.use_mask else None)
                # apply frame noising
                if self.args.use_mask or self.anim_args.use_noise_mask:
                    self.root.noise_mask = compose_mask_with_check(self.root, self.args, noise_mask_seq, noise_mask_vals,
                                                              Image.fromarray(
                                                                  cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB)))
                noised_image = add_noise(contrast_image, noise, self.args.seed, self.anim_args.noise_type,
                                         (self.anim_args.perlin_w, self.anim_args.perlin_h, self.anim_args.perlin_octaves,
                                          self.anim_args.perlin_persistence),
                                         self.root.noise_mask, self.args.invert_mask)

                # use transformed previous frame as init for current
                self.args.use_init = True
                self.root.init_sample = Image.fromarray(cv2.cvtColor(noised_image, cv2.COLOR_BGR2RGB))
                print("SETTING INIT SAMPLE #1")
                self.args.strength = max(0.0, min(1.0, strength))

            self.args.scale = scale

            # Pix2Pix Image CFG Scale - does *nothing* with non pix2pix checkpoints
            self.args.pix2pix_img_cfg_scale = float(keys.pix2pix_img_cfg_scale_series[frame_idx])

            # grab prompt for current frame
            self.args.prompt = prompt_series[frame_idx]

            if self.args.seed_behavior == 'schedule' or use_parseq:
                self.args.seed = int(keys.seed_schedule_series[frame_idx])

            if self.anim_args.enable_checkpoint_scheduling:
                self.args.checkpoint = keys.checkpoint_schedule_series[frame_idx]
            else:
                self.args.checkpoint = None

            # SubSeed scheduling
            if self.anim_args.enable_subseed_scheduling:
                self.root.subseed = int(keys.subseed_schedule_series[frame_idx])
                self.root.subseed_strength = float(keys.subseed_strength_schedule_series[frame_idx])

            if use_parseq:
                self.anim_args.enable_subseed_scheduling = True
                self.root.subseed = int(keys.subseed_schedule_series[frame_idx])
                self.root.subseed_strength = keys.subseed_strength_schedule_series[frame_idx]

            # set value back into the prompt - prepare and report prompt and seed
            self.args.prompt = prepare_prompt(self.args.prompt, self.anim_args.max_frames, self.args.seed, frame_idx)

            # grab init image for current frame
            if using_vid_init:
                init_frame = get_next_frame(self.args.outdir, self.anim_args.video_init_path, frame_idx, False)
                print(f"Using video init frame {init_frame}")
                self.args.init_image = init_frame
                self.args.strength = max(0.0, min(1.0, strength))
            if self.anim_args.use_mask_video:
                self.args.mask_file = get_mask_from_file(
                    get_next_frame(self.args.outdir, self.anim_args.video_mask_path, frame_idx, True), self.args)
                self.root.noise_mask = get_mask_from_file(
                    get_next_frame(self.args.outdir, self.anim_args.video_mask_path, frame_idx, True), self.args)

                mask_vals['video_mask'] = get_mask_from_file(
                    get_next_frame(self.args.outdir, self.anim_args.video_mask_path, frame_idx, True), self.args)

            if self.args.use_mask:
                self.args.mask_image = compose_mask_with_check(self.root, self.args, mask_seq, mask_vals,
                                                          self.root.init_sample) if self.root.init_sample is not None else None  # we need it only after the first frame anyway

            # setting up some arguments for the looper
            self.loop_args.imageStrength = loopSchedulesAndData.image_strength_schedule_series[frame_idx]
            self.loop_args.blendFactorMax = loopSchedulesAndData.blendFactorMax_series[frame_idx]
            self.loop_args.blendFactorSlope = loopSchedulesAndData.blendFactorSlope_series[frame_idx]
            self.loop_args.tweeningFrameSchedule = loopSchedulesAndData.tweening_frames_schedule_series[frame_idx]
            self.loop_args.colorCorrectionFactor = loopSchedulesAndData.color_correction_factor_series[frame_idx]
            self.loop_args.use_looper = loopSchedulesAndData.use_looper
            self.loop_args.imagesToKeyframe = loopSchedulesAndData.imagesToKeyframe
            if self.opts is not None:
                if 'img2img_fix_steps' in self.opts.data and self.opts.data[
                    "img2img_fix_steps"]:  # disable "with img2img do exactly x steps" from general setting, as it *ruins* deforum animations
                    self.opts.data["img2img_fix_steps"] = False
                if scheduled_clipskip is not None:
                    self.opts.data["CLIP_stop_at_last_layers"] = scheduled_clipskip
                if scheduled_noise_multiplier is not None:
                    self.opts.data["initial_noise_multiplier"] = scheduled_noise_multiplier
                if scheduled_ddim_eta is not None:
                    self.opts.data["eta_ddim"] = scheduled_ddim_eta
                if scheduled_ancestral_eta is not None:
                    self.opts.data["eta_ancestral"] = scheduled_ancestral_eta
            self.datacallback({"webui":"sd_to_gpu"})
            #if self.anim_args.animation_mode == '3D' and (cmd_opts.lowvram or cmd_opts.medvram):
            #    if predict_depths: depth_model.to('cpu')
            #    devices.torch_gc()
            #    lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
            #    sd_hijack.model_hijack.hijack(sd_model)

            # optical flow redo before generation
            if self.anim_args.optical_flow_redo_generation != 'None' and prev_img is not None and strength > 0:
                print(
                    f"Optical flow redo is diffusing and warping using {self.anim_args.optical_flow_redo_generation} optical flow before generation.")
                stored_seed = self.args.seed
                self.args.seed = random.randint(0, 2 ** 32 - 1)
                disposable_image = self.generate(self.args, keys, self.anim_args, self.loop_args, self.controlnet_args, self.root, frame_idx,
                                            sampler_name=scheduled_sampler_name)
                disposable_image = cv2.cvtColor(np.array(disposable_image), cv2.COLOR_RGB2BGR)
                disposable_flow = get_flow_from_images(prev_img, disposable_image,
                                                       self.anim_args.optical_flow_redo_generation, raft_model)
                disposable_image = cv2.cvtColor(disposable_image, cv2.COLOR_BGR2RGB)
                disposable_image = image_transform_optical_flow(disposable_image, disposable_flow, redo_flow_factor)
                self.args.seed = stored_seed
                print("SETTING INIT SAMPLE #2")

                self.root.init_sample = Image.fromarray(disposable_image)
                del (disposable_image, disposable_flow, stored_seed)
                gc.collect()

            # diffusion redo
            if int(self.anim_args.diffusion_redo) > 0 and prev_img is not None and strength > 0:
                stored_seed = self.args.seed
                for n in range(0, int(self.anim_args.diffusion_redo)):
                    print(f"Redo generation {n + 1} of {int(self.anim_args.diffusion_redo)} before final generation")
                    self.args.seed = random.randint(0, 2 ** 32 - 1)
                    disposable_image = self.generate(self.args, keys, self.anim_args, self.loop_args, self.controlnet_args, self.root, frame_idx,
                                                sampler_name=scheduled_sampler_name)
                    disposable_image = cv2.cvtColor(np.array(disposable_image), cv2.COLOR_RGB2BGR)
                    # color match on last one only
                    if n == int(self.anim_args.diffusion_redo):
                        disposable_image = maintain_colors(prev_img, color_match_sample, self.anim_args.color_coherence)
                    self.args.seed = stored_seed
                    print("SETTING INIT SAMPLE #3")

                    self.root.init_sample = Image.fromarray(cv2.cvtColor(disposable_image, cv2.COLOR_BGR2RGB))
                del (disposable_image, stored_seed)
                gc.collect()

            # generation
            image = self.generate(self.args, keys, self.anim_args, self.loop_args, self.controlnet_args, self.root, frame_idx,
                             sampler_name=scheduled_sampler_name)

            if image is None:
                break

            # do hybrid video after generation
            if frame_idx > 0 and self.anim_args.hybrid_composite == 'After Generation':
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                self.args, image = hybrid_composite(self.args, self.anim_args, frame_idx, image, depth_model, hybrid_comp_schedules,
                                               self.root)
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # color matching on first frame is after generation, color match was collected earlier, so we do an extra generation to avoid the corruption introduced by the color match of first output
            if frame_idx == 0 and (self.anim_args.color_coherence == 'Image' or (
                    self.anim_args.color_coherence == 'Video Input' and hybrid_available)):
                image = maintain_colors(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), color_match_sample,
                                        self.anim_args.color_coherence)
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif color_match_sample is not None and self.anim_args.color_coherence != 'None' and not self.anim_args.legacy_colormatch:
                image = maintain_colors(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), color_match_sample,
                                        self.anim_args.color_coherence)
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # intercept and override to grayscale
            if self.anim_args.color_force_grayscale:
                image = ImageOps.grayscale(image)
                image = ImageOps.colorize(image, black="black", white="white")

            # overlay mask
            if self.args.overlay_mask and (self.anim_args.use_mask_video or self.args.use_mask):
                image = do_overlay_mask(self.args, self.anim_args, image, frame_idx)

            # on strength 0, set color match to generation
            if ((not self.anim_args.legacy_colormatch and not self.args.use_init) or (
                    self.anim_args.legacy_colormatch and strength == 0)) and not self.anim_args.color_coherence in ['Image',
                                                                                                          'Video Input']:
                color_match_sample = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            if not using_vid_init:
                prev_img = opencv_image

            if turbo_steps > 1:
                turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
                turbo_next_image, turbo_next_frame_idx = opencv_image, frame_idx
                frame_idx += turbo_steps
            else:
                filename = f"{self.root.timestring}_{frame_idx:09}.png"
                save_image(image, 'PIL', filename, self.args, self.video_args, self.root)

                if self.anim_args.save_depth_maps:
                    done = self.datacallback({"webui":"sd_to_cpu"})
                    #if cmd_opts.lowvram or cmd_opts.medvram:
                    #    lowvram.send_everything_to_cpu()
                    #    sd_hijack.model_hijack.undo_hijack(sd_model)
                    #    devices.torch_gc()
                    #    depth_model.to(self.root.device)
                    depth = depth_model.predict(opencv_image, self.anim_args.midas_weight, self.root.half_precision)
                    depth_model.save(os.path.join(self.args.outdir, f"{self.root.timestring}_depth_{frame_idx:09}.png"), depth)
                    done = self.datacallback({"webui":"sd_to_cpu"})

                    #if cmd_opts.lowvram or cmd_opts.medvram:
                    #    depth_model.to('cpu')
                    #    devices.torch_gc()
                    #    lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
                    #    sd_hijack.model_hijack.hijack(sd_model)
                frame_idx += 1
            done = self.datacallback({"image": image})

            #state.current_image = image

            self.args.seed = next_seed(self.args, self.root)

        if predict_depths and not self.keep_in_vram:
            depth_model.delete_model()  # handles adabins too

        if load_raft:
            raft_model.delete_model()
        return True

    def datacallback(self, data):
        return None

    def generate(self, args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name):
        return None

    def save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root,
                                         full_out_file_path=None):
        return

    def handle_controlnet(self, args, anim_args, controlnet_args):
        print("DEFORUM DUMMY CONTROLNET HANDLER, REPLACE ME WITH YOUR UI's, or API's CONTROLNET HANDLER")
        return None