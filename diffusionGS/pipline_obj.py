import os
import warnings
from typing import Callable, List, Optional, Union, Dict, Any
import PIL.Image
import trimesh
import rembg
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from diffusers.utils import BaseOutput
import torch.nn.functional as F
import diffusionGS
from diffusionGS.utils.config import ExperimentConfig, load_config
from diffusionGS.systems.utils import *
from easydict import EasyDict as edict
from collections import OrderedDict
from diffusionGS.models.gsrenderer.gs_core import render_opencv_cam
class GSPipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[trimesh.Trimesh]` or `np.ndarray`)
            List of denoised trimesh meshes of length `batch_size` or a tuple of NumPy array with shape `((vertices, 3), (faces, 3)) of length `batch_size``.
    """

    gaussians: Optional[torch.Tensor] = None
    render_images: Optional[torch.Tensor] = None

class DiffusionGSPipeline():
    """
    Args:
        feature_extractor ([`CLIPFeatureExtractor`]):
            Feature extractor for image pre-processing before being encoded.
    """
    def __init__(
        self,
        device: str,
        cfg: ExperimentConfig,
        system,
    ):
        self.device = device
        self.cfg = cfg
        self.system = system

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        A simpler version that instantiate a PyTorch diffusion pipeline from pretrained pipeline weights.
        The pipeline is set in evaluation mode (`model.eval()`) by default.
        """
        # 1. Download the checkpoints and configshf download CaiYuanhao/DiffusionGS
        # use snapshot download here to get it working from from_pretrained
        if not os.path.isdir(pretrained_model_name_or_path):
            ckpt_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="obj_ckpt_512.ckpt", repo_type="model")
            config_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="obj_configs.yaml", repo_type="model")
            cam_template_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="camera_template.pt", repo_type="model")
        else:
            ckpt_path = os.path.join(pretrained_model_name_or_path, "obj_ckpt_512.ckpt")
            config_path = os.path.join(pretrained_model_name_or_path, "obj_configs.yaml")
            cam_template_path = os.path.join(pretrained_model_name_or_path, "camera_template.pt")

        # 2. Load the model
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        cfg = load_config(config_path)
        system = diffusionGS.find(cfg.system_type)(cfg.system)
        system.cam_template = torch.load(cam_template_path)
        print(f"Restoring states from the checkpoint path at {ckpt_path} with config {cfg}")
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        system.load_state_dict(
            ckpt["state_dict"] if "state_dict" in ckpt else ckpt,
        )
        system = system.to(device).eval()
        if 'torch_dtype' in kwargs:
            if kwargs['torch_dtype'] == torch.bfloat16:
                system.shape_model = system.shape_model.to(torch.float16) # shape vae only support fp16
            else:
                system = system.to(kwargs['torch_dtype'])

        return cls(
            device=device,
            cfg=cfg,
            system=system
        )

    def check_inputs(
        self,
        image,
    ):
        r"""
        Check if the inputs are valid. Raise an error if not.
        """
        if isinstance(image, str):
            assert os.path.isfile(image) or image.startswith("http"), "Input image must be a valid URL or a file path."
        elif not isinstance(image, (torch.Tensor, PIL.Image.Image)):
            raise ValueError("Input image must be a `torch.Tensor` or `PIL.Image.Image`.")
        
    def preprocess_image(
        self,
        images_pil: List[PIL.Image.Image],
        force: bool = False,
        background_color: List[int] = [255, 255, 255],
        foreground_ratio: float = 1.0,
    ):
        r"""
        Crop and remote the background of the input image
        Args:
            image_pil (`List[PIL.Image.Image]`):
                List of `PIL.Image.Image` objects representing the input image.
            force (`bool`, *optional*, defaults to `False`):
                Whether to force remove the background even if the image has an alpha channel.
        Returns:
            `List[PIL.Image.Image]`: List of `PIL.Image.Image` objects representing the preprocessed image.
        """
        preprocessed_images = []
        for i in range(len(images_pil)):
            image = images_pil[i]
            width, height, size = image.width, image.height, image.size
            do_remove = True
            if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
                # explain why current do not rm bg
                print("alhpa channl not empty, skip remove background, using alpha channel as mask")
                do_remove = False
            do_remove = do_remove or force
            if do_remove:
                image = rembg.remove(image)

            # calculate the min bbox of the image
            alpha = image.split()[-1]
            bboxs = alpha.getbbox()
            x1, y1, x2, y2 = bboxs
            dy, dx = y2 - y1, x2 - x1
            s = min(height * foreground_ratio / dy, width * foreground_ratio / dx)
            Ht, Wt = int(dy * s), int(dx * s)
            
            background = PIL.Image.new("RGBA", image.size, (*background_color, 255))
            image = PIL.Image.alpha_composite(background, image)
            image = image.crop(alpha.getbbox())
            alpha = alpha.crop(alpha.getbbox())

            # Calculate the new size after rescaling
            new_size = tuple(int(dim * foreground_ratio) for dim in size)
            # Resize the image while maintaining the aspect ratio
            resized_image = image.resize((Wt, Ht))
            resized_alpha = alpha.resize((Wt, Ht))
            # Create a new image with the original size and white background
            padded_image = PIL.Image.new("RGB", size, tuple(background_color))
            padded_alpha = PIL.Image.new("L", size, (0))
            paste_position = ((width - resized_image.width) // 2, (height - resized_image.height) // 2)
            padded_image.paste(resized_image, paste_position)
            padded_alpha.paste(resized_alpha, paste_position)

            # expand image to 1:1
            width, height = padded_image.size
            if width == height:
                padded_image.putalpha(padded_alpha)
                preprocessed_images.append(padded_image)
                continue
            new_size = (max(width, height), max(width, height))
            new_image = PIL.Image.new("RGB", new_size, tuple(background_color))
            new_alpha = PIL.Image.new("L", new_size, (0))
            paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
            new_image.paste(padded_image, paste_position)
            new_alpha.paste(padded_alpha, paste_position)
            new_image.putalpha(new_alpha)
            preprocessed_images.append(new_image)

        return preprocessed_images

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image, str],
        pose_pkg: Optional[dict] = None,
        return_dict: bool = True,
        seed: Optional[int] = None,
        force_remove_background: bool = False,
        background_color: List[int] = [255, 255, 255],
        foreground_ratio: float = 0.825,
        extract_mesh: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch. The image will be encoded to its CLIP/DINO-v2 embedding 
                which the DiT will be conditioned on. 
            num_inference_steps (`int`, *optional*, defaults to 20):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 10.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter as defined in [DDIM](https://arxiv.org/abs/2010.02502). `eta` is a parameter that
                controls the amount of noise added to the latent space. It is only used with the DDIM scheduler and
                will be ignored for other schedulers. `eta` should be between [0, 1].
            num_meshes_per_prompt (`int`, *optional*, defaults to 1):
                The number of meshes to generate per prompt.
            output_type (`str`, *optional*, defaults to `"trimesh"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image`, `latents` or `np.array of v and f`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            seed (`int`, *optional*, defaults to `None`):
                Seed for the random number generator. Setting a seed will ensure reproducibility.
            force_remove_background (`bool`, *optional*, defaults to `False`):
                Whether to force remove the background even if the image has an alpha channel.
            foreground_ratio (`float`, *optional*, defaults to 1.0):
                The ratio of the foreground in the image. The foreground is the part of the image that is not the
                background. The foreground is resized to the size of the background image while maintaining the aspect
                ratio. The background is filled with black color. The foreground ratio should be between [0, 1].
            mc_depth (`int`, *optional*, defaults to 8):
                The resolution of the Marching Cubes algorithm. The resolution is the number of cubes in the x, y, and z.
                8 means 2^8 = 256 cubes in each dimension. The higher the resolution, the more detailed the mesh will be.
            only_max_component (`bool`, *optional*, defaults to `False`):
                Whether to only keep the largest connected component of the mesh. This is useful when the mesh has
                multiple components and only the largest one is needed.
        Examples:

        Returns:
            [`~MeshPipelineOutput`] or `tuple`: [`~MeshPipelineOutput`] if `return_dict` is True, otherwise a `tuple`. 
            When returning a tuple, the first element is a list with the generated meshes.
        """
        torch.manual_seed(seed)
        # 0. Check inputs. Raise error if not correct
        self.check_inputs(
            image=image,
        )

        # 1. Define call parameters
        if isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        elif isinstance(image, PIL.Image.Image) or isinstance(image, str):
            batch_size = 1

        # 2. Preprocess input image
        if isinstance(image, torch.Tensor):
            images_pil = [TF.to_pil_image(image[i]) for i in range(image.shape[0])]
        elif isinstance(image, PIL.Image.Image):
            images_pil = [image]
        elif isinstance(image, str):
            if image.startswith("http"):
                import requests
                images_pil = [PIL.Image.open(requests.get(image, stream=True).raw)]
            else:
                images_pil = [PIL.Image.open(image)]

        if pose_pkg is not None:
            images_pil = images_pil
        else:
            images_pil = self.preprocess_image(
                images_pil, 
                force=force_remove_background,
                background_color=background_color,
                foreground_ratio=foreground_ratio
                )


        # 3. Inference (sample)
        input_images = np.asarray(images_pil[0])/255.
        image_torch = torch.from_numpy(input_images).unsqueeze(0)[...,:3].permute(0,3,1,2)
        image_torch = F.interpolate(image_torch, size=(512, 512), mode='bilinear', align_corners=False)
        image_torch = image_torch.float().to(self.system.device)
        if pose_pkg is not None:
            pose_pkg = torch.load(pose_pkg)
            input_c2w = pose_pkg['c2w']
            input_fxfycxcys = pose_pkg['fxfycxcys']
            gen_c2w = self.system.cam_template['gen_c2w']
            gen_fxfycxcys = self.system.cam_template['gen_fxfycxcys']
        else:
            #### seeing input image as the first image
            # breakpoint()
            input_c2w = self.system.cam_template['gen_c2w'][:,0]
            input_fxfycxcys = self.system.cam_template['gen_fxfycxcys'][:,0]
            gen_c2w = self.system.cam_template['gen_c2w'][:,1:4]
            gen_fxfycxcys = self.system.cam_template['gen_fxfycxcys'][:,1:4]
        # 4. Sample noise
        # breakpoint()
        sample_noise = torch.randn(1, gen_c2w.shape[1], 3, image_torch.shape[-2], image_torch.shape[-1], device=self.system.device)  # 在 timestep 为 T 的时候采的 noise
        rgbs_input = torch.cat((image_torch.unsqueeze(1),sample_noise),dim=1)
        c2ws_input = torch.cat((input_c2w.unsqueeze(1),gen_c2w),dim=1).to(self.system.device)
        fxfycxcys_input = torch.cat((input_fxfycxcys.unsqueeze(1),gen_fxfycxcys),dim=1).to(self.system.device)
        ray_o, ray_d = TransformInput(rgbs_input, c2ws_input,fxfycxcys_input)
        input_batch = edict(
            image = rgbs_input[:,:1].to(self.system.dtype),
            c2w = c2ws_input.to(self.system.dtype),
            fxfycxcy = fxfycxcys_input.to(self.system.dtype),
            ray_o=ray_o.to(self.system.dtype),
            ray_d=ray_d.to(self.system.dtype),
            )
        input_batch["image_noisy"] = sample_noise.to(self.system.dtype)
        with torch.autocast(device_type="cuda", dtype=self.system.dtype): 
            for out in self.system.diffusion_inference.p_sample_loop_progressive(
                self.system.shape_model,
                sample_noise.shape,
                input_batch,
                clip_denoised=False,
                progress=True,
                device=self.system.device,
            ):
                final_out = out
        gaussians = final_out['denoiser_output_dict']['pred_gaussians'][0]
        pred_images = final_out['denoiser_output_dict']['render_images'][0]
        # 4. Post-processing
        gaussians.apply_all_filters(
            opacity_thres=0.02,
            crop_bbx=[-0.91, 0.91, -0.91, 0.91, -0.91, 0.91],
            cam_origins=None,
            nearfar_percent=(0.0001, 1.0),
            )
        if not return_dict:
            return gaussians
        if extract_mesh:
            mesh = gaussians.extract_mesh()
            return GSPipelineOutput(gaussians=gaussians, render_images=pred_images, mesh=mesh)
        else:
            return GSPipelineOutput(gaussians=gaussians, render_images=pred_images)

    # ------------------------------------------------------------------ #
    # Helpers for dual-path (interleaved) denoising
    # ------------------------------------------------------------------ #

    def _prepare_batch(self, image_pil, shared_noise,
                       foreground_ratio=0.825,
                       force_remove_background=False,
                       background_color=None):
        """Set up one inference input_batch from a PIL image and a pre-generated noise tensor."""
        if background_color is None:
            background_color = [255, 255, 255]
        images_pil = self.preprocess_image(
            [image_pil],
            force=force_remove_background,
            background_color=background_color,
            foreground_ratio=foreground_ratio,
        )
        input_images = np.asarray(images_pil[0]) / 255.
        image_torch = torch.from_numpy(input_images).unsqueeze(0)[..., :3].permute(0, 3, 1, 2)
        image_torch = F.interpolate(image_torch, size=(512, 512), mode='bilinear', align_corners=False)
        image_torch = image_torch.float().to(self.system.device)

        input_c2w = self.system.cam_template['gen_c2w'][:, 0]
        input_fxfycxcys = self.system.cam_template['gen_fxfycxcys'][:, 0]
        gen_c2w = self.system.cam_template['gen_c2w'][:, 1:4]
        gen_fxfycxcys = self.system.cam_template['gen_fxfycxcys'][:, 1:4]

        rgbs_input = torch.cat((image_torch.unsqueeze(1), shared_noise), dim=1)
        c2ws_input = torch.cat((input_c2w.unsqueeze(1), gen_c2w), dim=1).to(self.system.device)
        fxfycxcys_input = torch.cat((input_fxfycxcys.unsqueeze(1), gen_fxfycxcys), dim=1).to(self.system.device)
        ray_o, ray_d = TransformInput(rgbs_input, c2ws_input, fxfycxcys_input)

        batch = edict(
            image=rgbs_input[:, :1].to(self.system.dtype),
            c2w=c2ws_input.to(self.system.dtype),
            fxfycxcy=fxfycxcys_input.to(self.system.dtype),
            ray_o=ray_o.to(self.system.dtype),
            ray_d=ray_d.to(self.system.dtype),
        )
        batch["image_noisy"] = shared_noise.to(self.system.dtype)
        return batch

    @torch.no_grad()
    def dual_call(self, geo_image, color_image,
                  seed=42, foreground_ratio=0.825,
                  force_remove_background=False, background_color=None):
        """Interleaved dual denoising.

        At every diffusion step the color path's noisy novel views are replaced
        with the geo path's, so the color model is always denoising geometry
        that matches geo while being conditioned on the color image.

        Returns a merged GaussianModel: geo geometry + color features.
        """
        from tqdm.auto import tqdm

        if background_color is None:
            background_color = [255, 255, 255]

        self.system = self.system.to(self.device)
        torch.manual_seed(seed)

        gen_c2w = self.system.cam_template['gen_c2w'][:, 1:4]
        # Generate shared initial noise so both paths start from the same geometry seed
        H = W = 512
        shared_noise = torch.randn(
            1, gen_c2w.shape[1], 3, H, W, device=self.system.device,
        )

        prep = dict(foreground_ratio=foreground_ratio,
                    force_remove_background=force_remove_background,
                    background_color=background_color)
        geo_batch   = self._prepare_batch(geo_image,   shared_noise.clone(), **prep)
        color_batch = self._prepare_batch(color_image, shared_noise.clone(), **prep)

        diffusion = self.system.diffusion_inference
        model     = self.system.shape_model
        indices   = list(range(diffusion.num_timesteps))[::-1]

        # Novel-view camera params for re-rendering merged Gaussians each step
        # cam_template is loaded from disk → CPU; rasterizer needs GPU tensors
        dev = self.system.device
        gen_c2w_rv      = self.system.cam_template['gen_c2w'][0, 1:4].float().to(dev)       # [3, 4, 4]
        gen_fxfycxcy_rv = self.system.cam_template['gen_fxfycxcys'][0, 1:4].float().to(dev) # [3, 4]
        n_gen_views = gen_c2w_rv.shape[0]
        H = W = shared_noise.shape[-1]

        geo_out = color_out = None
        with torch.autocast(device_type="cuda", dtype=self.system.dtype):
            for i in tqdm(indices, desc="Dual DGS"):
                t = torch.tensor([i], device=self.system.device)

                geo_out   = diffusion.p_sample(model, geo_batch,   t, clip_denoised=False)
                color_out = diffusion.p_sample(model, color_batch, t, clip_denoised=False)

                geo_batch   = geo_out["input_batch"]
                color_batch = color_out["input_batch"]

                if i > 0:
                    # ---- Splatter merge + re-render --------------------------------
                    # Both paths produce the same Gaussian count (architecture-fixed,
                    # no filter yet) → direct index swap, no NN needed.
                    geo_gs_s   = geo_out["denoiser_output_dict"]["pred_gaussians"][0]
                    color_gs_s = color_out["denoiser_output_dict"]["pred_gaussians"][0]

                    # Build merged features tensor [N, K+1, 3] from color path.
                    # Must go through set_data so _features_dc/_features_rest
                    # are guaranteed contiguous (rasterizer hard-requires this).
                    c_dc   = color_gs_s._features_dc.float()    # [N, 1, 3]
                    c_rest = color_gs_s._features_rest
                    if c_rest is not None:
                        color_feats = torch.cat([c_dc, c_rest.float()], dim=1)
                    else:
                        color_feats = c_dc                        # [N, 1, 3]

                    # Save geo's originals, temporarily install merged data
                    orig = (geo_gs_s._xyz, geo_gs_s._features_dc, geo_gs_s._features_rest,
                            geo_gs_s._scaling, geo_gs_s._rotation, geo_gs_s._opacity)
                    geo_gs_s.set_data(
                        geo_gs_s._xyz.float(),
                        color_feats,
                        geo_gs_s._scaling.float(),
                        geo_gs_s._rotation.float(),
                        geo_gs_s._opacity.float(),
                    )

                    # Render outside autocast (rasterizer is float32-only)
                    renders = []
                    with torch.autocast(device_type="cuda", enabled=False):
                        for j in range(n_gen_views):
                            r = render_opencv_cam(
                                geo_gs_s, H, W,
                                gen_c2w_rv[j], gen_fxfycxcy_rv[j],
                            )["render"].clamp(0, 1)  # [3, H, W], float32
                            renders.append(r)

                    # Restore geo's original attributes
                    (geo_gs_s._xyz, geo_gs_s._features_dc, geo_gs_s._features_rest,
                     geo_gs_s._scaling, geo_gs_s._rotation, geo_gs_s._opacity) = orig

                    # [1, V, 3, H, W] — feed as the color path's next noisy state
                    merged_views = torch.stack(renders, dim=0).unsqueeze(0)  # float32
                    t_prev = torch.tensor([i - 1], device=self.system.device)
                    color_batch["image_noisy"] = diffusion.q_sample(
                        merged_views, t_prev
                    ).to(self.system.dtype)

        color_gs = color_out["denoiser_output_dict"]["pred_gaussians"][0]

        color_gs.apply_all_filters(
            opacity_thres=0.02,
            crop_bbx=[-0.91, 0.91, -0.91, 0.91, -0.91, 0.91],
            cam_origins=None,
            nearfar_percent=(0.0001, 1.0),
        )
        return color_gs
