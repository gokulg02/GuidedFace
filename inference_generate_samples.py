import os, random, json, shutil
import numpy as np
from PIL import Image
import torch
import einops
from tqdm import tqdm
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from annotator.util import resize_image, HWC3
from pytorch_lightning import seed_everything
from torch_fidelity import calculate_metrics

# --- Paths ---
src_image_dir = "./training/guided_face/source"
prompt_file = "./training/guided_face/prompts.jsonl"
output_dir = "./training/guided_face/generated"
target_dir = "./training/guided_face/target"
subset_target_dir = "./training/guided_face/target_subset"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(subset_target_dir, exist_ok=True)

# --- Settings ---
num_samples = 1
image_resolution = 512
ddim_steps = 50
strength = 1.0
scale = 9.0
seed = 12345
eta = 0.0
low_threshold = 100
high_threshold = 200
guess_mode = False

N = 2000  # Number of images to generate

# --- Model Loading ---
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/model-epoch=02-step=21900.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# --- Conditioning Prompts ---
a_prompt = "best quality, extremely detailed"
n_prompt = "animated, non human, lowres, bad anatomy, worst quality"

# --- Load prompt.jsonl ---
filename_to_prompt = {}
with open(prompt_file, 'r') as f:
    for line in f:
        entry = json.loads(line)
        filename = os.path.basename(entry['source'])
        filename_to_prompt[filename] = entry['prompt']

# --- Select random N images ---
all_images = list(filename_to_prompt.keys())
random.seed(42)
selected_images = random.sample(all_images, N)

# --- Inference ---
seed_everything(seed)

for fname in tqdm(selected_images, desc="Generating images"):
    prompt = filename_to_prompt.get(fname, None)
    if prompt is None:
        continue

    try:
        input_path = os.path.join(src_image_dir, fname)
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img)
    except Exception as e:
        print(f"Failed to process {fname}: {e}")
        continue

    img_resized = resize_image(HWC3(img_np), image_resolution)
    H, W, _ = img_resized.shape

    control = torch.from_numpy(img_resized.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    full_prompt = f"{prompt}, {a_prompt}"

    cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning([full_prompt] * num_samples)]
    }
    un_cond = {
        "c_concat": None if guess_mode else [control],
        "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
    }

    shape = (4, H // 8, W // 8)
    model.control_scales = [strength] * 13

    samples, _ = ddim_sampler.sample(
        ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond
    )

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()
    x_samples = x_samples.clip(0, 255).astype(np.uint8)

    out_path = os.path.join(output_dir, fname)
    Image.fromarray(x_samples[0]).save(out_path)

    # Copy corresponding real target image
    src_target_img = os.path.join(target_dir, fname)
    dst_target_img = os.path.join(subset_target_dir, fname)
    if os.path.exists(src_target_img):
        shutil.copyfile(src_target_img, dst_target_img)

print(f"\n Generated {len(os.listdir(output_dir))} images in: {output_dir}")
print(f" Prepared {len(os.listdir(subset_target_dir))} target images in: {subset_target_dir}")

