#!/usr/bin/env python3
"""
Quick video generation script
Text prompt → Image → Video
"""
import torch
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
from PIL import Image
import imageio

def generate_video_from_text(prompt: str, output_path: str = "output.mp4"):
    """Generate video from text prompt."""
    
    print("Step 1: Generating base image...")
    # Generate image
    pipe_img = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    image = pipe_img(prompt).images[0]
    image.save("base_image.png")
    print("✅ Image saved to base_image.png")
    
    print("\nStep 2: Animating image into video...")
    # Animate image
    pipe_video = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    video_frames = pipe_video(image, num_frames=25, decode_chunk_size=8).frames[0]
    
    print("\nStep 3: Saving video...")
    # Save as MP4
    try:
        imageio.mimwrite(output_path, video_frames, fps=8)
        print(f"✅ Video saved to {output_path}")
    except:
        # Fallback to GIF
        video_frames[0].save(output_path.replace(".mp4", ".gif"), 
                           save_all=True, 
                           append_images=video_frames[1:], 
                           duration=100, 
                           loop=0)
        print(f"✅ Video saved as GIF: {output_path.replace('.mp4', '.gif')}")

if __name__ == "__main__":
    import sys
    
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a cat walking on the street"
    output = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"
    
    print(f"Generating video from prompt: '{prompt}'")
    generate_video_from_text(prompt, output)


