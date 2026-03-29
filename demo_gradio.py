import gradio as gr
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import imageio

from megaflow.model import MegaFlow
from megaflow.utils.basic import gridcloud2d
from megaflow.utils.visualizer import Visualizer
from megaflow.utils.flow_viz import flow_to_image

# 1. Global Setup and Model Loading
device = "cuda" if torch.cuda.is_available() else "cpu"


class ModelManager:
    """Manages lazy loading and switching between track and flow models."""

    def __init__(self, device):
        self.device = device
        self.model_track = None
        self.model_flow = None

    def unload_track_model(self):
        """Unload the tracking model to free memory."""
        if self.model_track is not None:
            self.model_track = None
            torch.cuda.empty_cache()

    def unload_flow_model(self):
        """Unload the flow model to free memory."""
        if self.model_flow is not None:
            self.model_flow = None
            torch.cuda.empty_cache()

    def get_track_model(self):
        """Load and return the tracking model if not already loaded."""
        if self.model_track is None:
            print("Loading tracking model...")
            self.model_track = MegaFlow.from_pretrained(
                "megaflow-track", device=self.device
            )
            self.model_track.eval()
        return self.model_track

    def get_flow_model(self):
        """Load and return the flow model if not already loaded."""
        if self.model_flow is None:
            print("Loading flow model...")
            self.model_flow = MegaFlow.from_pretrained(
                "megaflow-flow", device=self.device
            )
            self.model_flow.eval()
        return self.model_flow


def get_model_manager(device):
    """Create and return a new ModelManager instance."""
    return ModelManager(device)


# 2. Shared Utilities
def calculate_dynamic_size(
    orig_h, orig_w, target_fix_width, patch_size=14, mode="track"
):
    if mode == "track":
        new_w = target_fix_width
        new_h = round(orig_h * (new_w / orig_w) / patch_size) * patch_size
    else:
        if orig_w >= orig_h:
            new_w = target_fix_width
            new_h = round(orig_h * (new_w / orig_w) / patch_size) * patch_size
        else:
            new_h = target_fix_width
            new_w = round(orig_w * (new_h / orig_h) / patch_size) * patch_size
    return int(new_h), int(new_w)


def get_video_frames(input_path, fix_width, mode="track"):
    cap = cv2.VideoCapture(input_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 24.0

    frames, orig_shape = [], None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if orig_shape is None:
            orig_shape = frame.shape[:2]

        new_h, new_w = calculate_dynamic_size(
            orig_shape[0], orig_shape[1], fix_width, mode=mode
        )
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    cap.release()
    return frames, orig_shape, native_fps


# 3. Tracking Wrapper
@torch.inference_mode()
def run_tracking(video_in, grid_size, iters, restore_size, model_manager):
    if not video_in:
        raise gr.Error("Please upload or select a video first.")

    # Unload flow model to free memory before loading track model
    model_manager.unload_flow_model()

    frames_np, native_size, fps = get_video_frames(
        video_in, fix_width=518, mode="track"
    )
    if len(frames_np) < 2:
        raise gr.Error("Video requires at least 2 frames.")

    input_image = [torch.from_numpy(f).permute(2, 0, 1).float() for f in frames_np]
    frames = torch.stack(input_image, dim=0)[None].to(device)
    B, T, _, H, W = frames.shape

    grid_xy = gridcloud2d(1, H, W, norm=False, device=device).float()
    grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)

    # Get track model (loads if not already loaded)
    model_track = model_manager.get_track_model()

    compute_dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    with torch.autocast(
        device_type=device, dtype=compute_dtype, enabled=(device == "cuda")
    ):
        results = model_track.forward_track(frames, num_reg_refine=iters)
        flows_e = results["flow_final"]
        traj_maps = flows_e.to(device) + grid_xy

    traj_sub = traj_maps[..., ::grid_size, ::grid_size]
    pred_tracks = traj_sub.flatten(3).permute(0, 1, 3, 2)

    if restore_size and native_size is not None:
        orig_H, orig_W = native_size
        pred_tracks[..., 0] *= orig_W / W
        pred_tracks[..., 1] *= orig_H / H
        frames = F.interpolate(
            frames[0], size=(orig_H, orig_W), mode="bilinear", align_corners=True
        ).unsqueeze(0)

    output_dir = "gradio_output_track"
    os.makedirs(output_dir, exist_ok=True)
    filename = "track_result"

    vis = Visualizer(
        save_dir=output_dir, pad_value=0, linewidth=1, tracks_leave_trace=0, fps=fps
    )
    vis.visualize(frames, pred_tracks, filename=filename, opacity=0.5)

    return os.path.join(output_dir, f"{filename}.mp4")


# 4. Flow Wrapper
@torch.inference_mode()
def run_flow(video_in, window_size, iters, restore_size, model_manager):
    if not video_in:
        raise gr.Error("Please upload or select a video first.")

    # Unload track model to free memory before loading flow model
    model_manager.unload_track_model()

    frames_np, native_size, fps = get_video_frames(video_in, fix_width=952, mode="flow")
    if len(frames_np) < 2:
        raise gr.Error("Video requires at least 2 frames.")

    input_image = [torch.from_numpy(f).permute(2, 0, 1).float() for f in frames_np]
    input_scene = torch.stack(input_image, dim=0)[None]
    B, T, C, H, W = input_scene.shape

    output_path = "gradio_flow_result.mp4"
    video_writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264", macro_block_size=None
    )

    # Get flow model (loads if not already loaded)
    model_flow = model_manager.get_flow_model()

    infer_window = window_size
    for start in range(0, T - 1, infer_window - 1):
        end = min(start + infer_window, T)
        chunk = input_scene[:, start:end].to(device)

        compute_dtype = (
            torch.bfloat16
            if device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        with torch.autocast(
            device_type=device, dtype=compute_dtype, enabled=(device == "cuda")
        ):
            results_dict = model_flow(chunk, num_reg_refine=iters)

        flow_pr = results_dict["flow_preds"][-1]

        if native_size is not None and restore_size:
            scaled_flow = F.interpolate(
                flow_pr.view(-1, 2, H, W),
                size=native_size,
                mode="bilinear",
                align_corners=True,
            )
            scaled_flow[:, 0, :, :] *= native_size[1] / W
            scaled_flow[:, 1, :, :] *= native_size[0] / H
            flow_pr = scaled_flow.view(*flow_pr.shape[:2], 2, *native_size)

        flow_all = flow_pr[0].permute(0, 2, 3, 1).cpu().numpy()

        for t, flow in enumerate(flow_all):
            flow_vis_rgb = flow_to_image(flow, convert_to_bgr=False)
            video_writer.append_data(flow_vis_rgb)

    video_writer.close()
    return output_path


# 5. UI Design
with gr.Blocks(theme=gr.themes.Soft(), title="MegaFlow Demo") as demo:
    gr.HTML(
        """
        <div align="center">
          <h1 style="font-weight: 800; margin-bottom: 15px;">MegaFlow: Zero-Shot Large Displacement Optical Flow</h1>

          <div style="font-size: 1.2em; margin-bottom: 10px;">
              <strong><a href="https://kristen-z.github.io/" style="text-decoration: none;">Dingxi Zhang</a></strong><sup>1</sup>&nbsp;&nbsp;&nbsp;&nbsp;
              <strong><a href="https://fangjinhuawang.github.io/" style="text-decoration: none;">Fangjinhua Wang</a></strong><sup>1</sup>&nbsp;&nbsp;&nbsp;&nbsp;
              <strong><a href="https://people.inf.ethz.ch/marc.pollefeys/" style="text-decoration: none;">Marc Pollefeys</a></strong><sup>1,2</sup>&nbsp;&nbsp;&nbsp;&nbsp;
              <strong><a href="https://haofeixu.github.io/" style="text-decoration: none;">Haofei Xu</a></strong><sup>1,3</sup>
          </div>

          <div style="font-size: 1.1em; color: #666; font-weight: 400; margin-bottom: 20px;">
              <sup>1</sup> ETH Zurich&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              <sup>2</sup> Microsoft&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              <sup>3</sup> University of Tübingen, Tübingen AI Center
          </div>

          <div style="display: flex; justify-content: center; gap: 15px; margin-bottom: 20px;">
              <a href="https://kristen-z.github.io/projects/megaflow/"><img src="https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=white" alt="Project Page"></a>
              <a href="https://arxiv.org/abs/2603.25739"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?style=flat&logo=arxiv&logoColor=white" alt="arXiv"></a>
              <a href="https://github.com/cvg/megaflow"><img src="https://img.shields.io/badge/GitHub-Repository-black?style=flat&logo=github&logoColor=white" alt="GitHub"></a>
          </div>
        </div>
        """
    )

    with gr.Row():
        vid_input = gr.Video(label="Input Video", height=400)
        vid_output = gr.Video(label="Output Result", height=400, interactive=False)

    with gr.Tabs():
        # Tab 1: Optical Flow (Moved to the front)
        with gr.TabItem("🌈 1. Optical Flow"):
            gr.Markdown(
                "### Compute dense optical flow between consecutive frames\nConfigure your parameters below, then click **Run Optical Flow** to generate the sequence."
            )
            with gr.Row():
                flow_window_size = gr.Slider(
                    minimum=2, maximum=16, step=1, value=4, label="Sliding Window Size"
                )
                flow_iters = gr.Slider(
                    minimum=1,
                    maximum=16,
                    step=1,
                    value=8,
                    label="Refinement Iterations",
                )
            flow_restore = gr.Checkbox(value=True, label="Restore Original Resolution")
            flow_btn = gr.Button("🚀 Run Optical Flow", variant="primary", size="lg")

        # Tab 2: Point Tracking
        with gr.TabItem("🎯 2. Point Tracking"):
            gr.Markdown(
                "### Track specific points across the video sequence\nConfigure your parameters below, then click **Run Point Tracking** to track the grid."
            )
            gr.Markdown(
                "*Note: As a zero-shot tracking application of our flow model, point visibility is not explicitly predicted, resulting in tracking through occlusions.*"
            )
            with gr.Row():
                track_grid_size = gr.Slider(
                    minimum=2,
                    maximum=16,
                    step=2,
                    value=8,
                    label="Grid Size (Subsampling)",
                )
                track_iters = gr.Slider(
                    minimum=1,
                    maximum=16,
                    step=1,
                    value=8,
                    label="Refinement Iterations",
                )
            track_restore = gr.Checkbox(value=True, label="Restore Original Resolution")
            track_btn = gr.Button("🚀 Run Point Tracking", variant="primary", size="lg")

    gr.Markdown("### Try an Example")

    example_videos = [
        ["assets/apple.mp4"],
        ["assets/longboard.mp4"],
        ["assets/chamaleon.mp4"],
    ]

    gr.Examples(
        examples=example_videos,
        inputs=vid_input,
        label="Click any video to load it instantly into the Input Video player",
        examples_per_page=5,
    )

    # Create model manager for use in callbacks
    model_manager = get_model_manager(device)

    # Store model_manager as a state variable to pass to callbacks
    model_state = gr.State(model_manager)

    # Wire up the logic
    track_btn.click(
        fn=run_tracking,
        inputs=[vid_input, track_grid_size, track_iters, track_restore, model_state],
        outputs=vid_output,
    )

    flow_btn.click(
        fn=run_flow,
        inputs=[vid_input, flow_window_size, flow_iters, flow_restore, model_state],
        outputs=vid_output,
    )

if __name__ == "__main__":
    demo.launch()
