import os
import gradio as gr

from video_processor import video_info, make_summary_video
from scene_detection import detect_scenes
from motion_analysis import analyze_motion
from object_detection import detect_objects
from video_summary import find_important_frames


BASE = os.path.dirname(os.path.abspath(__file__))

FRAMES = os.path.join(BASE, "results", "frames")
MOTION = os.path.join(BASE, "results", "motion")
OBJECTS = os.path.join(BASE, "results", "objects")
OUTPUT = os.path.join(BASE, "data", "output")


# Create folders
for folder in [FRAMES, MOTION, OBJECTS, OUTPUT]:
    os.makedirs(folder, exist_ok=True)


def analyze_video(video_path):

    if not video_path:
        return "Please upload a video.", None, None, None, None

    # Handle newer Gradio video input format
    if isinstance(video_path, dict):
        video_path = (
            video_path.get("path")
            or video_path.get("video")
            or video_path.get("filepath")
        )

    if not video_path:
        return "Could not find the uploaded video file.", None, None, None, None

    try:

        # ==================================================
        # VIDEO INFORMATION
        # ==================================================

        info = video_info(video_path)

        # ==================================================
        # SCENE DETECTION
        # ==================================================

        scenes = detect_scenes(
            video_path,
            sample_every=10,
            threshold=0.45
        )

        # ==================================================
        # IMPORTANT FRAME EXTRACTION
        # ==================================================

        chosen, frame_paths = find_important_frames(
            video_path,
            FRAMES,
            max_frames=12,
            sample_every=10
        )

        # ==================================================
        # MOTION ANALYSIS
        # ==================================================

        motion_plot = os.path.join(
            MOTION,
            "motion_analysis.png"
        )

        motion = analyze_motion(
            video_path,
            motion_plot,
            sample_every=5
        )

        # ==================================================
        # OBJECT DETECTION
        # ==================================================

        object_plot = os.path.join(
            OBJECTS,
            "object_detection.jpg"
        )

        counts = detect_objects(
            video_path,
            object_plot
        )

        # ==================================================
        # VIDEO SUMMARY
        # ==================================================

        summary_path = os.path.join(
            OUTPUT,
            "video_summary.mp4"
        )

        selected_frames = []

        for frame_number, frame in chosen:
            selected_frames.append(frame)

        make_summary_video(
            video_path,
            summary_path,
            selected_frames
        )

        # ==================================================
        # SCENE INFORMATION
        # ==================================================

        scene_text = ""

        if scenes:

            for scene in scenes[:20]:

                scene_text += (
                    "- "
                    + str(scene["time"])
                    + " seconds "
                    + "(change score: "
                    + str(scene["score"])
                    + ")"
                    + "\n"
                )

        else:

            scene_text = "No strong scene boundaries detected."

        # ==================================================
        # OBJECT INFORMATION
        # ==================================================

        object_text = ""

        if counts:

            sorted_objects = sorted(
                counts.items(),
                key=lambda x: -x[1]
            )

            for name, count in sorted_objects:

                object_text += (
                    "- **"
                    + str(name)
                    + "**: "
                    + str(count)
                    + " detections\n"
                )

        else:

            object_text = "No objects detected."

        # ==================================================
        # REPORT
        # ==================================================

        report = (
            "# Video Analysis Complete\n\n"

            "## Video Information\n\n"

            "| Property | Value |\n"
            "|---|---:|\n"

            "| Duration | "
            + f"**{info['duration']:.2f} seconds**"
            + " |\n"

            "| Resolution | "
            + f"**{info['width']} x {info['height']}**"
            + " |\n"

            "| FPS | "
            + f"**{info['fps']:.2f}**"
            + " |\n"

            "| Total Frames | "
            + f"**{info['frames']}**"
            + " |\n\n"

            "---\n\n"

            "## Important Frames\n\n"

            "The system selected **"
            + str(len(chosen))
            + " important frames** using frame-difference analysis.\n\n"

            "---\n\n"

            "## Scene Detection\n\n"

            "Detected **"
            + str(len(scenes))
            + " potential scene changes**.\n\n"

            "### Scene Boundaries\n\n"

            + scene_text

            + "\n---\n\n"

            "## Motion Analysis\n\n"

            "Average optical-flow magnitude: **"
            + f"{motion['average_motion']:.4f}"
            + "**\n\n"

            "The motion graph is displayed below.\n\n"

            "---\n\n"

            "## Object Detection\n\n"

            + object_text

            + "\n---\n\n"

            "## Video Summary\n\n"

            "A short summary video was generated using the selected important frames.\n\n"

            "Output file:\n\n"

            "`"
            + summary_path
            + "`\n\n"

            "---\n\n"

            "## Processing Complete\n\n"

            "The system successfully performed:\n\n"

            "- Video information extraction\n"
            "- Important frame extraction\n"
            "- Scene detection\n"
            "- Motion analysis\n"
            "- Object detection\n"
            "- Video summarization\n"
        )

        return (
            report,
            frame_paths,
            motion_plot,
            object_plot,
            summary_path
        )

    except Exception as e:

        # No triple-quoted f-string here
        error_message = (
            "# Video Analysis Failed\n\n"
            "## Error Type\n\n"
            "`"
            + type(e).__name__
            + "`\n\n"
            "## Error\n\n"
            "```text\n"
            + str(e)
            + "\n```\n\n"
            "Please check the uploaded video and installed dependencies."
        )

        return (
            error_message,
            None,
            None,
            None,
            None
        )


# ==========================================================
# GRADIO INTERFACE
# ==========================================================

with gr.Blocks(
    title="Intelligent Video Processing"
) as demo:

    gr.Markdown(
        """
# Intelligent Video Processing and Analysis System

Upload a video to perform:

### Important Frame Extraction
### Scene Detection
### Motion Analysis
### Object Detection
### Video Summarization
"""
    )

    gr.Markdown("---")

    video = gr.Video(
        label="Upload Video"
    )

    button = gr.Button(
        "Analyze Video",
        variant="primary",
        size="lg"
    )

    gr.Markdown("---")

    report = gr.Markdown()

    gr.Markdown(
        "## Important Frames"
    )

    gallery = gr.Gallery(
        label="Key Frames",
        columns=4,
        height="auto"
    )

    gr.Markdown(
        "## Motion Analysis"
    )

    motion_image = gr.Image(
        label="Motion Analysis"
    )

    gr.Markdown(
        "## Object Detection"
    )

    object_image = gr.Image(
        label="Detected Objects"
    )

    gr.Markdown(
        "## Short Video Summary"
    )

    summary_video = gr.Video(
        label="Generated Summary"
    )

    button.click(
        fn=analyze_video,
        inputs=video,
        outputs=[
            report,
            gallery,
            motion_image,
            object_image,
            summary_video
        ]
    )


# ==========================================================
# START APPLICATION
# ==========================================================

if __name__ == "__main__":
    demo.launch()