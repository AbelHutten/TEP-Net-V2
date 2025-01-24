from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def confidence_to_color(confidence: float) -> tuple[int, int, int]:
    """
    Map a confidence value (0 to 1) to a color interpolating between red (low confidence) and green (high confidence).

    Returns:
        Color of the overlay.
    """  # noqa: E501
    r = int(255 * (1 - confidence))  # Red decreases as confidence increases
    g = int(255 * confidence)  # Green increases as confidence increases
    b = 0  # Blue stays constant
    return (r, g, b)


def draw_egopath(
    img: Image.Image,
    egopath: list | np.ndarray,
    opacity: float = 0.7,
    color: tuple[int, int, int] = (0, 189, 80),
    crop_coords: tuple[int, int, int, int] | None = None,
    confidences=None,
    visualize_conf=False,
) -> Image.Image:
    """Overlays the train ego-path on the input image.

    Args:
        img (PIL.Image.Image): Input image on which rails are to be visualized.
        egopath (list or numpy.ndarray): Ego-path to be visualized on the image, either
        as a list of points (classification/regression) or as a mask (segmentation).
        opacity (float, optional): Opacity level of the overlay. Defaults to 0.5.
        color (tuple, optional): Color of the overlay. Defaults to (0, 189, 80).
        crop_coords (tuple, optional): Crop coordinates used during inference. If
        provided, a red rectangle will be drawn around the cropped region.
        Defaults to None.

    Returns:
        PIL.Image.Image: Image with the ego-path overlay.o
    """
    vis = img.copy()
    show_conf = confidences is not None and visualize_conf
    if isinstance(egopath, list):  # classification/regression
        left_rail, right_rail = egopath
        if not left_rail or not right_rail:
            return vis

        mask = Image.new("RGBA", vis.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(mask)

        confidences = torch.mean(confidences, dim=0)

        for i in range(len(left_rail) - 1):  # Iterate through the segments
            # Define the polygon for the current segment
            segment_points = [
                left_rail[i],
                left_rail[i + 1],
                right_rail[i + 1],
                right_rail[i],
            ]
            # Compute the color for this confidence value
            if show_conf:
                confidence = confidences[i]
                color = confidence_to_color(confidence)
            color_with_opacity = (*color, int(255 * opacity))

            # Draw the polygon for this segment
            draw.polygon([tuple(xy) for xy in segment_points], fill=color_with_opacity)

        # Paste the colored mask onto the original image
        vis.paste(mask, (0, 0), mask)
    elif isinstance(egopath, Image.Image):  # segmentation
        mask = Image.fromarray(np.array(egopath) * opacity).convert("L")
        colored_mask = Image.new("RGBA", mask.size, (*color, 0))
        colored_mask.putalpha(mask)
        vis.paste(colored_mask, (0, 0), colored_mask)
    if crop_coords is not None:
        draw = ImageDraw.Draw(vis)
        draw.rectangle(crop_coords, outline=(255, 0, 0), width=1)
    if confidences is not None and visualize_conf is True:
        draw = ImageDraw.Draw(vis)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60
        )
        draw.text((10, 10), f"{torch.mean(confidences):.2f}", font=font)
    return vis
