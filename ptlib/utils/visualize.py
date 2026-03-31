"""Visualize tools"""

import io
from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import PIL


# Color palette (12 colors)
COLORS = [
    "#0eeee7",  # cyan
    "#fefe08",  # yellow
    "#ff6b6b",  # pink-red
    "#ffa600",  # orange
    "#ffca3a",  # warm yellow
    "#8ac926",  # lime green
    "#1982c4",  # blue
    "#6a4c93",  # purple
    "#e36df5",  # lilac
    "#00ffa8",  # mint
    "#ffd1dc",  # pastel pink
    "#3cf0ff",  # bright turquoise
    "#ff4d00",  # vivid orange-red
]


def _create_axis_config(show_elements: bool = False) -> dict:
    return dict(
        backgroundcolor="black",
        showgrid=show_elements,
        showline=show_elements,
        showticklabels=show_elements,
        zeroline=show_elements,
    )


def _create_scatter3d(
    points: np.ndarray,
    color: str,
    size: float = 0.7,
    name: Optional[str] = None,
    showlegend: bool = False,
    colors: Optional[np.ndarray] = None,
) -> go.Scatter3d:
    """
    Create a 3D scatter plot.

    Args:
        points: Array of shape (N, 3) with (x, y, z) coordinates
        color: Default color for all points (used if colors is None)
        size: Marker size
        name: Name for the trace
        showlegend: Whether to show in legend
        colors: Optional array of shape (N, 3) with RGB values in range [0, 1] or (N,) with scalar values
    """
    marker_config = {"size": size}

    if colors is not None:
        # If colors is provided, use it for per-point coloring
        if colors.ndim == 2 and colors.shape[1] == 3:
            # RGB colors: convert to plotly format
            marker_config["color"] = [
                f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
                for r, g, b in colors
            ]
        else:
            # Scalar colors or other format
            marker_config["color"] = colors
            marker_config["colorscale"] = "Viridis"
    else:
        # Use default color for all points
        marker_config["color"] = color

    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=marker_config,
        name=name,
        showlegend=showlegend,
    )


def _get_camera_config(view: str) -> dict:
    camera_configs = dict(
        side=dict(
            eye=dict(x=1, y=1, z=1),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1),
        ),
        top=dict(
            eye=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1),
        ),
    )
    return camera_configs.get(view, camera_configs["top"])


def _compute_bounding_box(
    point_clouds: List[np.ndarray],
) -> Tuple[float, float, float, float]:
    all_pts = np.vstack(point_clouds)
    mins = np.nanmin(all_pts, axis=0)
    maxs = np.nanmax(all_pts, axis=0)

    # Cube center - middle of bbox
    cx, cy, cz = (mins + maxs) / 2.0

    # Half of cube edge = half of maximum bbox size
    max_extent = float(np.max(maxs - mins))
    half = 0.5 * max_extent

    # Small margin to avoid clipping
    pad = 0.05 * half  # 5%
    half = half + pad

    return cx, cy, cz, half


def _create_visibility_buttons(
    pcd_names: List[str],
    data_length: int,
    cx: float,
    cy: float,
    cz: float,
    half: float,
) -> List[dict]:
    # Fixed scene range for all buttons
    scene_range = {
        "scene.xaxis.range": [cx - half, cx + half],
        "scene.yaxis.range": [cy - half, cy + half],
        "scene.zaxis.range": [cz - half, cz + half],
    }

    buttons = [
        dict(
            label="Show all",
            method="update",
            args=[{"visible": [True] * data_length}, scene_range],
        )
    ]

    for idx, name in enumerate(pcd_names):
        # Create visibility array: False for all point clouds except current one, True for center
        visibility = [False] * data_length
        visibility[idx] = True  # Show current point cloud
        visibility[-1] = True  # Always show center
        buttons.append(
            dict(
                label=name, method="update", args=[{"visible": visibility}, scene_range]
            )
        )

    return buttons


def get_figure(
    data: List[go.Scatter3d],
    camera: dict,
    height: int = 800,
    width: int = 1200,
) -> go.Figure:
    fig = go.Figure(
        data=data,
        layout=dict(
            scene=dict(
                aspectmode="cube",
                bgcolor="black",
            ),
            height=height,
            width=width,
            scene_camera=camera,
            paper_bgcolor="black",
            plot_bgcolor="black",
            margin=dict(l=0, r=0, t=0, b=0),
        ),
    )
    return fig


def points_visualize(
    pcd_list: List[dict],
    center: Optional[Tuple[float, float, float]] = None,
    frame_name: Optional[str] = None,
    return_as_image: bool = False,
    path_to_html: Optional[str] = None,
    view="top",
    height=800,
    width=1200,
    show_fig=True,
) -> None:
    """
    Visualizes multiple point clouds in 3D space.

    Args:
        pcd_list (List[dict]): List of point cloud dictionaries, each containing:
                               - 'pcd': np.ndarray of shape (N, 3) with (x, y, z) coordinates
                               - 'name': str with the name of the point cloud
                               - 'colors': (optional) np.ndarray of shape (N, 3) with RGB values in [0, 1]
                                          or (N,) with scalar values for per-point coloring
        center (Tuple[float, float, float], optional): The (x, y, z) coordinates of the center.
                                                      If None, the mean of the first pcd is used.
        frame_name (str, optional): Name of the file to save the visualization as an image.
                                    If None, the visualization is displayed interactively.
        return_as_image (bool): If True, returns the figure as a numpy array image.
        path_to_html (str, optional): Path to save the figure as an HTML file.
        view (str): Camera view type ('top' or 'side').
        height (int): Figure height in pixels.
        width (int): Figure width in pixels.
        show_fig (bool): If True, displays the figure.

    Returns:
        Optional[np.ndarray]: If return_as_image is True, returns the figure as an image array.

    Example usage:
        pcd_list = [
            dict(pcd=origin_merged_points, name="merged_points"),
            dict(pcd=new_merged_points, name="merged_points from panorama",
                 colors=np.random.rand(len(new_merged_points), 3)),
        ]

        points_visualize(
            pcd_list=pcd_list,
            center=(0, 0, 0),
            view="top",
        )
    """

    point_clouds = [item["pcd"] for item in pcd_list]
    pcd_names = [item["name"] for item in pcd_list]
    pcd_colors = [item.get("colors", None) for item in pcd_list]

    if center is None:
        center = np.mean(point_clouds[0], axis=0)
    else:
        center = np.array(center)

    camera = _get_camera_config(view)

    data = []
    for idx, (pcd, name, colors) in enumerate(zip(point_clouds, pcd_names, pcd_colors)):
        color = COLORS[idx % len(COLORS)]
        data.append(
            _create_scatter3d(
                pcd, color, size=0.7, name=name, showlegend=True, colors=colors
            )
        )

    data.append(
        _create_scatter3d(
            np.array([center]), "Yellow", size=5, name="center", showlegend=True
        )
    )

    fig = get_figure(data=data, camera=camera, height=height, width=width)

    # Compute bounding box for all points
    cx, cy, cz, half = _compute_bounding_box(point_clouds)

    axis_config = _create_axis_config(show_elements=False)
    fig.update_layout(
        scene=dict(
            xaxis=dict(**axis_config, range=[cx - half, cx + half]),
            yaxis=dict(**axis_config, range=[cy - half, cy + half]),
            zaxis=dict(**axis_config, range=[cz - half, cz + half]),
        ),
    )

    if len(pcd_list) > 1:
        buttons = _create_visibility_buttons(pcd_names, len(data), cx, cy, cz, half)
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.5,
                    y=1.05,
                    xanchor="center",
                    yanchor="bottom",
                    buttons=buttons,
                )
            ]
        )

    if path_to_html is not None:
        fig.write_html(path_to_html)

    if return_as_image:
        img_bytes = fig.to_image(format="png", scale=2)
        img = PIL.Image.open(io.BytesIO(img_bytes))
        return np.array(img)
    elif frame_name is not None:
        fig.write_image(frame_name, scale=2)

    if show_fig:
        fig.show()
