from pathlib import Path
from typing import Callable, Literal, Tuple

import numpy as np
import viser
from nerfview import RenderTabState, Viewer


class GsplatRenderTabState(RenderTabState):
    # non-controlable parameters
    total_gs_count: int = 0
    rendered_gs_count: int = 0
    date: str = ""

    # controlable parameters
    time: float = 0
    hour: float = 12
    weather: float = 1

    backgrounds: Tuple[float, float, float] = (255, 255, 255)
    render_mode: Literal["full", "albedo", "shading"] = "full"

    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"


class GsplatViewer(Viewer):
    """
    Viewer for gsplat.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mode: Literal["rendering", "training"] = "rendering",
    ):
        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("Time Splatting Viewer")
        # server.gui.configure_theme(control_width="large")
        server.gui.configure_theme(
            control_layout="collapsible",
            dark_mode=True,
            brand_color=(255, 211, 105),
        )

    def _init_rendering_tab(self):
        self.render_tab_state = GsplatRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Time Splatting")

        self._camera_folder = self.server.gui.add_folder("Camera")

    def _populate_rendering_tab(self):
        server = self.server
        with self._rendering_folder:
            date = server.gui.add_text(
                "Date",
                initial_value=self.render_tab_state.date,
                disabled=True,
                hint="Date rendered",
            )
            total_gs_count_number = server.gui.add_number(
                "Total",
                initial_value=self.render_tab_state.total_gs_count,
                disabled=True,
                hint="Total number of splats in the scene.",
            )
            rendered_gs_count_number = server.gui.add_number(
                "Rendered",
                initial_value=self.render_tab_state.rendered_gs_count,
                disabled=True,
                hint="Number of splats rendered.",
            )

            time = server.gui.add_slider(
                "Normalized Date",
                initial_value=self.render_tab_state.time,
                min=0.0,
                max=1.0,
                step=0.01,
            )

            @time.on_update
            def _(_) -> None:
                self.render_tab_state.time = time.value
                self.rerender(_)

            hour = server.gui.add_slider(
                "Hour of Day",
                initial_value=self.render_tab_state.hour,
                min=5.0,
                max=22.0,
                step=0.1,
            )

            @hour.on_update
            def _(_) -> None:
                self.render_tab_state.hour = hour.value
                self.rerender(_)

            weather = server.gui.add_slider(
                "Weather",
                initial_value=1,
                min=0.0,
                max=1.0,
                step=0.01,
            )

            @weather.on_update
            def _(_) -> None:
                self.render_tab_state.weather = weather.value
                self.rerender(_)

            render_mode_dropdown = server.gui.add_dropdown(
                "Intrinsic Images",
                ("full", "albedo", "shading"),
                initial_value=self.render_tab_state.render_mode,
                hint="Render mode to use.",
            )

            @render_mode_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.render_mode = render_mode_dropdown.value
                self.rerender(_)

            backgrounds_slider = server.gui.add_rgb(
                "Background",
                initial_value=self.render_tab_state.backgrounds,
                hint="Background color for rendering.",
            )

            @backgrounds_slider.on_update
            def _(_) -> None:
                self.render_tab_state.backgrounds = backgrounds_slider.value
                self.rerender(_)

            rasterize_mode_dropdown = server.gui.add_dropdown(
                "Anti-Aliasing",
                ("classic", "antialiased"),
                initial_value=self.render_tab_state.rasterize_mode,
                hint="Whether to use classic or antialiased rasterization.",
            )

            @rasterize_mode_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.rasterize_mode = rasterize_mode_dropdown.value
                self.rerender(_)

        with self._camera_folder:
            fov_degrees_slider = server.gui.add_slider(
                "FOV",
                initial_value=120.0,
                min=0.1,
                max=175.0,
                step=0.01,
                hint="Field-of-view for rendering, which can also be overridden on a per-keyframe basis.",
            )

            @fov_degrees_slider.on_update
            def _(_) -> None:
                fov_radians = fov_degrees_slider.value / 180.0 * np.pi
                for client in server.get_clients().values():
                    client.camera.fov = fov_radians

            viewer_res_slider = self.server.gui.add_slider(
                "Viewer Res",
                min=64,
                max=2048,
                step=1,
                initial_value=2048,
                hint="Maximum resolution of the viewer rendered image.",
            )

            @viewer_res_slider.on_update
            def _(_) -> None:
                self.render_tab_state.viewer_res = int(viewer_res_slider.value)
                self.rerender(_)

        self._rendering_tab_handles.update(
            {
                "total_gs_count_number": total_gs_count_number,
                "rendered_gs_count_number": rendered_gs_count_number,
                "date": date,
                "time": time,
                "hour": hour,
                "weather": weather,
                "backgrounds_slider": backgrounds_slider,
                "render_mode_dropdown": render_mode_dropdown,
                "rasterize_mode_dropdown": rasterize_mode_dropdown,
                "fov_degrees_slider": fov_degrees_slider,
                "viewer_res_slider": viewer_res_slider,
            }
        )
        # super()._populate_rendering_tab()

    def _after_render(self):
        # Update the GUI elements with current values
        self._rendering_tab_handles[
            "total_gs_count_number"
        ].value = self.render_tab_state.total_gs_count
        self._rendering_tab_handles[
            "rendered_gs_count_number"
        ].value = self.render_tab_state.rendered_gs_count

        self._rendering_tab_handles["date"].value = self.render_tab_state.date
