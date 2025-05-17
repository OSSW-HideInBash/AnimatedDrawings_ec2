# animated_drawings/view/headless_view.py

from animated_drawings.view.view import View
from animated_drawings.config import ViewConfig
from animated_drawings.model.scene import Scene
from animated_drawings.model.camera import Camera
from animated_drawings.model.transform import Transform

from typing import Tuple
import logging


class HeadlessView(View):
    """ Headless View for rendering without a visible window (e.g., server or CI/CD). """

    def __init__(self, cfg: ViewConfig) -> None:
        super().__init__(cfg)
        self.camera: Camera = Camera(cfg.camera_pos, cfg.camera_fwd)
        self.scene: Scene = None  # type: ignore
        self.frame_count = 0
        logging.info("Initialized HeadlessView: No OpenGL context created.")

    def set_scene(self, scene: Scene) -> None:
        self.scene = scene

    def render(self, scene: Transform) -> None:
        """ Simulates rendering without drawing anything. Could save to file/GIF if needed. """
        logging.debug(f"HeadlessView: Simulated render for frame {self.frame_count}")
        self.frame_count += 1
        # No actual rendering, but logic could include:
        # - Exporting scene to image/GIF via CPU (optional)
        # - Logging data per frame
        # - Collecting metrics

    def get_framebuffer_size(self) -> Tuple[int, int]:
        """ Return dummy framebuffer size from config. """
        return (self.cfg.window_dimensions[0], self.cfg.window_dimensions[1])

    def swap_buffers(self) -> None:
        """ No buffers to swap in headless mode. """
        pass

    def clear_window(self) -> None:
        """ No OpenGL window to clear. """
        pass

    def cleanup(self) -> None:
        logging.info("HeadlessView cleanup: nothing to destroy.")
        # No OpenGL context to destroy
        # No shaders to delete