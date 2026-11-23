"""Render the simulation canvas and map pointer positions to cells."""

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from ui_style import MUTED, TEXT, WELL


class GridView(tk.Canvas):
    """Fit the rendered image and map display coordinates to simulation cells."""

    def __init__(self, parent: ttk.Frame) -> None:
        super().__init__(
            parent,
            background=WELL,
            highlightthickness=1,
            highlightbackground=WELL,
            highlightcolor="#6e855f",
            takefocus=True,
            width=1,
            height=1,
        )
        self._image_item = self.create_image(0, 0, anchor="nw")
        self._placeholder = self.create_text(0, 0, text="Open a model to begin", fill=MUTED)
        self._brush_border = self.create_oval(
            0, 0, 0, 0, outline=WELL, width=2, state="hidden", tags="brush-preview"
        )
        self._brush_outline = self.create_oval(
            0, 0, 0, 0, outline=TEXT, width=1, state="hidden", tags="brush-preview"
        )
        self._brush_pointer: tuple[int, int, int] | None = None
        self._brush_geometry: tuple[int, int, float] | None = None
        self._image: Image.Image | None = None
        self._crisp_image = False
        self._photo: ImageTk.PhotoImage | None = None
        self._cells = 0
        self._show_grid = False
        self._viewport: tuple[int, int, int] | None = None
        self._grid_geometry: tuple[int, ...] | None = None
        self._redraw_job: str | None = None
        self.bind("<Configure>", self._request_redraw)

    def set_image(self, image: Image.Image, cells: int, *, crisp: bool = False) -> None:
        self._image, self._cells = image, cells
        self._crisp_image = crisp
        self._request_redraw()

    def show_grid(self, enabled: bool) -> None:
        self._show_grid = enabled
        self._request_redraw()

    def show_brush(self, x: int, y: int, radius: int) -> None:
        self._brush_pointer = x, y, radius
        self._draw_brush()

    def resize_brush(self, radius: int) -> None:
        if self._brush_pointer is not None:
            x, y, _ = self._brush_pointer
            self.show_brush(x, y, radius)

    def hide_brush(self) -> None:
        self._brush_pointer = None
        self._draw_brush()

    def _request_redraw(self, _event: tk.Event | None = None) -> None:
        if self._redraw_job is None:
            self._redraw_job = self.after_idle(self._redraw)

    def display_size(self) -> int:
        return max(0, min(self.winfo_width(), self.winfo_height()) - 40)

    def _redraw(self) -> None:
        self._redraw_job = None
        width, height = self.winfo_width(), self.winfo_height()
        if self._image is None:
            self.coords(self._placeholder, width // 2, height // 2)
            return
        side = self.display_size()
        if side < 1:
            self._viewport = None
            self.itemconfigure(self._image_item, state="hidden")
            self._draw_grid()
            self._draw_brush()
            return
        native_cells = self._image.width == self._cells
        if native_cells and side >= self._cells:
            side -= side % self._cells
        left, top = (width - side) // 2, (height - side) // 2
        self._viewport = left, top, side
        sampling = (
            Image.Resampling.NEAREST
            if native_cells or self._crisp_image
            else Image.Resampling.BICUBIC
        )
        resized = (
            self._image
            if self._image.size == (side, side)
            else self._image.resize((side, side), sampling)
        )
        self._photo = ImageTk.PhotoImage(resized, master=self)
        self.coords(self._image_item, left, top)
        self.itemconfigure(self._image_item, image=self._photo, state="normal")
        self.itemconfigure(self._placeholder, state="hidden")
        self._draw_grid()
        self._draw_brush()

    def _draw_grid(self) -> None:
        geometry = None
        if self._show_grid and self._viewport is not None and self._viewport[2] >= self._cells * 3:
            geometry = (*self._viewport, self._cells)
        if geometry == self._grid_geometry:
            return
        self.delete("cell-grid")
        self._grid_geometry = geometry
        if geometry is not None:
            left, top, side, cells = geometry
            for index in range(1, cells):
                offset = (index * side + cells - 1) // cells
                self.create_line(
                    left + offset,
                    top,
                    left + offset,
                    top + side,
                    fill="#546157",
                    stipple="gray50",
                    tags="cell-grid",
                )
                self.create_line(
                    left,
                    top + offset,
                    left + side,
                    top + offset,
                    fill="#546157",
                    stipple="gray50",
                    tags="cell-grid",
                )
            self.tag_raise("brush-preview")

    def _draw_brush(self) -> None:
        geometry = None
        viewport = self._viewport
        if viewport is not None and self._brush_pointer is not None:
            x, y, radius = self._brush_pointer
            if self.cell_at(x, y) is not None:
                geometry = x, y, (radius + 0.5) * viewport[2] / self._cells
        if geometry == self._brush_geometry:
            return
        if (geometry is None) != (self._brush_geometry is None):
            self.configure(cursor="" if geometry is None else "none")
        self._brush_geometry = geometry
        if geometry is None:
            self.itemconfigure("brush-preview", state="hidden")
            return

        x, y, radius = geometry
        for item in (self._brush_border, self._brush_outline):
            self.coords(item, x - radius, y - radius, x + radius, y + radius)
        self.itemconfigure("brush-preview", state="normal")

    def cell_at(self, x: int, y: int) -> tuple[int, int] | None:
        if self._viewport is None:
            return None
        left, top, side = self._viewport
        if left <= x < left + side and top <= y < top + side:
            return (x - left) * self._cells // side, (y - top) * self._cells // side
        return None

    def cancel_redraw(self) -> None:
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
            self._redraw_job = None
