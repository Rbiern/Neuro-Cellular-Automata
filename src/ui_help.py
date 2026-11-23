"""Hover tooltips and clickable option help for Studio controls."""

import tkinter as tk
from tkinter import messagebox, ttk


def info_button(
    parent: ttk.Frame, title: str, description: str, *, style: str = "Info.TButton"
) -> ttk.Button:
    button = ttk.Button(parent, text="ⓘ", width=2, style=style)
    tooltip: tk.Toplevel | None = None
    pending: str | None = None

    def hide(_event: tk.Event | None = None) -> None:
        nonlocal tooltip, pending
        if pending is not None:
            button.after_cancel(pending)
            pending = None
        if tooltip is not None:
            if tooltip.winfo_exists():
                tooltip.destroy()
            tooltip = None

    def show() -> None:
        nonlocal tooltip, pending
        pending = None
        tooltip = tk.Toplevel(button)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        tooltip.transient(button.winfo_toplevel())
        ttk.Label(
            tooltip, text=description, justify="left", wraplength=260, style="HelpTooltip.TLabel"
        ).pack()
        tooltip.update_idletasks()
        x = button.winfo_rootx()
        y = button.winfo_rooty() + button.winfo_height() + 4
        left, top = button.winfo_vrootx(), button.winfo_vrooty()
        right = left + button.winfo_vrootwidth()
        bottom = top + button.winfo_vrootheight()
        x = max(left, min(x, right - tooltip.winfo_reqwidth()))
        if y + tooltip.winfo_reqheight() > bottom:
            y = max(top, button.winfo_rooty() - tooltip.winfo_reqheight() - 4)
        tooltip.geometry(f"{x:+d}{y:+d}")
        tooltip.deiconify()

    def schedule(_event: tk.Event) -> None:
        nonlocal pending
        hide()
        pending = button.after(350, show)

    def open_help() -> None:
        hide()
        messagebox.showinfo(title, description, parent=parent.winfo_toplevel())

    button.configure(command=open_help)
    button.bind("<Enter>", schedule)
    for event in ("<Leave>", "<ButtonPress-1>", "<Destroy>"):
        button.bind(event, hide)
    return button
