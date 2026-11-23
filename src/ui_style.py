"""Tk theme and palette shared by the Studio views."""

import tkinter as tk
from tkinter import font, ttk

BACKGROUND = "#151b19"
PANEL = "#202824"
WELL = "#181f1c"
BUTTON = "#303b35"
TEXT = "#f2f0e6"
MUTED = "#a8bab0"
ACCENT = "#d0e9a6"


def configure_style(root: tk.Tk) -> tuple[font.Font, ...]:
    root.configure(background=BACKGROUND)
    style = ttk.Style(root)
    style.theme_use("clam")
    body = font.nametofont("TkDefaultFont", root=root).copy()
    body.configure(size=10)
    heading = body.copy()
    heading.configure(size=28, weight="bold")
    section = body.copy()
    section.configure(size=12, weight="bold")
    mono = font.nametofont("TkFixedFont", root=root).copy()
    mono.configure(size=9)

    style.configure(".", background=BACKGROUND, foreground=TEXT, focuscolor=ACCENT, font=body)
    style.configure("Heading.TLabel", font=heading)
    style.configure("Muted.TLabel", foreground=MUTED)
    style.configure("Status.TLabel", foreground=ACCENT)
    style.configure("Counter.TLabel", foreground=MUTED, font=mono)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("Panel.TLabel", background=PANEL)
    style.configure("Section.Panel.TLabel", font=section)
    style.configure("Muted.Panel.TLabel", foreground=MUTED)
    style.configure("Value.Panel.TLabel", foreground=ACCENT, font=mono)
    style.configure(
        "TButton",
        padding=(13, 9),
        background=BUTTON,
        borderwidth=1,
        bordercolor=BUTTON,
        lightcolor=BUTTON,
        darkcolor=BUTTON,
    )
    style.map(
        "TButton",
        background=[("pressed", PANEL), ("active", "#3e4c43")],
        foreground=[("disabled", "#85928b")],
    )
    style.configure(
        "Accent.TButton",
        foreground=WELL,
        background=ACCENT,
        bordercolor=ACCENT,
        lightcolor=ACCENT,
        darkcolor=ACCENT,
        focuscolor=WELL,
    )
    style.map(
        "Accent.TButton",
        background=[("disabled", BUTTON), ("pressed", "#a9c880"), ("active", "#e3f4c9")],
        foreground=[("disabled", "#85928b")],
    )
    style.configure(
        "TCheckbutton",
        background=PANEL,
        indicatorbackground=WELL,
        indicatorforeground=ACCENT,
        padding=(0, 3),
    )
    style.map(
        "TCheckbutton",
        background=[("active", PANEL)],
        foreground=[("disabled", "#85928b")],
        indicatorbackground=[("selected", ACCENT)],
        indicatorforeground=[("selected", WELL)],
    )
    style.configure(
        "Horizontal.TScale",
        background=ACCENT,
        troughcolor=WELL,
        bordercolor=ACCENT,
        lightcolor=PANEL,
        darkcolor=PANEL,
        sliderlength=17,
    )
    style.configure(
        "TCombobox",
        fieldbackground=WELL,
        background=BUTTON,
        foreground=TEXT,
        selectbackground=WELL,
        selectforeground=TEXT,
        arrowcolor=ACCENT,
        bordercolor=MUTED,
        lightcolor=PANEL,
        darkcolor=PANEL,
        padding=4,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", WELL), ("disabled", PANEL)],
        foreground=[("disabled", MUTED)],
        selectbackground=[("readonly", WELL)],
    )
    style.configure(
        "TSpinbox",
        fieldbackground=WELL,
        background=BUTTON,
        foreground=TEXT,
        selectbackground=ACCENT,
        selectforeground=WELL,
        arrowcolor=ACCENT,
        bordercolor=MUTED,
        lightcolor=PANEL,
        darkcolor=PANEL,
        padding=4,
    )
    root.option_add("*TCombobox*Listbox.background", PANEL)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", WELL)
    style.configure("TSeparator", background="#55645a")
    style.configure("Training.TNotebook", background=BACKGROUND, bordercolor=PANEL)
    style.configure("Training.TNotebook.Tab", background=BUTTON, foreground=TEXT, padding=(12, 6))
    style.map(
        "Training.TNotebook.Tab",
        background=[("selected", PANEL)],
        foreground=[("selected", ACCENT)],
    )
    style.configure(
        "Info.TButton",
        padding=(4, 0),
        foreground=ACCENT,
        background=PANEL,
        borderwidth=0,
        bordercolor=PANEL,
        lightcolor=PANEL,
        darkcolor=PANEL,
    )
    style.map(
        "Info.TButton",
        background=[("pressed", PANEL), ("active", PANEL)],
        foreground=[("active", TEXT)],
    )
    style.configure(
        "Training.Info.TButton",
        background=BACKGROUND,
        bordercolor=BACKGROUND,
        lightcolor=BACKGROUND,
        darkcolor=BACKGROUND,
    )
    style.map("Training.Info.TButton", background=[("pressed", BACKGROUND), ("active", BACKGROUND)])
    style.configure(
        "HelpTooltip.TLabel",
        padding=(10, 7),
        background=WELL,
        foreground=TEXT,
        relief="solid",
        borderwidth=1,
    )
    return body, heading, section, mono
