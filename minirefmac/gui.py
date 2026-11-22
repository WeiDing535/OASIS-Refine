"""Native desktop front end for the OASIS-Refine CLI."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading

try:
    import tkinter as tk
    from tkinter import filedialog, font, messagebox, ttk
except ImportError:
    tk = None


CYCLE_LINE = re.compile(r"^\s*cycle\s+(\d+)\s+R=\s*([\d.]+)\s+Rfree=\s*([\d.]+|nan)")
ANGLE_LINE = re.compile(r"rmsANGL=([\d.]+)deg")
BOND_LINE = re.compile(r"rmsBOND=([\d.]+)")
AUTO_COLUMN = "(Auto)"
COLUMN_TYPES = {"fp": "F", "sigma": "Q", "free": "I"}
METRICS = (
    ("Rwork", 100, 2, "%", "#126754"),
    ("Rfree", 100, 2, "%", "#36579a"),
    ("Angle RMSD", 1, 2, "°", "#935420"),
    ("Bond length RMSD", 1, 4, "Å", "#79488a"),
)

CONFORMATIONS = ("Primary conformer", "All conformers · Fixed occupancy", "All conformers · Refine occupancy")
WINDOW_BG = "#eef2f0"
PAPER = "#ffffff"
INK = "#152826"
MUTED = "#4e6460"
ACCENT = "#126754"


def number(text, label, convert, minimum, maximum=None):
    try:
        value = convert(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{label} must be between {minimum} and {maximum}." if maximum is not None
                         else f"{label} must be at least {minimum}.")
    return value


def path_value(raw, label, file_required=True):
    if not raw.strip():
        raise ValueError(f"Select a value for {label}.")
    path = Path(raw.strip()).expanduser().resolve()
    if file_required and not path.is_file():
        raise ValueError(f"{label} must point to an existing file.\n{path}")
    if not file_required and path.exists() and not path.is_dir():
        raise ValueError(f"{label} must point to a directory.\n{path}")
    return path


class Desktop:
    def __init__(self, root, initial):
        self.root = root
        self.root.title("OASIS-Refine · Crystal structure refinement")
        self.root.geometry("1120x780")
        self.root.minsize(970, 720)
        self.root.configure(bg=WINDOW_BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.fields = {}
        self.cifs = ()
        self.cif_label = tk.StringVar(value="None selected")
        self.events = queue.Queue()
        self.process = None
        self.running = False
        self.closing = False
        self.cancelled = threading.Event()
        self.output = None
        self.job = None
        self.history_window = None
        self.r_work = tk.StringVar(value="—")
        self.r_free = tk.StringVar(value="—")
        self.angle = tk.StringVar(value="—")
        self.bond = tk.StringVar(value="—")
        self.status = tk.StringVar(value="Waiting for input files")
        self.output_label = tk.StringVar(value="Output files appear after refinement.")
        families = set(font.families(root))
        ui_font = next((name for name in ("Inter", "Segoe UI", "Helvetica")
                        if name in families), font.nametofont("TkDefaultFont").actual("family"))
        self.style(ui_font)
        self.points = {}
        self.saved_cycle = None
        self.layout(initial)
        self.column_cache = None
        self.column_refresh = None
        self.fields["mtz"].trace_add("write", self.queue_column_refresh)
        self.refresh_columns()
        self.refresh_history()

    def style(self, ui_font):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=(ui_font, 10), background=PAPER,
                        foreground=INK, bordercolor="#c4d2cd",
                        lightcolor="#dce6e2", darkcolor="#c4d2cd",
                        fieldbackground=PAPER, troughcolor="#e2ebe8")
        style.configure("TEntry", padding=4)
        style.configure("TCombobox", padding=4)
        style.configure("TButton", padding=(10, 5), background="#edf3f0")
        style.map("TButton", background=[("active", "#dcebe5")])
        style.configure("Run.TButton", padding=(16, 7), background=ACCENT, foreground=PAPER)
        style.map("Run.TButton", background=[("active", "#0e5646"), ("disabled", "#cbd9d4")],
                  foreground=[("disabled", "#4e6460")])
        style.configure("Title.TLabel", font=(ui_font, 24, "bold"),
                        background="#143934", foreground="#f7fbf9")
        style.configure("Header.TLabel", background="#143934", foreground="#bddbd2")
        style.configure("Section.TLabel", font=(ui_font, 13, "bold"))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Metric.TLabel", font=(ui_font, 20, "bold"), foreground=ACCENT)
        style.configure("TNotebook", background=PAPER, tabmargins=(0, 1, 0, 0))
        style.configure("TNotebook.Tab", padding=(12, 5))
        style.configure("Horizontal.TProgressbar", background=ACCENT)

    def layout(self, initial):
        head = tk.Frame(self.root, bg="#143934", padx=24, pady=12)
        head.pack(fill="x")
        ttk.Label(head, text="OASIS-Refine", style="Title.TLabel").pack(anchor="w")
        ttk.Label(head, text="Crystal structure refinement  ·  PDB / mmCIF + MTZ",
                  style="Header.TLabel").pack(anchor="w", pady=(2, 0))
        body = tk.Frame(self.root, bg=WINDOW_BG, padx=14, pady=14)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, padding=14, width=445)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.columnconfigure(0, weight=1)
        files = ttk.Frame(left)
        files.grid(row=0, column=0, sticky="ew")
        files.columnconfigure(1, weight=1)
        ttk.Label(files, text="Inputs and output", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self.file_row(files, 1, "MTZ", "mtz", initial.mtz,
                      (("MTZ reflections", "*.mtz"), ("All files", "*")))
        self.file_row(files, 2, "Model", "pdb", initial.pdb,
                      (("Coordinate model", "*.pdb *.cif *.mmcif *.ent"), ("All files", "*")))
        self.file_row(files, 3, "Output folder", "output", initial.out_dir, None)
        ttk.Label(files, text="Job name").grid(row=4, column=0, sticky="w")
        self.fields["name"] = tk.StringVar()
        ttk.Entry(files, textvariable=self.fields["name"]).grid(
            row=4, column=1, columnspan=2, sticky="ew", pady=(3, 10))

        tabs = ttk.Notebook(left, width=418, height=315)
        tabs.grid(row=1, column=0, sticky="nsew")
        basic = ttk.Frame(tabs, padding=(10, 8))
        advanced = ttk.Frame(tabs, padding=(10, 8))
        tabs.add(basic, text="Refinement")
        tabs.add(advanced, text="Weights and reflections")
        for frame in (basic, advanced):
            frame.columnconfigure(1, weight=1)
            frame.columnconfigure(3, weight=1)
        self.field(basic, 0, 0, "Cycles", "cycles", "5")
        self.field(basic, 0, 2, "Iterations per cycle", "iterations", "10")
        self.field(basic, 1, 0, "Kernel", "kernel", "auto", ("auto", "simd", "numpy", "fused"))
        self.field(basic, 1, 2, "SIMD threads", "threads", "4")
        self.field(basic, 2, 0, "Bulk solvent", "solvent", "mask", ("mask", "babinet", "none"))
        self.field(basic, 2, 2, "Angle weight", "angle", "1")
        self.mode_selector = self.field(basic, 3, 0, "Conformer mode", "mode", CONFORMATIONS[0],
                                        CONFORMATIONS, span=3)
        self.mode_selector.bind("<<ComboboxSelected>>", self.update_occupancy_state)
        self.fields["freeze"] = tk.BooleanVar(value=True)
        self.freeze = ttk.Checkbutton(basic, text="Freeze occupancies of nearly overlapping conformers",
                                      variable=self.fields["freeze"])
        self.freeze.grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 3))
        self.prior = self.field(basic, 5, 0, "Occupancy prior sigma", "prior", "", span=3)
        ttk.Label(basic, text="Leave sigma blank to disable the prior. Freezing applies to detected overlaps.",
                  style="Muted.TLabel", wraplength=350).grid(row=6, column=0, columnspan=4,
                                                            sticky="w", pady=(2, 10))

        self.fields["adaptive"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(basic, text="Use FreeR to adjust weights and select the output cycle",
                        variable=self.fields["adaptive"]).grid(row=7, column=0, columnspan=4,
                                                               sticky="w")
        ttk.Label(basic, text="By default, fit the work set and save the final cycle.",
                  style="Muted.TLabel").grid(row=8, column=0, columnspan=4, sticky="w", pady=3)
        self.update_occupancy_state()

        self.column_controls = {
            key: self.field(advanced, row, col, label, key, AUTO_COLUMN, (AUTO_COLUMN,))
            for key, row, col, label in (
                ("fp", 0, 0, "F column"), ("sigma", 0, 2, "SIGF column"),
                ("free", 1, 0, "FreeR column"))
        }
        self.field(advanced, 1, 2, "Free flag value", "free_value", "auto")
        self.column_status = tk.StringVar()
        ttk.Label(advanced, textvariable=self.column_status, wraplength=360,
                  style="Muted.TLabel").grid(row=2, column=0, columnspan=4, sticky="w", pady=(0, 5))
        self.field(advanced, 3, 0, "Min. d (Å)", "dmin", "0")
        self.field(advanced, 3, 2, "Max. d (Å)", "dmax", "0")
        self.field(advanced, 4, 0, "X-ray weight scale", "weight_scale", "0.05")
        self.field(advanced, 4, 2, "Fixed weight", "weight", "")
        ttk.Label(advanced, text="Use 0 for unlimited resolution. Leave fixed weight blank to balance gradients.",
                  style="Muted.TLabel", wraplength=350).grid(row=5, column=0, columnspan=4,
                                                            sticky="w", pady=(0, 12))
        ttk.Button(advanced, text="Select ligand CIF...", command=self.choose_cifs).grid(
            row=6, column=0, columnspan=2, sticky="w")
        ttk.Button(advanced, text="Clear", command=self.clear_cifs).grid(row=6, column=2,
                                                                       columnspan=2, sticky="e")
        ttk.Label(advanced, textvariable=self.cif_label, style="Muted.TLabel",
                  wraplength=350).grid(row=7, column=0, columnspan=4, sticky="w", pady=(6, 0))

        buttons = ttk.Frame(left)
        buttons.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.run_button = ttk.Button(buttons, text="Start refinement", style="Run.TButton", command=self.start)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)

        right = ttk.Frame(body, padding=14)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)
        ttk.Label(right, text="Run and results", style="Section.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Label(right, textvariable=self.status, style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(5, 4))
        self.progress = ttk.Progressbar(right, mode="determinate")
        self.progress.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(right, text="Job history", command=self.show_history).grid(row=0, column=0, sticky="e")
        self.results_tabs = ttk.Notebook(right)
        self.results_tabs.grid(row=3, column=0, sticky="nsew")
        progress_tab = ttk.Frame(self.results_tabs)
        log_area = ttk.Frame(self.results_tabs)
        self.results_tabs.add(progress_tab, text="Progress")
        self.results_tabs.add(log_area, text="Run log")
        self.log_tab = log_area
        self.plot = tk.Canvas(progress_tab, bg=PAPER, highlightthickness=0, width=360, height=260)
        self.plot.pack(fill="both", expand=True)
        self.plot.bind("<Configure>", self.draw_curves)
        log_area.columnconfigure(0, weight=1)
        log_area.rowconfigure(0, weight=1)
        self.log_widget = tk.Text(log_area, width=40, height=10, wrap="word",
                                  bg="#122925", fg="#d9eeea", insertbackground=PAPER,
                                  relief="flat", padx=11, pady=9, state="disabled",
                                  font=("TkFixedFont", 10))
        self.log_widget.grid(row=0, column=0, sticky="nsew")
        ybar = ttk.Scrollbar(log_area, command=self.log_widget.yview)
        ybar.grid(row=0, column=1, sticky="ns")
        self.log_widget.configure(yscrollcommand=ybar.set)
        ttk.Label(right, textvariable=self.output_label, wraplength=470,
                  style="Muted.TLabel").grid(row=4, column=0, sticky="w", pady=(9, 4))
        outputs = ttk.Frame(right)
        outputs.grid(row=5, column=0, sticky="w")
        self.output_buttons = {}
        for label, name in (("Open folder", ""), ("PDB", "refined.pdb"),
                            ("MTZ", "refined.mtz"), ("JSON report", "report.json")):
            button = ttk.Button(outputs, text=label, width=-1, command=lambda n=name: self.open_output(n),
                                state="disabled")
            button.pack(side="left", padx=(0, 5))
            self.output_buttons[name] = button

    def field(self, parent, row, column, label, key, initial, choices=None, span=1):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 6), pady=4)
        value = self.fields[key] = tk.StringVar(value=initial)
        control = (ttk.Entry(parent, textvariable=value, width=9) if choices is None else
                   ttk.Combobox(parent, textvariable=value, values=choices, state="readonly", width=8))
        control.grid(row=row, column=column + 1, columnspan=span,
                     sticky="ew", padx=(0, 7), pady=3)
        return control

    def file_row(self, parent, row, label, key, initial, filetypes):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        value = self.fields[key] = tk.StringVar(value=initial)
        ttk.Entry(parent, textvariable=value, width=32).grid(row=row, column=1, sticky="ew", pady=3)

        def choose_file():
            if filetypes is None:
                path = filedialog.askdirectory(parent=self.root)
            else:
                path = filedialog.askopenfilename(parent=self.root, filetypes=filetypes)
            if path:
                value.set(path)
                if key == "output":
                    self.refresh_history()

        ttk.Button(parent, text="Browse...", command=choose_file).grid(
            row=row, column=2, sticky="e", padx=(7, 0), pady=3)

    def queue_column_refresh(self, *_):
        if self.column_refresh is not None:
            self.root.after_cancel(self.column_refresh)
        self.column_refresh = self.root.after(250, self.refresh_columns)

    def refresh_columns(self):
        if self.column_refresh is not None:
            self.root.after_cancel(self.column_refresh)
            self.column_refresh = None
        raw = self.fields["mtz"].get().strip()
        choices = {}
        try:
            if raw:
                import gemmi

                path = Path(raw).expanduser().resolve()
                stat = path.stat()
                fingerprint = (path, stat.st_mtime_ns, stat.st_size)
                if self.column_cache == fingerprint:
                    return
                mtz = gemmi.read_mtz_file(str(path))
                choices = {
                    key: tuple(dict.fromkeys(col.label for col in mtz.columns if col.type == kind))
                    for key, kind in COLUMN_TYPES.items()
                }
                self.column_cache = fingerprint
        except (OSError, RuntimeError, ValueError):
            pass
        if not choices:
            self.column_cache = None
        for key, control in self.column_controls.items():
            values = (AUTO_COLUMN, *choices.get(key, ()))
            control.configure(values=values)
            if self.fields[key].get() not in values:
                self.fields[key].set(AUTO_COLUMN)
        if choices.get("fp") and choices.get("sigma"):
            message = "Choose MTZ labels or keep (Auto)."
        elif choices:
            message = "This MTZ is missing an amplitude or sigma column."
        else:
            message = "Unable to read MTZ columns." if raw else "Select an MTZ to list its columns."
        self.column_status.set(message)

    def record_cycle_line(self, line):
        match = CYCLE_LINE.match(line)
        if match is None:
            return None
        cycle = int(match[1])
        angle = ANGLE_LINE.search(line)
        bond = BOND_LINE.search(line)
        self.points[cycle] = (
            float(match[2]), float(match[3]),
            float(angle[1]) if angle else math.nan, float(bond[1]) if bond else math.nan,
        )
        self.set_metric_values(self.points[cycle])
        return cycle

    def set_metric_values(self, values):
        for var, value, (_, scale, places, unit, _) in zip(
                (self.r_work, self.r_free, self.angle, self.bond), values, METRICS):
            var.set(f"{value * scale:.{places}f}{unit}" if math.isfinite(value) else "—")

    def load_report_curves(self, cycles, saved_cycle):
        keys = ("r_work", "r_free", "rmsd_angle_deg", "rmsd_bond")
        for row in cycles:
            self.points[int(row["cycle"])] = tuple(
                float(row[key]) if row.get(key) is not None else math.nan for key in keys
            )
        self.saved_cycle = int(saved_cycle)
        if self.saved_cycle in self.points:
            self.set_metric_values(self.points[self.saved_cycle])
        self.draw_curves()

    def draw_curves(self, _event=None):
        canvas = self.plot
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 250 or height < 150:
            return
        tile_w, tile_h = width / 2, height / 2
        last = max(self.points, default=0)
        max_cycle = max(last, 1)
        variables = (self.r_work, self.r_free, self.angle, self.bond)
        for i, (label, scale, places, unit, color) in enumerate(METRICS):
            x, y = (i % 2) * tile_w, (i // 2) * tile_h
            canvas.create_rectangle(x + 5, y + 5, x + tile_w - 5, y + tile_h - 5,
                                    fill="#f8fbf9", outline="#dce6e2")
            canvas.create_text(x + 14, y + 17, text=label, anchor="w", fill=INK)
            canvas.create_text(x + 14, y + 37, text=variables[i].get(), anchor="w",
                               fill=color, font=("TkDefaultFont", 12, "bold"))
            series = [(cycle, row[i] * scale) for cycle, row in sorted(self.points.items())
                      if math.isfinite(row[i])]
            x0, x1 = x + 56, x + tile_w - 19
            y0, y1 = y + tile_h - 27, y + 55
            canvas.create_line(x0, y1, x0, y0, x1, y0, fill="#a6b6b0")
            for cycle in (0, last) if last else (0,):
                tick = x0 + cycle / max_cycle * (x1 - x0)
                canvas.create_text(tick, y0 + 9, text=str(cycle), fill=MUTED)
            canvas.create_text((x0 + x1) / 2, y0 + 9, text="Cycle", fill=MUTED)
            if not series:
                message = "Unavailable" if self.points else "Waiting for cycle data"
                canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=message, fill=MUTED)
                continue
            low, high = min(v for _, v in series), max(v for _, v in series)
            pad = max((high - low) * 0.12, 10 ** -places)
            low, high = max(0, low - pad), high + pad
            for v in (low, high):
                tick = y0 - (v - low) / (high - low) * (y0 - y1)
                canvas.create_text(x0 - 4, tick, text=f"{v:.{places}f}", anchor="e", fill=MUTED)
            coords = [(x0 + cycle / max_cycle * (x1 - x0),
                       y0 - (v - low) / (high - low) * (y0 - y1)) for cycle, v in series]
            if len(coords) > 1:
                canvas.create_line(*coords, fill=color, width=2, tags=("curve",))
            for (cx, cy), (cycle, _) in zip(coords, series):
                radius = 4 if cycle == self.saved_cycle else 2.5
                canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                   fill=color, outline=color)

    def write_job(self):
        pending = self.output / ".job.json.tmp"
        pending.write_text(json.dumps(self.job, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        pending.replace(self.output / "job.json")

    def show_history(self):
        if self.history_window is not None and self.history_window.winfo_exists():
            self.history_window.deiconify()
            self.history_window.lift()
            self.refresh_history()
            return
        window = self.history_window = tk.Toplevel(self.root)
        window.title("OASIS-Refine · Job history")
        window.geometry("720x380")
        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        self.history_folder = ttk.Label(frame, wraplength=660)
        self.history_folder.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.history_tree = ttk.Treeview(frame, columns=("name", "status", "created"), show="headings",
                                         selectmode="browse")
        for key, title, width in (("name", "Job name", 320), ("status", "Status", 140),
                                  ("created", "Started", 175)):
            self.history_tree.heading(key, text=title)
            self.history_tree.column(key, width=width)
        self.history_tree.grid(row=1, column=0, sticky="nsew")
        bar = ttk.Scrollbar(frame, command=self.history_tree.yview)
        bar.grid(row=1, column=1, sticky="ns")
        self.history_tree.configure(yscrollcommand=bar.set)
        self.history_tree.bind("<Double-1>", lambda _: self.view_history())
        controls = ttk.Frame(frame)
        controls.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for title, command in (("View log", self.view_history), ("Reuse parameters", self.reuse_history),
                               ("Refresh", self.refresh_history), ("Browse folder...", self.pick_history_folder)):
            ttk.Button(controls, text=title, command=command).pack(side="left", padx=(0, 6))
        self.history_parent = None
        self.refresh_history()

    def pick_history_folder(self):
        folder = filedialog.askdirectory(parent=self.history_window)
        if folder:
            self.history_parent = Path(folder)
            self.refresh_history()

    def refresh_history(self):
        if self.history_window is None or not self.history_window.winfo_exists():
            return
        raw = self.fields["output"].get().strip()
        folder = self.history_parent or (Path(raw).expanduser() if raw else Path.cwd())
        self.history_folder.configure(text=str(folder))
        tree = self.history_tree
        selected = tree.selection()
        tree.delete(*tree.get_children())
        try:
            jobs = []
            for path in folder.iterdir():
                if not path.is_dir() or not ((path / "job.json").is_file() or (path / "gui.log").is_file()):
                    continue
                job = self.read_job(path)
                jobs.append((job["created"], path, job))
            for _, path, job in sorted(jobs, reverse=True):
                tree.insert("", "end", iid=str(path.resolve()),
                            values=(job["name"], job["status"], job["created"][:19].replace("T", " ")))
            if selected and tree.exists(selected[0]):
                tree.selection_set(selected[0])
        except OSError as exc:
            self.history_folder.configure(text=f"Unable to read jobs in {folder}. {exc}")

    @staticmethod
    def read_job(path):
        try:
            job = json.loads((path / "job.json").read_text(encoding="utf-8"))
            if isinstance(job, dict) and all(isinstance(job.get(key), str)
                                              for key in ("name", "status", "created")):
                return job
        except (OSError, ValueError):
            pass
        return {"name": path.name, "status": "Complete" if (path / "report.json").is_file()
                else "Incomplete", "created": datetime.fromtimestamp(path.stat().st_mtime).isoformat()}

    def selected_history(self):
        selection = self.history_tree.selection()
        return Path(selection[0]) if selection else None

    def update_output_buttons(self, complete):
        for name, button in self.output_buttons.items():
            path = self.output / name
            button.configure(state="normal" if path.exists() and (complete or not name) else "disabled")

    def view_history(self):
        path = self.selected_history()
        if path is None:
            return
        if self.running:
            messagebox.showinfo("Job history", "Stop or finish the current run before switching jobs.",
                                parent=self.history_window)
            return
        self.job = None
        self.output = path
        job = self.read_job(path)
        self.output_label.set(f"Job · {job['name']}")
        self.status.set(f"Job history · {job['status']}")
        self.points.clear()
        self.saved_cycle = None
        self.set_metric_values((math.nan,) * len(METRICS))
        self.progress.configure(value=0)
        log_path = path / "gui.log"
        if not log_path.is_file():
            log_path = path / "refine.log"
        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = "No log is available for this job.\n"
        self.log_widget.configure(state="normal")
        self.log_widget.delete("1.0", "end")
        self.log_widget.insert("end", content)
        self.log_widget.configure(state="disabled")
        for line in content.splitlines():
            self.record_cycle_line(line)
        try:
            report = json.loads((path / "report.json").read_text(encoding="utf-8"))
            self.load_report_curves(report.get("cycles") or [], (report.get("written") or report["final"])["cycle"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self.draw_curves()
        self.update_output_buttons(True)
        self.results_tabs.select(self.log_tab)
        self.history_window.withdraw()
        self.root.lift()

    @staticmethod
    def legacy_parameters(path):
        from .cli import build_parser

        with (path / "gui.log").open(encoding="utf-8") as log:
            command = shlex.split(log.readline())
        index = command.index("minirefmac")
        try:
            options = build_parser().parse_args(command[index + 1:])
        except SystemExit as exc:
            raise ValueError("Unable to read the original command.") from exc
        values = {key: getattr(options, opt) for key, opt in (
            ("mtz", "mtz"), ("pdb", "pdb"), ("fp", "fp"), ("sigma", "sigfp"), ("free", "free"),
            ("cycles", "ncyc"), ("iterations", "maxiter"), ("kernel", "kernel"),
            ("threads", "simd_threads"), ("solvent", "solvent"), ("angle", "w_angle"),
            ("prior", "occupancy_prior_sigma"), ("weight", "weight"), ("weight_scale", "weight_scale"),
            ("free_value", "free_value"), ("dmin", "dmin"), ("dmax", "dmax"),
            ("freeze", "freeze_overlapping_occupancies"))}
        mode = 2 if options.refine_occupancies else int(options.all_altlocs)
        values.update(output=str(path.parent), name=Path(options.pdb).stem,
                      mode=CONFORMATIONS[mode], adaptive=not options.no_auto_weight)
        return values, options.restraint_cif

    def reuse_history(self):
        path = self.selected_history()
        if path is None:
            return
        try:
            job = self.read_job(path)
            if "parameters" in job:
                values, cifs = job["parameters"], job.get("ligand_cifs", [])
            else:
                values, cifs = self.legacy_parameters(path)
            self.fields["mtz"].set(values["mtz"])
            self.refresh_columns()
            for key, value in values.items():
                if key in self.column_controls and not value:
                    value = AUTO_COLUMN
                if key in self.fields:
                    self.fields[key].set("" if value is None else value)
            self.cifs = tuple(cifs)
            self.cif_label.set(", ".join(Path(p).name for p in cifs) or "None selected")
            self.update_occupancy_state()
            self.history_window.withdraw()
            self.root.lift()
        except (OSError, KeyError, ValueError, tk.TclError) as exc:
            messagebox.showerror("Unable to reuse parameters", str(exc), parent=self.history_window)

    def choose_cifs(self):
        paths = filedialog.askopenfilenames(parent=self.root,
                                            filetypes=(("Component dictionary", "*.cif"), ("All files", "*")))
        if paths:
            self.cifs = tuple(paths)
            self.cif_label.set(", ".join(Path(p).name for p in paths))

    def clear_cifs(self):
        self.cifs = ()
        self.cif_label.set("None selected")

    def update_occupancy_state(self, _event=None):
        state = "normal" if self.fields["mode"].get() == CONFORMATIONS[2] else "disabled"
        self.freeze.configure(state=state)
        self.prior.configure(state=state)

    def prepare(self):
        values = {key: value.get() for key, value in self.fields.items()}
        v = values
        mtz = path_value(v["mtz"], "MTZ")
        self.refresh_columns()
        if self.column_cache is None:
            raise ValueError("Unable to read the MTZ column labels.")
        for key, control in self.column_controls.items():
            if v[key] and v[key] not in control["values"]:
                raise ValueError(f"Select a valid {key} column for this MTZ.")
        model = path_value(v["pdb"], "Coordinate model")
        output = path_value(v["output"], "Output folder", False)
        cycles = number(v["cycles"], "Cycles", int, 0)
        iterations = number(v["iterations"], "Iterations per cycle", int, 1)
        threads = number(v["threads"], "SIMD threads", int, 1, 64)
        angle = number(v["angle"], "Angle weight", float, 0)
        dmin = number(v["dmin"], "Min. d", float, 0)
        dmax = number(v["dmax"], "Max. d", float, 0)
        if dmin and dmax and dmin > dmax:
            raise ValueError("Min. d must be less than or equal to max. d.")
        free_value = v["free_value"].strip().lower() or "auto"
        if free_value != "auto":
            try:
                free_value = str(int(free_value))
            except ValueError as exc:
                raise ValueError("The FreeR flag value must be an integer or auto.") from exc

        args = ["--mtz", str(mtz), "--pdb", str(model),
                "--ncyc", str(cycles), "--maxiter", str(iterations),
                "--kernel", v["kernel"], "--simd-threads", str(threads),
                "--solvent", v["solvent"], "--w-angle", str(angle),
                "--free-value", free_value, "--dmin", str(dmin), "--dmax", str(dmax)]
        if v["weight"].strip():
            args += ["--weight", str(number(v["weight"], "Fixed weight", float, 0))]
        else:
            args += ["--weight-scale", str(number(v["weight_scale"], "X-ray weight scale", float, 0))]
        for key, flag in (("fp", "--fp"), ("sigma", "--sigfp"), ("free", "--free")):
            if v[key].strip() and v[key] != AUTO_COLUMN:
                args += [flag, v[key].strip()]
        for path in self.cifs:
            args += ["--restraint-cif", str(path_value(path, "Ligand CIF"))]
        if v["mode"] == CONFORMATIONS[1]:
            args += ["--all-altlocs"]
        elif v["mode"] == CONFORMATIONS[2]:
            args += ["--refine-occupancies"]
            if v["freeze"]:
                args += ["--freeze-overlapping-occupancies"]
            if v["prior"].strip():
                prior = number(v["prior"], "Occupancy prior sigma", float, 0)
                if prior == 0:
                    raise ValueError("Occupancy prior sigma must be greater than zero.")
                args += ["--occupancy-prior-sigma", str(prior)]
        if not v["adaptive"]:
            args += ["--no-auto-weight", "--no-keep-best", "--early-stop", "off"]
        return output, cycles, args

    def start(self):
        if self.running:
            return
        try:
            output, cycles, args = self.prepare()
            output.mkdir(parents=True, exist_ok=True)
            name = self.fields["name"].get().strip() or Path(self.fields["pdb"].get().strip()).stem
            prefix = re.sub(r"[^\w.-]+", "-", name, flags=re.ASCII).strip(".-")[:40] or "job"
            self.output = Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=output))
            parameters = {key: field.get() for key, field in self.fields.items()}
            parameters.update(mtz=args[1], pdb=args[3], output=str(output), name=name)
            self.job = {"name": name, "created": datetime.now().astimezone().isoformat(),
                        "status": "Running / interrupted", "parameters": parameters,
                        "ligand_cifs": [str(path_value(path, "Ligand CIF")) for path in self.cifs]}
            self.write_job()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Unable to start refinement", str(exc), parent=self.root)
            return
        self.expected_cycles = cycles
        self.cancelled = threading.Event()
        self.running = True
        self.status.set("Loading model and reflections...")
        self.output_label.set(f"Run folder · {self.output.name}")
        self.r_work.set("—")
        self.r_free.set("—")
        self.angle.set("—")
        self.bond.set("—")
        self.points.clear()
        self.saved_cycle = None
        self.draw_curves()
        self.results_tabs.select(0)
        self.refresh_history()
        self.progress.configure(maximum=max(1, cycles), value=0)
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        for key, button in self.output_buttons.items():
            button.configure(state="disabled" if key else "normal")
        self.log_widget.configure(state="normal")
        self.log_widget.delete("1.0", "end")
        self.log_widget.configure(state="disabled")
        cmd = [sys.executable, "-u", "-m", "minirefmac", *args, "--out-dir", str(self.output)]
        threading.Thread(target=self.run_process, args=(cmd, self.output), daemon=True).start()
        self.root.after(65, self.poll)

    def run_process(self, command, output):
        code, error = -1, None
        try:
            env = os.environ.copy()
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("OMP_NUM_THREADS", "1")
            with (output / "gui.log").open("w", encoding="utf-8") as logfile:
                logfile.write(shlex.join(command) + "\n\n")
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding="utf-8", errors="replace", env=env,
                                      start_new_session=(os.name == "posix")) as proc:
                    self.process = proc
                    if self.cancelled.is_set():
                        self.signal_process(proc)
                    for line in proc.stdout:
                        logfile.write(line)
                        logfile.flush()
                        self.events.put(("line", line))
                    code = proc.wait()
        except (OSError, ValueError) as exc:
            error = str(exc)
            try:
                with (output / "gui.log").open("a", encoding="utf-8") as logfile:
                    logfile.write(f"\nUnable to run refinement. {error}\n")
            except OSError:
                pass
        finally:
            self.process = None
            self.events.put(("done", (code, error)))

    def poll(self):
        lines = []
        done = None
        updated_curves = False
        for _ in range(400):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                lines.append(value.replace(str(self.output) + os.sep, "./"))
                cycle = self.record_cycle_line(value)
                if cycle is not None:
                    updated_curves = True
                    self.progress["value"] = cycle
                    if not self.cancelled.is_set():
                        self.status.set(f"Refining · Cycle {cycle} of {self.expected_cycles}")
            else:
                done = value
                break
        if lines:
            self.append_log("".join(lines))
        if updated_curves:
            self.draw_curves()
        if done is None:
            self.root.after(65, self.poll)
        else:
            self.finish(*done)

    def append_log(self, text):
        follow = self.log_widget.yview()[1] >= 0.995
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", text)
        self.log_widget.configure(state="disabled")
        if follow:
            self.log_widget.see("end")

    @staticmethod
    def percent(value):
        try:
            number_value = float(value)
        except (TypeError, ValueError):
            return "—"
        return f"{number_value:.2%}" if math.isfinite(number_value) else "—"

    def finish(self, code, error):
        self.running = False
        self.stop_button.configure(state="disabled")
        self.run_button.configure(state="normal")
        if error:
            self.append_log(f"\n{error}\n")
        complete = False
        if self.cancelled.is_set():
            self.status.set("Stopped · Run log saved")
        elif code != 0:
            self.status.set(f"Run failed (exit code {code}) · See the log")
        else:
            try:
                report = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
                if not all((self.output / name).is_file() for name in ("refined.pdb", "refined.mtz")):
                    raise ValueError("Refined PDB or MTZ is missing.")
                written = report.get("written") or report["final"]
                cycle = written.get("cycle", 0)
                cycles = report.get("cycles") or []
                match = next((row for row in cycles if row.get("cycle") == cycle), {})
                self.load_report_curves(cycles, cycle)
                self.r_work.set(self.percent(written.get("r_work")))
                self.r_free.set(self.percent(written.get("r_free")))
                angle = match.get("rmsd_angle_deg")
                if isinstance(angle, (float, int)) and math.isfinite(angle):
                    self.angle.set(f"{angle:.2f}°")
                self.progress["value"] = max(1, self.expected_cycles)
                if cycle == 0 and len(cycles) > 1:
                    self.status.set("Refinement complete · Saved starting model (cycle 0)")
                else:
                    self.status.set(f"Refinement complete · Saved cycle {cycle} · {report.get('kernel', '—')}")
                if cycles:
                    start_free = self.percent(cycles[0].get("r_free"))
                    last_free = self.percent(cycles[-1].get("r_free"))
                    self.append_log(
                        f"Rfree: start {start_free}; last cycle {last_free}; "
                        f"saved cycle {cycle} {self.r_free.get()}\n"
                    )
                complete = True
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.status.set("Run ended · Unable to read the report")
                self.append_log(f"\nUnable to read the report. {exc}\n")
        self.update_output_buttons(complete)
        if self.job is not None:
            self.job["status"] = "Complete" if complete else ("Stopped" if self.cancelled.is_set() else "Failed")
            try:
                self.write_job()
            except OSError as exc:
                self.append_log(f"Unable to update job history. {exc}\n")
            self.refresh_history()
        if self.closing:
            self.root.destroy()

    def stop(self):
        if not self.running:
            return
        self.cancelled.set()
        self.status.set("Stopping refinement...")
        self.stop_button.configure(state="disabled")
        proc = self.process
        if proc is not None and proc.poll() is None:
            self.signal_process(proc)
        token = self.cancelled
        self.root.after(3000, lambda: self.force_stop(token))

    def force_stop(self, token):
        proc = self.process
        if self.cancelled is token and proc is not None and proc.poll() is None:
            self.signal_process(proc, force=True)

    @staticmethod
    def signal_process(proc, force=False):
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif force:
                proc.kill()
            else:
                proc.terminate()
        except ProcessLookupError:
            pass

    def close(self):
        if not self.running:
            self.root.destroy()
        elif messagebox.askyesno("Close OASIS-Refine", "Stop the current run and close the window?", parent=self.root):
            self.closing = True
            self.stop()

    def open_output(self, name):
        if self.output is None:
            return
        path = self.output / name
        try:
            if sys.platform == "win32":
                os.startfile(str(path))
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                proc = subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                self.root.after(500, lambda: self.check_opener(proc, path))
        except OSError as exc:
            messagebox.showerror("Unable to open result", f"{path}\n{exc}", parent=self.root)

    def check_opener(self, proc, path):
        if proc.poll() not in (None, 0):
            messagebox.showerror("Unable to open result", f"Open this file with an appropriate application.\n{path}", parent=self.root)


def main(argv=None):
    parser = argparse.ArgumentParser(description="OASIS-Refine desktop GUI")
    parser.add_argument("--mtz", default="", help="preselect an input MTZ")
    parser.add_argument("--pdb", default="", help="preselect an input PDB/mmCIF")
    parser.add_argument("--out-dir", default=str(Path.cwd() / "oasis_gui_runs"),
                        help="parent directory for new runs")
    args = parser.parse_args(argv)
    if tk is None:
        print("Tkinter is unavailable. Use a Python installation with Tk support.", file=sys.stderr)
        return 2
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Cannot start OASIS-Refine GUI: {exc}", file=sys.stderr)
        return 2
    Desktop(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
