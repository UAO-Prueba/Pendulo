import cv2
import os
import csv
import json
import subprocess
import threading
import datetime
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  GIT
# ─────────────────────────────────────────────
def _find_git() -> str:
    import shutil
    g = shutil.which("git")
    if g: return g
    for c in [r"C:\Program Files\Git\cmd\git.exe",
              r"C:\Program Files\Git\bin\git.exe"]:
        if os.path.isfile(c): return c
    return "git"

def git_auto_commit(exp_dir, message):
    log = []
    try:
        git  = _find_git()
        root = os.path.dirname(os.path.abspath(__file__))
        def run(args):
            full = [git] + args
            log.append("$ " + " ".join(full))
            r = subprocess.run(full, cwd=root, capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
            if r.stdout.strip(): log.append("  " + r.stdout.strip())
            if r.stderr.strip(): log.append("  " + r.stderr.strip())
            return r
        run(["init"])
        run(["config", "user.email", "pendulo@lab"])
        run(["config", "user.name",  "Pendulo Analyzer"])
        gi = os.path.join(root, ".gitignore")
        if not os.path.exists(gi):
            open(gi, "w").write("__pycache__/\n*.pyc\n")
            run(["add", ".gitignore"])
        run(["add", "-A"])
        cr = run(["commit", "-m", message])
        out = (cr.stdout + cr.stderr).lower()
        if cr.returncode != 0 and "nothing to commit" not in out:
            return False, "\n".join(log)
        log.append("COMMIT OK")
        rr = run(["remote", "-v"])
        if not rr.stdout.strip():
            log.append("Sin remote — push omitido"); return True, "\n".join(log)
        br = run(["rev-parse", "--abbrev-ref", "HEAD"])
        branch = br.stdout.strip() or "main"
        pr = run(["push", "origin", branch])
        if pr.returncode != 0:
            run(["push", "--set-upstream", "origin", branch])
        log.append("PUSH OK")
        return True, "\n".join(log)
    except Exception as e:
        log.append(str(e)); return False, "\n".join(log)

def next_experiment_folder(base):
    os.makedirs(base, exist_ok=True)
    nums = []
    for d in os.listdir(base):
        if d.startswith("experimento_"):
            try: nums.append(int(d.split("_")[1]))
            except: pass
    n = max(nums) + 1 if nums else 1
    return os.path.join(base, f"experimento_{n:03d}")


# ─────────────────────────────────────────────
#  ANÁLISIS MATEMÁTICO
# ─────────────────────────────────────────────
def smooth(arr, w=5):
    return np.convolve(arr, np.ones(w) / w, mode='same')

def seno_amort(t, A, omega, phi, gamma, offset):
    return A * np.exp(-gamma * t) * np.cos(omega * t + phi) + offset

def calcular_resultados(t, x_data, y_data, theta, L_cuerda):
    x_s = smooth(x_data); y_s = smooth(y_data)
    vx  = smooth(np.gradient(x_s, t))
    vy  = smooth(np.gradient(y_s, t))
    v_total = np.sqrt(vx**2 + vy**2)

    theta_adj = np.deg2rad(np.mod(np.degrees(theta) + 720, 360) - 180)

    ajuste_ok = False
    popt = None
    T_fit = g_exp = A_fit = omega_fit = gamma_fit = phi_fit = off_fit = 0.0

    try:
        A0    = (np.max(theta_adj) - np.min(theta_adj)) / 2
        off0  = np.mean(theta_adj)
        cruces = np.where(np.diff(np.sign(theta_adj - off0)))[0]
        if len(cruces) >= 2:
            T_est = 2 * np.mean(np.diff(t[cruces]))
            omega0 = 2 * np.pi / T_est if T_est > 0 else 3.0
        else:
            omega0 = 3.0
        p0 = [A0, omega0, 0.0, 0.01, off0]
        popt, _ = curve_fit(seno_amort, t, theta_adj, p0=p0, maxfev=15000,
                            bounds=([-np.pi, 0.1, -np.pi, 0, -np.pi],
                                    [ np.pi,  50,  np.pi,  5,  np.pi]))
        A_fit, omega_fit, phi_fit, gamma_fit, off_fit = popt
        T_fit = 2 * np.pi / omega_fit
        g_exp = (2 * np.pi / T_fit) ** 2 * L_cuerda
        ajuste_ok = True
    except Exception as e:
        print(f"Ajuste falló: {e}")

    return {
        "x_s": x_s, "y_s": y_s, "vx": vx, "vy": vy,
        "v_total": v_total, "theta_adj": theta_adj,
        "ajuste_ok": ajuste_ok, "popt": popt,
        "T_fit": T_fit, "g_exp": g_exp,
        "A_fit": A_fit, "omega_fit": omega_fit,
        "gamma_fit": gamma_fit, "phi_fit": phi_fit, "off_fit": off_fit,
    }


# ─────────────────────────────────────────────
#  GRÁFICAS
# ─────────────────────────────────────────────
def _style_ax(ax):
    ax.set_facecolor("#1a1a2e")
    for sp in ax.spines.values(): sp.set_color("#333366")
    ax.tick_params(colors="#aaaacc", labelsize=8)
    ax.yaxis.label.set_color("#aaaacc")
    ax.xaxis.label.set_color("#aaaacc")
    ax.title.set_color("#e0e0ff")
    ax.grid(alpha=0.2)

def generate_fig1(t, x_data, y_data, r, path):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axs.flat: _style_ax(ax)

    axs[0,0].plot(t, x_data, 'o', ms=3, alpha=0.35, color='#00BFFF')
    axs[0,0].plot(t, r["x_s"], '-', lw=2, color='#00BFFF', label='Suavizado')
    axs[0,0].set_title("Posición X"); axs[0,0].set_ylabel("x (m)"); axs[0,0].legend(fontsize=8)

    axs[0,1].plot(t, y_data, 'o', ms=3, alpha=0.35, color='#FF6347')
    axs[0,1].plot(t, r["y_s"], '-', lw=2, color='#FF6347', label='Suavizado')
    axs[0,1].set_title("Posición Y"); axs[0,1].set_ylabel("y (m)"); axs[0,1].legend(fontsize=8)

    axs[1,0].plot(t, r["vx"], '-', lw=2, color='#7FFF00')
    axs[1,0].axhline(0, color='gray', lw=0.5, linestyle=':')
    axs[1,0].set_title("Velocidad X"); axs[1,0].set_ylabel("vx (m/s)"); axs[1,0].set_xlabel("t (s)")

    axs[1,1].plot(t, r["vy"], '-', lw=2, color='#FFD700')
    axs[1,1].axhline(0, color='gray', lw=0.5, linestyle=':')
    axs[1,1].set_title("Velocidad Y"); axs[1,1].set_ylabel("vy (m/s)"); axs[1,1].set_xlabel("t (s)")

    fig.suptitle("Péndulo — Posición y Velocidad", color="#fff", fontsize=13, fontweight="bold")
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)

def generate_fig2(t, theta, r, path):
    theta_deg = np.mod(np.degrees(theta) + 720, 360) - 180
    fig, ax = plt.subplots(figsize=(11, 5)); fig.patch.set_facecolor("#0d0d0d"); _style_ax(ax)
    ax.plot(t, theta_deg, 'o', ms=3, alpha=0.4, color='#DA70D6', label='θ datos')
    if r["ajuste_ok"] and r["popt"] is not None:
        tf = np.linspace(t[0], t[-1], 800)
        ax.plot(tf, np.degrees(seno_amort(tf, *r["popt"])), '-', lw=2, color='#00ffcc',
                label=f"Ajuste  T={r['T_fit']:.4f}s  g={r['g_exp']:.4f}m/s²  γ={r['gamma_fit']:.4f}s⁻¹")
    ax.axhline(0, color='gray', lw=0.5, linestyle=':')
    ax.set_xlabel("t (s)"); ax.set_ylabel("θ (°)"); ax.set_title("Ángulo θ vs Tiempo")
    ax.legend(facecolor="#1a1a2e", labelcolor="#ccccff", fontsize=9)
    fig.suptitle("Péndulo — Ángulo", color="#fff", fontsize=13, fontweight="bold")
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)

def generate_fig3(t, x_data, y_data, theta, L, path):
    fig, ax = plt.subplots(figsize=(7, 7)); fig.patch.set_facecolor("#0d0d0d"); _style_ax(ax)
    sc = ax.scatter(x_data, y_data, c=t, cmap='plasma', s=12, alpha=0.85, zorder=3)
    plt.colorbar(sc, ax=ax, label='t (s)')
    ax.plot(0, 0, 'y*', ms=16, label='Pivote (origen)', zorder=4)
    ang = np.linspace(np.min(theta), np.max(theta), 300)
    ax.plot(L * np.sin(ang), -L * np.cos(ang), '--', color='#555577', lw=1.2, label='Arco teórico')
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title("Trayectoria del Bob")
    ax.legend(facecolor="#1a1a2e", labelcolor="#ccccff", fontsize=9)
    ax.set_aspect('equal')
    fig.suptitle("Péndulo — Trayectoria", color="#fff", fontsize=13, fontweight="bold")
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)

def generate_fig4(t, pivot_tray, escala, path):
    if len(pivot_tray) < 2: return
    t_p = np.linspace(t[0], t[-1], len(pivot_tray))
    pt  = np.array(pivot_tray, dtype=float)
    dx  = (pt[:,0] - pt[0,0]) * escala * 1000
    dy  = (pt[:,1] - pt[0,1]) * escala * 1000
    dt  = np.sqrt(dx**2 + dy**2)

    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axs: _style_ax(ax)

    axs[0].plot(t_p, dx, '-', color='#00BFFF', lw=1.5); axs[0].axhline(0, color='gray', lw=0.5, linestyle=':')
    axs[0].set_ylabel("Deriva X (mm)")
    axs[1].plot(t_p, dy, '-', color='#FF6347', lw=1.5); axs[1].axhline(0, color='gray', lw=0.5, linestyle=':')
    axs[1].set_ylabel("Deriva Y (mm)")
    axs[2].plot(t_p, dt, '-', color='#FFD700', lw=1.5); axs[2].axhline(0, color='gray', lw=0.5, linestyle=':')
    axs[2].set_ylabel("Deriva total (mm)"); axs[2].set_xlabel("t (s)")

    fig.suptitle(f"Péndulo — Deriva del Pivote  (max={np.max(dt):.2f} mm)", color="#fff", fontsize=13, fontweight="bold")
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)

def save_evidence_image(frame_bgr, result, exp_name, out_path):
    TARGET_H = 900
    fh, fw = frame_bgr.shape[:2]
    sc = TARGET_H / fh
    frame_rs = cv2.resize(frame_bgr, (int(fw * sc), TARGET_H))
    fw2 = frame_rs.shape[1]
    frame_pil = Image.fromarray(cv2.cvtColor(frame_rs, cv2.COLOR_BGR2RGB))

    PANEL_W = 440; BG = (13,13,35); ACC = (0,255,180)
    YEL = (255,220,0); RED = (255,80,100); SUB = (130,130,160); WHT = (210,210,255)

    panel = Image.new("RGB", (PANEL_W, TARGET_H), BG)
    draw  = ImageDraw.Draw(panel)
    try:
        from PIL import ImageFont
        try:
            fb=ImageFont.truetype("cour.ttf",46); fm=ImageFont.truetype("cour.ttf",24)
            fs=ImageFont.truetype("cour.ttf",19); ft=ImageFont.truetype("cour.ttf",15)
            ftt=ImageFont.truetype("cour.ttf",28)
        except: fb=fm=fs=ft=ftt=ImageFont.load_default()
    except: fb=fm=fs=ft=ftt=None

    r = result
    g_col = ACC if abs(r.get("g_exp",0) - 9.81) / 9.81 * 100 < 5 else RED

    def t_(text, y, col, fnt, x=16): draw.text((x, y), str(text), fill=col, font=fnt)
    def sep(y): draw.line([(8,y),(PANEL_W-8,y)], fill=(50,50,100), width=1)

    y = 16
    t_("PENDULO",              y, ACC, ftt); y += 38
    t_("Analisis cinematico",  y, SUB, ft);  y += 24; sep(y); y += 12
    t_("ARCHIVO",              y, SUB, ft);  y += 16
    t_(r.get("video","")[:30], y, WHT, fs);  y += 28; sep(y); y += 12
    t_("LONGITUD L (m)",       y, SUB, ft);  y += 16
    t_(f"{r.get('L',0):.4f} m",y, WHT, fm); y += 34
    t_("ESCALA (m/px)",        y, SUB, ft);  y += 16
    t_(f"{r.get('escala',0):.5f}", y, SUB, fs); y += 28; sep(y); y += 12
    t_("PERIODO T (s)",        y, SUB, ft);  y += 16
    t_(f"{r.get('T_fit',0):.5f}", y, YEL, fm); y += 34
    t_("AMPLITUD A (deg)",     y, SUB, ft);  y += 16
    t_(f"{np.degrees(r.get('A_fit',0)):.3f}", y, WHT, fm); y += 34
    t_("AMORT. gamma (1/s)",   y, SUB, ft);  y += 16
    t_(f"{r.get('gamma_fit',0):.5f}", y, WHT, fm); y += 34
    sep(y); y += 12
    t_("g  (m/s2)",            y, ACC, ft);  y += 16
    t_(f"{r.get('g_exp',0):.5f}", y, g_col, fb); y += 56
    err = abs(r.get("g_exp",0) - 9.81) / 9.81 * 100
    t_("ERROR vs 9.81",        y, SUB, ft);  y += 16
    t_(f"{err:.3f} %",         y, g_col, fm); y += 34
    sep(y); y += 12
    t_("PIVOTE px",            y, SUB, ft);  y += 16
    pv = r.get("pivot_px", (0,0))
    t_(f"({pv[0]}, {pv[1]})",  y, SUB, fs);  y += 26
    t_("PUNTOS TRACKING",      y, SUB, ft);  y += 16
    t_(str(r.get("n_puntos",0)),y, (180,120,255), fm); y += 30
    sep(y); y += 12
    t_("EXPERIMENTO",          y, SUB, ft);  y += 16
    t_(exp_name,               y, ACC, fs);  y += 26
    t_("FECHA",                y, SUB, ft);  y += 16
    ts = r.get("timestamp","")[:19].replace("T","  ")
    t_(ts,                     y, SUB, ft)

    draw.line([(0,0),(0,TARGET_H)], fill=(0,200,150), width=3)
    out = Image.new("RGB", (fw2 + PANEL_W, TARGET_H), BG)
    out.paste(frame_pil, (0,0)); out.paste(panel, (fw2,0))
    out.save(out_path, quality=95)


# ─────────────────────────────────────────────
#  APP PRINCIPAL
# ─────────────────────────────────────────────
class PenduloApp(tk.Tk):

    BG    = "#0d0d0d"; PANEL = "#121228"; CARD  = "#1a1a2e"
    ACCENT= "#00ffcc"; ACC2  = "#7b61ff"; TEXT  = "#e0e0ff"
    SUB   = "#888899"; RED   = "#ff4466"; YELLOW= "#ffdd00"
    BLUE  = "#66aaff"; PURP  = "#cc88ff"; GREEN = "#44ff88"

    STEPS = ["video","frames","bob","eje","calibracion","tracking","resultados"]

    def __init__(self):
        super().__init__()
        self.title("🔵  Analizador de Péndulo")
        self.configure(bg=self.BG)
        self.geometry("1400x900"); self.state("zoomed")

        # Video
        self.cap=None; self.video_path=""; self.total_frames=0
        self.fps=30.0; self.current_frame=0
        self.frame_inicio=None; self.frame_fin=None
        self._video_rotation=0; self._slider_updating=False
        self.playing=False; self._display_scale=1.0; self._display_offset=(0,0)

        # Calibración colores
        self.bob_lower=None; self.bob_upper=None
        self.eje_lower=None; self.eje_upper=None

        # Pivote y calibración L
        self.pivot_px=None
        self.L_cuerda=0.0
        self.escala=0.0
        self.punto_cal=None

        # Tracking
        self.datos=[]; self.trail=[]; self.pivot_trail=[]
        self._tracking_running=False
        self.result={}

        # Modo canvas
        self.canvas_mode="none"
        self.roi_start=None; self.roi_rect_canvas=None

        # Experimento
        self.base_results=os.path.join(os.path.dirname(os.path.abspath(__file__)),"resultados")
        self.exp_dir=None; self.step="video"

        self._build_ui()
        self._new_experiment()

    # ══════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════
    def _build_ui(self):
        self.columnconfigure(0,weight=0); self.columnconfigure(1,weight=1)
        self.columnconfigure(2,weight=0); self.rowconfigure(0,weight=1)
        self._build_sidebar(); self._build_canvas_area(); self._build_right_panel()

    # ── Sidebar con scroll ────────────────────
    def _build_sidebar(self):
        # Contenedor externo fijo
        sb_outer = tk.Frame(self, bg=self.PANEL, width=290)
        sb_outer.grid(row=0, column=0, sticky="ns")
        sb_outer.grid_propagate(False)
        sb_outer.columnconfigure(0, weight=1)
        sb_outer.rowconfigure(0, weight=1)

        # Canvas interno con scrollbar
        sb_canvas = tk.Canvas(sb_outer, bg=self.PANEL, highlightthickness=0, width=290)
        sb_canvas.grid(row=0, column=0, sticky="nsew")

        sb_scroll = ttk.Scrollbar(sb_outer, orient="vertical", command=sb_canvas.yview)
        sb_scroll.grid(row=0, column=1, sticky="ns")
        sb_canvas.configure(yscrollcommand=sb_scroll.set)

        # Frame interior con todo el contenido
        sb = tk.Frame(sb_canvas, bg=self.PANEL)
        sb_win = sb_canvas.create_window((0, 0), window=sb, anchor="nw")

        def _on_sb_configure(e):
            sb_canvas.configure(scrollregion=sb_canvas.bbox("all"))
        def _on_canvas_resize(e):
            sb_canvas.itemconfig(sb_win, width=e.width)

        sb.bind("<Configure>", _on_sb_configure)
        sb_canvas.bind("<Configure>", _on_canvas_resize)

        # Scroll con rueda del mouse
        def _on_mousewheel(e):
            sb_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        sb_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── Contenido ──
        sb.columnconfigure(0, weight=1)

        tk.Label(sb,text="🔵",font=("Courier New",28),bg=self.PANEL,fg=self.ACCENT).grid(row=0,column=0,pady=(16,0))
        tk.Label(sb,text="PENDULO",font=("Courier New",11,"bold"),bg=self.PANEL,fg=self.TEXT).grid(row=1,column=0)
        tk.Label(sb,text="Analizador cinemático",font=("Courier New",7),bg=self.PANEL,fg=self.SUB).grid(row=2,column=0,pady=(0,8))
        ttk.Separator(sb,orient="horizontal").grid(row=3,column=0,sticky="ew",padx=14)

        tk.Label(sb,text="EXPERIMENTO ACTIVO",font=("Courier New",6,"bold"),
                 bg=self.PANEL,fg=self.SUB).grid(row=4,column=0,sticky="w",padx=16,pady=(10,1))
        self.lbl_exp=tk.Label(sb,text="—",font=("Courier New",9,"bold"),
                               bg=self.PANEL,fg=self.ACCENT,wraplength=260,anchor="w")
        self.lbl_exp.grid(row=5,column=0,sticky="ew",padx=16,pady=(0,4))
        self._sbtn(sb,6,"＋  Nuevo Experimento",self._new_experiment,self.CARD,self.TEXT)
        ttk.Separator(sb,orient="horizontal").grid(row=7,column=0,sticky="ew",padx=14,pady=5)

        # Pasos
        tk.Label(sb,text="FLUJO DE TRABAJO",font=("Courier New",6,"bold"),
                 bg=self.PANEL,fg=self.SUB).grid(row=8,column=0,sticky="w",padx=16,pady=(2,3))
        self.step_labels={}
        step_names={"video":"1. Cargar Video","frames":"2. Marcar Frames",
                    "bob":"3. ROI Bob","eje":"4. ROI Eje/Pivote",
                    "calibracion":"5. Calibrar L","tracking":"6. Tracking",
                    "resultados":"7. Resultados"}
        for i,(k,name) in enumerate(step_names.items()):
            lbl=tk.Label(sb,text=f"  {name}",font=("Courier New",7),
                         bg=self.PANEL,fg=self.SUB,anchor="w")
            lbl.grid(row=9+i,column=0,sticky="ew",padx=14,pady=1)
            self.step_labels[k]=lbl

        ttk.Separator(sb,orient="horizontal").grid(row=16,column=0,sticky="ew",padx=14,pady=5)

        # Video
        tk.Label(sb,text="VIDEO",font=("Courier New",6,"bold"),bg=self.PANEL,fg=self.SUB).grid(row=17,column=0,sticky="w",padx=16,pady=(2,1))
        self._sbtn(sb,18,"📂  Cargar Video",self._load_video,self.CARD,self.TEXT)
        self.lbl_video=tk.Label(sb,text="Sin video",font=("Courier New",7),
                                bg=self.PANEL,fg=self.SUB,wraplength=260,anchor="w")
        self.lbl_video.grid(row=19,column=0,sticky="ew",padx=16)
        ttk.Separator(sb,orient="horizontal").grid(row=20,column=0,sticky="ew",padx=14,pady=4)

        # Nav
        tk.Label(sb,text="NAVEGACIÓN",font=("Courier New",6,"bold"),bg=self.PANEL,fg=self.SUB).grid(row=21,column=0,sticky="w",padx=16,pady=(2,2))
        nf=tk.Frame(sb,bg=self.PANEL); nf.grid(row=22,column=0,sticky="ew",padx=14)
        nf.columnconfigure((0,1,2),weight=1)
        for col,(txt,cmd,fg) in enumerate([("◀◀",self._prev_frame,self.TEXT),
                                            ("⏯",self._toggle_play,self.ACCENT),
                                            ("▶▶",self._next_frame,self.TEXT)]):
            tk.Button(nf,text=txt,command=cmd,bg=self.CARD,fg=fg,relief="flat",
                      font=("Courier New",12,"bold"),cursor="hand2",
                      activebackground=self.ACC2,activeforeground="#fff",
                      bd=0,pady=8).grid(row=0,column=col,sticky="ew",padx=2)
        ttk.Separator(sb,orient="horizontal").grid(row=23,column=0,sticky="ew",padx=14,pady=4)

        # Frames
        tk.Label(sb,text="FRAMES",font=("Courier New",6,"bold"),bg=self.PANEL,fg=self.SUB).grid(row=24,column=0,sticky="w",padx=16,pady=(2,2))
        mf=tk.Frame(sb,bg=self.PANEL); mf.grid(row=25,column=0,sticky="ew",padx=14)
        mf.columnconfigure((0,1),weight=1)
        tk.Button(mf,text="① INICIO",command=self._set_inicio,bg="#002244",fg=self.BLUE,
                  relief="flat",font=("Courier New",9,"bold"),cursor="hand2",
                  activebackground="#0044aa",activeforeground="#fff",bd=0,pady=9
                  ).grid(row=0,column=0,sticky="ew",padx=(0,2))
        tk.Button(mf,text="② FIN",command=self._set_fin,bg="#220044",fg=self.PURP,
                  relief="flat",font=("Courier New",9,"bold"),cursor="hand2",
                  activebackground="#6600aa",activeforeground="#fff",bd=0,pady=9
                  ).grid(row=0,column=1,sticky="ew",padx=(2,0))
        ttk.Separator(sb,orient="horizontal").grid(row=26,column=0,sticky="ew",padx=14,pady=4)

        # Calibración
        tk.Label(sb,text="CALIBRACIÓN",font=("Courier New",6,"bold"),bg=self.PANEL,fg=self.SUB).grid(row=27,column=0,sticky="w",padx=16,pady=(2,2))
        self._sbtn(sb,28,"🔵  ROI Bob (cuerpo)",       self._activate_roi_bob, "#002233",self.BLUE, bold=True)
        self._sbtn(sb,29,"🟡  ROI Eje / Pivote",        self._activate_roi_eje, "#332200",self.YELLOW,bold=True)
        self._sbtn(sb,30,"📏  Calibrar L (clic en bob)",self._activate_cal_L,   "#220033",self.GREEN, bold=True)
        ttk.Separator(sb,orient="horizontal").grid(row=31,column=0,sticky="ew",padx=14,pady=4)

        self._sbtn(sb,32,"▶  Ejecutar Tracking",  self._run_tracking,   "#003322",self.ACCENT,bold=True)
        self._sbtn(sb,33,"⏹  Cancelar Tracking",  self._cancel_tracking,"#220000",self.RED)
        self._sbtn(sb,34,"↺  Resetear",            self._reset_all,      "#221100",self.YELLOW)
        ttk.Separator(sb,orient="horizontal").grid(row=35,column=0,sticky="ew",padx=14,pady=4)

        self._sbtn(sb,36,"📊  Ver Gráficas",      self._show_graficas,    "#1a0033",self.ACC2,bold=True)
        # AJUSTE 1: Botón "Commit + Push Git" OCULTO — se dispara automáticamente al terminar tracking
        # self._sbtn(sb,37,"☁  Commit + Push Git",  self._do_git_commit,   "#001a11","#00dd88",bold=True)
        self._sbtn(sb,38,"🔗  Configurar Remote",  self._configure_remote,"#0a0a22","#6688ff")

        self.lbl_status=tk.Label(sb,text="Carga un video para comenzar",
                                  font=("Courier New",7),bg=self.PANEL,fg=self.SUB,
                                  wraplength=270,justify="left",anchor="w")
        self.lbl_status.grid(row=39,column=0,sticky="ew",padx=14,pady=(8,16))

    def _sbtn(self,parent,row,text,cmd,bg,fg,bold=False):
        tk.Button(parent,text=text,command=cmd,bg=bg,fg=fg,relief="flat",cursor="hand2",
                  font=("Courier New",9,"bold" if bold else ""),
                  activebackground=self.ACC2,activeforeground="#fff",
                  bd=0,pady=8).grid(row=row,column=0,sticky="ew",padx=14,pady=2)

    # ── Canvas ────────────────────────────────
    def _build_canvas_area(self):
        mid=tk.Frame(self,bg=self.BG); mid.grid(row=0,column=1,sticky="nsew")
        mid.rowconfigure(0,weight=1); mid.rowconfigure(1,weight=0); mid.columnconfigure(0,weight=1)

        self.canvas=tk.Canvas(mid,bg="#000",highlightthickness=2,
                               highlightbackground=self.ACC2,cursor="crosshair")
        self.canvas.grid(row=0,column=0,sticky="nsew",padx=(10,4),pady=10)

        sf=tk.Frame(mid,bg=self.BG); sf.grid(row=1,column=0,sticky="ew",padx=10,pady=(0,8))
        sf.columnconfigure(1,weight=1)
        tk.Label(sf,text="0",font=("Courier New",7),bg=self.BG,fg=self.SUB).grid(row=0,column=0,padx=(0,4))
        self.slider=ttk.Scale(sf,from_=0,to=100,orient="horizontal",command=self._on_slider)
        self.slider.grid(row=0,column=1,sticky="ew")
        self.lbl_slider_end=tk.Label(sf,text="0",font=("Courier New",7),bg=self.BG,fg=self.SUB)
        self.lbl_slider_end.grid(row=0,column=2,padx=(4,0))

        self.canvas.bind("<ButtonPress-1>",  self._canvas_click)
        self.canvas.bind("<B1-Motion>",       self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.bind("<Left>",  lambda e: self._prev_frame())
        self.bind("<Right>", lambda e: self._next_frame())
        self.bind("<i>",     lambda e: self._set_inicio())
        self.bind("<f>",     lambda e: self._set_fin())
        self.bind("<space>", lambda e: self._toggle_play())
        self.bind("<Escape>",lambda e: self._deactivate_modes())

    # ── Panel derecho ─────────────────────────
    def _build_right_panel(self):
        ro=tk.Frame(self,bg=self.CARD,width=350)
        ro.grid(row=0,column=2,sticky="nsew",padx=(0,10),pady=10)
        ro.grid_propagate(False); ro.columnconfigure(0,weight=1); ro.rowconfigure(0,weight=1)

        rc=tk.Canvas(ro,bg=self.CARD,highlightthickness=0)
        rc.grid(row=0,column=0,sticky="nsew")
        vsb=ttk.Scrollbar(ro,orient="vertical",command=rc.yview)
        vsb.grid(row=0,column=1,sticky="ns"); rc.configure(yscrollcommand=vsb.set)

        panel=tk.Frame(rc,bg=self.CARD)
        rc.create_window((0,0),window=panel,anchor="nw")
        def _cfg(e): rc.configure(scrollregion=rc.bbox("all")); rc.itemconfig(1,width=rc.winfo_width())
        panel.bind("<Configure>",_cfg)
        rc.bind("<Configure>",lambda e: rc.itemconfig(1,width=e.width))

        BG=self.CARD; PAD=dict(fill="x",padx=12)
        def title(text,color=None):
            tk.Label(panel,text=text,font=("Courier New",8,"bold"),
                     bg=BG,fg=color or self.SUB,anchor="w").pack(**PAD,pady=(12,1))
        def val(attr,color,fs=14):
            lbl=tk.Label(panel,text="—",font=("Courier New",fs,"bold"),bg=BG,fg=color,anchor="w")
            lbl.pack(**PAD,pady=(0,2)); setattr(self,attr,lbl)
        def sep():
            ttk.Separator(panel,orient="horizontal").pack(fill="x",padx=8,pady=5)

        tk.Label(panel,text="🔵  DATOS EN VIVO",font=("Courier New",11,"bold"),
                 bg=BG,fg=self.ACCENT,anchor="w").pack(**PAD,pady=(14,4))
        sep()

        title("ARCHIVO")
        self.rp_video=tk.Label(panel,text="—",font=("Courier New",9),bg=BG,fg=self.TEXT,
                                anchor="w",wraplength=320)
        self.rp_video.pack(**PAD,pady=(0,4)); sep()

        title("FRAME ACTUAL"); val("rp_frame",self.BLUE,14)
        title("INICIO / FIN",self.BLUE)
        self.rp_fi_ff=tk.Label(panel,text="— / —",font=("Courier New",13,"bold"),bg=BG,fg=self.BLUE,anchor="w")
        self.rp_fi_ff.pack(**PAD,pady=(0,4)); sep()

        title("MODO CANVAS")
        self.rp_mode=tk.Label(panel,text="Normal",font=("Courier New",10,"bold"),bg=BG,fg=self.SUB,anchor="w")
        self.rp_mode.pack(**PAD,pady=(0,4)); sep()

        title("PIVOTE (px)"); val("rp_pivot",self.YELLOW,13)
        title("L CUERDA (m)")
        self.entry_L=tk.Entry(panel,bg="#0f0f22",fg=self.ACCENT,
                               font=("Courier New",13,"bold"),relief="flat",
                               insertbackground=self.ACCENT,justify="center")
        self.entry_L.insert(0,"0.700"); self.entry_L.pack(**PAD,pady=(2,4),ipady=5)
        title("ESCALA (m/px)"); val("rp_escala",self.SUB,12)
        sep()

        title("PERIODO T (s)"); val("rp_T",self.YELLOW,16)
        title("AMPLITUD A (°)"); val("rp_A",self.TEXT,14)
        title("AMORT. γ (1/s)"); val("rp_gamma",self.TEXT,13)
        sep()

        title("g  (m/s²)", self.ACCENT)
        self.rp_g=tk.Label(panel,text="—",font=("Courier New",30,"bold"),bg=BG,fg=self.ACCENT,anchor="w")
        self.rp_g.pack(**PAD,pady=(0,2))
        title("ERROR vs 9.81"); val("rp_err",self.SUB,14)
        sep()

        title("PIVOTE DETECTADO"); val("rp_piv_status",self.GREEN,12)
        title("PUNTOS TRACKING"); val("rp_pts",self.PURP,13)
        sep()

        title("EXPERIMENTOS")
        tbl_f=tk.Frame(panel,bg=BG); tbl_f.pack(fill="x",padx=8,pady=(0,12))
        tbl_f.columnconfigure(0,weight=1)
        cols=("Exp","L(m)","T(s)","g(m/s²)","Err%")
        self.table=ttk.Treeview(tbl_f,columns=cols,show="headings",height=4)
        style=ttk.Style(); style.theme_use("clam")
        style.configure("Treeview",background="#0f0f22",foreground=self.TEXT,
                        fieldbackground="#0f0f22",font=("Courier New",7),rowheight=22)
        style.configure("Treeview.Heading",background=self.PANEL,foreground=self.ACCENT,
                        font=("Courier New",7,"bold"))
        style.map("Treeview",background=[("selected",self.ACC2)])
        cw={"Exp":65,"L(m)":50,"T(s)":55,"g(m/s²)":70,"Err%":50}
        for c in cols:
            self.table.heading(c,text=c); self.table.column(c,width=cw.get(c,60),anchor="center",stretch=True)
        sb2=ttk.Scrollbar(tbl_f,orient="vertical",command=self.table.yview)
        self.table.configure(yscrollcommand=sb2.set)
        sb2.grid(row=0,column=1,sticky="ns"); self.table.grid(row=0,column=0,sticky="ew")

    # ══════════════════════════════════════════
    #  EXPERIMENTO
    # ══════════════════════════════════════════
    def _new_experiment(self):
        self.exp_dir=next_experiment_folder(self.base_results)
        os.makedirs(self.exp_dir,exist_ok=True)
        self.lbl_exp.config(text=os.path.basename(self.exp_dir))
        self._status(f"Experimento: {os.path.basename(self.exp_dir)}")
        self._set_step("video")

    # ══════════════════════════════════════════
    #  VIDEO
    # ══════════════════════════════════════════
    def _load_video(self):
        path=filedialog.askopenfilename(
            filetypes=[("Videos","*.mp4 *.avi *.mov *.mkv *.MOV *.MP4"),("Todos","*.*")])
        if not path: return
        if self.cap: self.cap.release()
        self.cap=cv2.VideoCapture(path)
        if not self.cap.isOpened(): messagebox.showerror("Error","No se pudo abrir"); return
        self.video_path=path
        self.total_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps=self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_frame=0; self.frame_inicio=None; self.frame_fin=None
        rot=int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
        self._video_rotation=rot
        self.slider.config(to=max(1,self.total_frames-1))
        self.lbl_slider_end.config(text=str(self.total_frames-1))
        self.lbl_video.config(text=os.path.basename(path))
        self.rp_video.config(text=os.path.basename(path))
        self._show_frame(); self._set_step("frames")
        self._status(f"Video · {self.total_frames} frames · {self.fps:.1f} fps")

    def _get_frame(self,n):
        if not self.cap: return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,n)
        ret,frame=self.cap.read()
        if not ret: return None
        rot=self._video_rotation
        if rot==90:  frame=cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
        elif rot==180: frame=cv2.rotate(frame,cv2.ROTATE_180)
        elif rot==270: frame=cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _show_frame(self,frame_bgr=None):
        if not self.cap: return
        if frame_bgr is None: frame_bgr=self._get_frame(self.current_frame)
        if frame_bgr is None: return
        self.update_idletasks()
        cw=max(self.canvas.winfo_width(),400); ch=max(self.canvas.winfo_height(),500)
        fh,fw=frame_bgr.shape[:2]
        scale=min(cw/fw,ch/fh)
        nw,nh=max(1,int(fw*scale)),max(1,int(fh*scale))
        self._display_scale=scale; self._display_offset=((cw-nw)//2,(ch-nh)//2)
        disp=cv2.resize(frame_bgr,(nw,nh))
        self._draw_overlay(disp)
        img=Image.fromarray(cv2.cvtColor(disp,cv2.COLOR_BGR2RGB))
        self._tk_img=ImageTk.PhotoImage(img)
        ox,oy=self._display_offset
        self.canvas.delete("all")
        self.canvas.create_image(ox,oy,anchor="nw",image=self._tk_img)

        if self.roi_rect_canvas:
            x1,y1,x2,y2=self.roi_rect_canvas
            self.canvas.create_rectangle(x1,y1,x2,y2,outline=self.BLUE,width=2,dash=(4,3))

        if self.pivot_px:
            px=int(self.pivot_px[0]*scale)+ox; py=int(self.pivot_px[1]*scale)+oy
            self.canvas.create_oval(px-8,py-8,px+8,py+8,outline=self.YELLOW,width=2)
            self.canvas.create_line(px-12,py,px+12,py,fill=self.YELLOW,width=1)
            self.canvas.create_line(px,py-12,px,py+12,fill=self.YELLOW,width=1)
            self.canvas.create_text(px+14,py,text="PIVOTE",fill=self.YELLOW,
                                     font=("Courier New",8,"bold"),anchor="w")

        if self.punto_cal:
            bx=int(self.punto_cal[0]*scale)+ox; by=int(self.punto_cal[1]*scale)+oy
            self.canvas.create_oval(bx-6,by-6,bx+6,by+6,fill=self.GREEN,outline="white")
            if self.pivot_px:
                px=int(self.pivot_px[0]*scale)+ox; py=int(self.pivot_px[1]*scale)+oy
                self.canvas.create_line(px,py,bx,by,fill=self.GREEN,width=1,dash=(3,3))

        self.rp_frame.config(text=str(self.current_frame))
        fi=self.frame_inicio; ff=self.frame_fin
        self.rp_fi_ff.config(text=f"{fi if fi is not None else '—'} / {ff if ff is not None else '—'}")
        self._slider_updating=True
        try: self.slider.set(self.current_frame)
        except: pass
        self._slider_updating=False

    def _draw_overlay(self,frame):
        h,w=frame.shape[:2]
        if self.frame_inicio==self.current_frame:
            cv2.line(frame,(0,h//3),(w,h//3),(80,160,255),2)
            cv2.putText(frame,f"INICIO f{self.frame_inicio}",(8,h//3-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(80,160,255),2)
        if self.frame_fin==self.current_frame:
            cv2.line(frame,(0,2*h//3),(w,2*h//3),(200,100,255),2)
            cv2.putText(frame,f"FIN f{self.frame_fin}",(8,2*h//3-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,100,255),2)

    # ══════════════════════════════════════════
    #  NAVEGACIÓN
    # ══════════════════════════════════════════
    def _prev_frame(self):
        if self.cap: self.current_frame=max(0,self.current_frame-1); self._show_frame()
    def _next_frame(self):
        if self.cap: self.current_frame=min(self.total_frames-1,self.current_frame+1); self._show_frame()
    def _on_slider(self,val):
        if self.cap and not self._slider_updating:
            self.current_frame=int(float(val)); self._show_frame()
    def _toggle_play(self):
        if not self.cap: return
        self.playing=not self.playing
        if self.playing: self._play_loop()
    def _play_loop(self):
        if not self.playing: return
        self._next_frame()
        if self.current_frame>=self.total_frames-1: self.playing=False; return
        self.after(max(1,int(1000/self.fps)),self._play_loop)
    def _set_inicio(self):
        if self.cap: self.frame_inicio=self.current_frame; self._status(f"✅ Inicio → {self.frame_inicio}"); self._show_frame()
    def _set_fin(self):
        if self.cap: self.frame_fin=self.current_frame; self._status(f"✅ Fin → {self.frame_fin}"); self._show_frame()

    # ══════════════════════════════════════════
    #  MODOS CANVAS
    # ══════════════════════════════════════════
    def _activate_roi_bob(self):
        if not self.cap: self._status("⚠ Carga un video primero"); return
        self.canvas_mode="roi_bob"; self.roi_start=None; self.roi_rect_canvas=None
        self.rp_mode.config(text="ROI Bob — arrastra",fg=self.BLUE)
        self._status("🔵 Arrastra sobre el cuerpo (bob) del péndulo")

    def _activate_roi_eje(self):
        if not self.cap: self._status("⚠ Carga un video primero"); return
        self.canvas_mode="roi_eje"; self.roi_start=None; self.roi_rect_canvas=None
        self.rp_mode.config(text="ROI Eje — arrastra",fg=self.YELLOW)
        self._status("🟡 Arrastra sobre el eje / pivote del péndulo")

    def _activate_cal_L(self):
        if not self.cap: self._status("⚠ Carga un video primero"); return
        if self.pivot_px is None: self._status("⚠ Primero calibra el eje/pivote"); return
        self.canvas_mode="cal_L"
        self.rp_mode.config(text="Cal L — clic sobre el bob",fg=self.GREEN)
        self._status("📏 Haz clic en el centro del bob para medir L en píxeles")

    def _deactivate_modes(self):
        self.canvas_mode="none"; self.rp_mode.config(text="Normal",fg=self.SUB)

    def _canvas_to_video(self,cx,cy):
        ox,oy=self._display_offset; sc=self._display_scale
        return (cx-ox)/sc, (cy-oy)/sc

    def _canvas_click(self,event):
        if self.canvas_mode=="cal_L":
            vx,vy=self._canvas_to_video(event.x,event.y)
            self.punto_cal=(vx,vy)
            try:
                L_real=float(self.entry_L.get())
            except ValueError:
                self._status("⚠ Ingresa la longitud L antes"); return
            L_px=np.linalg.norm(np.array(self.pivot_px)-np.array(self.punto_cal))
            if L_px < 1: self._status("⚠ Punto demasiado cerca del pivote"); return
            self.L_cuerda=L_real
            self.escala=L_real/L_px
            self.rp_escala.config(text=f"{self.escala:.5f} m/px")
            self._deactivate_modes()
            self._show_frame()
            self._status(f"✅ L calibrada: {L_px:.1f}px → {L_real}m  escala={self.escala*1000:.3f}mm/px")
            self._set_step("tracking")

        elif self.canvas_mode in ("roi_bob","roi_eje"):
            self.roi_start=(event.x,event.y)

    def _canvas_drag(self,event):
        if self.canvas_mode in ("roi_bob","roi_eje") and self.roi_start:
            x0,y0=self.roi_start
            self.roi_rect_canvas=(min(x0,event.x),min(y0,event.y),
                                   max(x0,event.x),max(y0,event.y))
            self._show_frame()

    def _canvas_release(self,event):
        if self.canvas_mode not in ("roi_bob","roi_eje"): return
        if not self.roi_start: return
        x0,y0=self.roi_start; x1,y1=event.x,event.y
        vx0,vy0=self._canvas_to_video(min(x0,x1),min(y0,y1))
        vx1,vy1=self._canvas_to_video(max(x0,x1),max(y0,y1))
        rw,rh=int(vx1-vx0),int(vy1-vy0)
        if rw>5 and rh>5:
            mode=self.canvas_mode
            frame=self._get_frame(self.frame_inicio if self.frame_inicio is not None else 0)
            if frame is not None:
                crop=frame[int(vy0):int(vy0+rh),int(vx0):int(vx0+rw)]
                if crop.size>0:
                    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
                    hm=np.mean(hsv[:,:,0]); sm=np.mean(hsv[:,:,1]); vm=np.mean(hsv[:,:,2])
                    mh=15; ms=max(40,sm*0.4); mv=max(40,vm*0.4)
                    lo=np.array([max(0,hm-mh),max(0,sm-ms),max(0,vm-mv)])
                    hi=np.array([min(179,hm+mh),min(255,sm+ms),min(255,vm+mv)])
                    if mode=="roi_bob":
                        self.bob_lower=lo; self.bob_upper=hi
                        self._status(f"✅ Bob calibrado — H≈{hm:.0f}")
                        self.rp_mode.config(text="Bob OK",fg=self.BLUE)
                        self._set_step("eje")
                    else:
                        self.eje_lower=lo; self.eje_upper=hi
                        self.after(80, self._detect_pivot)
                        self._set_step("calibracion")
        self.roi_rect_canvas=None; self.roi_start=None
        self.canvas_mode="none"

    # ══════════════════════════════════════════
    #  DETECCIÓN AUTOMÁTICA DEL PIVOTE
    # ══════════════════════════════════════════
    def _detect_pivot(self):
        frame=self._get_frame(self.frame_inicio if self.frame_inicio is not None else 0)
        if frame is None or self.eje_lower is None: return

        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,self.eje_lower,self.eje_upper)
        k=np.ones((5,5),np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
        cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if cnts:
            c=max(cnts,key=cv2.contourArea)
            M=cv2.moments(c)
            if M["m00"]>0:
                px=int(M["m10"]/M["m00"]); py=int(M["m01"]/M["m00"])
                self.pivot_px=(px,py)
                self.rp_pivot.config(text=f"({px}, {py})")
                self.rp_piv_status.config(text="✅ Detectado",fg=self.GREEN)
                self._show_frame()
                self._status(f"✅ Pivote detectado en ({px},{py}) — ahora calibra L")
                return

        self._status("⚠ Pivote no detectado — ajusta el ROI del eje")
        self.rp_piv_status.config(text="⚠ No detectado",fg=self.RED)

    # ══════════════════════════════════════════
    #  TRACKING
    # ══════════════════════════════════════════
    def _run_tracking(self):
        errors=[]
        if self.frame_inicio is None or self.frame_fin is None: errors.append("marca frames de inicio y fin")
        if self.bob_lower is None: errors.append("calibra el color del bob")
        if self.eje_lower is None: errors.append("calibra el color del eje/pivote")
        if self.pivot_px is None: errors.append("detección del pivote pendiente")
        if self.escala == 0: errors.append("calibra la longitud L (clic en bob)")
        if errors: messagebox.showwarning("Faltan pasos","Por favor:\n• "+"\n• ".join(errors)); return
        self._tracking_running=True; self.datos=[]; self.trail=[]; self.pivot_trail=[]
        self.rp_mode.config(text="TRACKING ▶",fg=self.ACCENT)
        self._status("⚙ Tracking en curso...")
        threading.Thread(target=self._tracking_worker,daemon=True).start()

    def _cancel_tracking(self):
        if self._tracking_running:
            self._tracking_running=False
            self.rp_mode.config(text="Cancelado",fg=self.RED)
            self._status("⏹ Tracking cancelado")

    def _tracking_worker(self):
        fi=self.frame_inicio; ff=self.frame_fin; fps=self.fps
        total=ff-fi+1; escala=self.escala
        pivot_actual=list(self.pivot_px)
        k=np.ones((5,5),np.uint8)
        datos=[]; trail=[]; pivot_trail=[]

        for fn in range(fi,ff+1):
            if not self._tracking_running: break
            frame=self._get_frame(fn)
            if frame is None: break

            hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

            mask_eje=cv2.inRange(hsv,self.eje_lower,self.eje_upper)
            mask_eje=cv2.morphologyEx(mask_eje,cv2.MORPH_OPEN,k)
            mask_eje=cv2.morphologyEx(mask_eje,cv2.MORPH_DILATE,k)
            cnts_e,_=cv2.findContours(mask_eje,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            pivot_ok=False
            if cnts_e:
                ce=max(cnts_e,key=cv2.contourArea)
                Me=cv2.moments(ce)
                if Me["m00"]>0:
                    pivot_actual=[int(Me["m10"]/Me["m00"]),int(Me["m01"]/Me["m00"])]
                    pivot_ok=True
            pivot_trail.append(tuple(pivot_actual))

            mask_bob=cv2.inRange(hsv,self.bob_lower,self.bob_upper)
            mask_bob=cv2.morphologyEx(mask_bob,cv2.MORPH_OPEN,k)
            mask_bob=cv2.morphologyEx(mask_bob,cv2.MORPH_DILATE,k)
            cnts_b,_=cv2.findContours(mask_bob,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            cx_px=cy_px=None
            if cnts_b:
                cb=max(cnts_b,key=cv2.contourArea)
                Mb=cv2.moments(cb)
                if Mb["m00"]>0:
                    cx_px=int(Mb["m10"]/Mb["m00"]); cy_px=int(Mb["m01"]/Mb["m00"])
                    dx_px=cx_px-pivot_actual[0]
                    dy_px=pivot_actual[1]-cy_px
                    x_m=dx_px*escala; y_m=dy_px*escala
                    theta=np.arctan2(dx_px,dy_px)
                    t_=(fn-fi)/fps
                    datos.append((t_,x_m,y_m,theta))
                    trail.append((cx_px,cy_px))

            prog=int((fn-fi+1)/total*100)
            frame_disp=frame.copy()
            self._draw_tracking_frame(frame_disp,cx_px,cy_px,trail,
                                       pivot_actual,pivot_ok,fn,fi,ff,
                                       datos[-1] if datos else None)
            self.after(0,lambda f=frame_disp,p=prog: self._update_canvas(f,p))
            import time; time.sleep(0.001)

        self.datos=datos; self.trail=trail; self.pivot_trail=pivot_trail
        self._tracking_running=False
        self.after(0,self._tracking_done)

    def _draw_tracking_frame(self,frame,cx,cy,trail,pivot,pivot_ok,fn,fi,ff,last_dato):
        h,w=frame.shape[:2]; total=ff-fi+1; prog=(fn-fi+1)/total
        pv_col=(0,255,200) if pivot_ok else (0,200,255)

        ov=frame.copy()
        cv2.rectangle(ov,(0,0),(w,44),(8,8,28),-1)
        cv2.addWeighted(ov,0.72,frame,0.28,0,frame)
        cv2.rectangle(frame,(0,40),(int(w*prog),44),(0,200,150),-1)
        cv2.rectangle(frame,(0,40),(w,44),(40,40,80),1)

        piv_str="PIVOTE OK" if pivot_ok else "PIVOTE FALLBACK"
        cv2.putText(frame,f"TRACKING  f{fn}/{ff}  {int(prog*100)}%  {piv_str}",
                    (8,28),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,255,180),1,cv2.LINE_AA)

        if last_dato:
            _,xm,ym,theta=last_dato
            theta_deg=np.mod(np.degrees(theta)+720,360)-180
            cv2.putText(frame,f"θ={theta_deg:.1f}°  x={xm:.3f}m  y={ym:.3f}m",
                        (w-310,28),cv2.FONT_HERSHEY_SIMPLEX,0.46,(255,220,0),1,cv2.LINE_AA)

        cv2.circle(frame,(pivot[0],pivot[1]),8,pv_col,-1)
        cv2.circle(frame,(pivot[0],pivot[1]),10,(255,255,255),1)

        n=len(trail)
        for i in range(1,n):
            alpha=i/n; rc=int(255*alpha); gc=int(200*(1-alpha*0.5))
            cv2.line(frame,(int(trail[i-1][0]),int(trail[i-1][1])),
                           (int(trail[i][0]),int(trail[i][1])),(0,gc,rc),2,cv2.LINE_AA)

        if cx is not None:
            cv2.circle(frame,(cx,cy),11,(0,255,180),-1)
            cv2.circle(frame,(cx,cy),14,(255,255,255),1)
            cv2.line(frame,(cx-20,cy),(cx+20,cy),(255,255,255),1)
            cv2.line(frame,(cx,cy-20),(cx,cy+20),(255,255,255),1)
            cv2.line(frame,(pivot[0],pivot[1]),(cx,cy),(255,220,0),2)
            if last_dato:
                theta_deg=np.mod(np.degrees(last_dato[3])+720,360)-180
                cv2.putText(frame,f"θ={theta_deg:.1f}°",(cx+16,cy-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,180),1,cv2.LINE_AA)

    def _update_canvas(self,frame,prog):
        self._show_frame(frame_bgr=frame)
        self._status(f"⚙ Tracking {prog}%  —  {len(self.datos)} puntos")

    def _tracking_done(self):
        self._tracking_running=False; self.rp_mode.config(text="Normal",fg=self.SUB)
        if len(self.datos)<5:
            self._status("⚠ Datos insuficientes — revisa colores ROI"); return

        d=np.array(self.datos)
        t=d[:,0]; x_data=d[:,1]; y_data=d[:,2]; theta=d[:,3]

        r=calcular_resultados(t,x_data,y_data,theta,self.L_cuerda)
        err=abs(r["g_exp"]-9.81)/9.81*100 if r["ajuste_ok"] else 0

        self.result={
            "video":      os.path.basename(self.video_path),
            "L":          self.L_cuerda,
            "escala":     self.escala,
            "pivot_px":   self.pivot_px,
            "T_fit":      r["T_fit"],
            "g_exp":      r["g_exp"],
            "A_fit":      r["A_fit"],
            "gamma_fit":  r["gamma_fit"],
            "omega_fit":  r["omega_fit"],
            "ajuste_ok":  r["ajuste_ok"],
            "error_pct":  err,
            "n_puntos":   len(self.datos),
            "t":          t.tolist(),
            "x_data":     x_data.tolist(),
            "y_data":     y_data.tolist(),
            "theta":      theta.tolist(),
            "timestamp":  datetime.datetime.now().isoformat(),
            "experimento":os.path.basename(self.exp_dir),
        }
        self._r_cached = r

        g_col=self.ACCENT if err<5 else self.RED
        self.rp_T.config(text=f"{r['T_fit']:.5f} s")
        self.rp_A.config(text=f"{np.degrees(r['A_fit']):.3f}°")
        self.rp_gamma.config(text=f"{r['gamma_fit']:.5f} s⁻¹")
        self.rp_g.config(text=f"{r['g_exp']:.4f}",fg=g_col)
        self.rp_err.config(text=f"{err:.3f} %",fg=g_col)
        self.rp_pts.config(text=str(len(self.datos)))

        self._save_results(t,x_data,y_data,theta,r)
        self._set_step("resultados")
        self._status(f"✅ T={r['T_fit']:.4f}s  g={r['g_exp']:.4f}m/s²  Error={err:.2f}%")

        frame_final=self._get_frame(self.frame_fin)
        if frame_final is not None and self.trail:
            self._draw_final_trail(frame_final,r)
            self._show_frame(frame_bgr=frame_final)

        self.table.insert("","end",values=(
            os.path.basename(self.exp_dir),
            f"{self.L_cuerda:.3f}",f"{r['T_fit']:.4f}",
            f"{r['g_exp']:.4f}",f"{err:.2f}%"
        ))

        # AJUSTE 1: Commit + Push automático al terminar el tracking
        self._do_git_commit_auto()

    def _draw_final_trail(self,frame,r):
        h,w=frame.shape[:2]; n=len(self.trail)
        for i in range(1,n):
            alpha=i/n; rc=int(255*alpha); gc=int(200*(1-alpha*0.5))
            cv2.line(frame,(int(self.trail[i-1][0]),int(self.trail[i-1][1])),
                           (int(self.trail[i][0]),int(self.trail[i][1])),(0,gc,rc),2,cv2.LINE_AA)
        if self.trail:
            cv2.circle(frame,(int(self.trail[0][0]),int(self.trail[0][1])),9,(80,160,255),-1)
            cv2.circle(frame,(int(self.trail[-1][0]),int(self.trail[-1][1])),9,(200,100,255),-1)
        if self.pivot_px:
            cv2.circle(frame,self.pivot_px,10,(0,220,255),-1)
        ov=frame.copy()
        cv2.rectangle(ov,(0,h-46),(w,h),(8,8,28),-1)
        cv2.addWeighted(ov,0.8,frame,0.2,0,frame)
        g_col=(0,255,180) if r["g_exp"] and abs(r["g_exp"]-9.81)/9.81*100<5 else (80,100,255)
        cv2.putText(frame,
                    f"COMPLETO  T={r['T_fit']:.4f}s  g={r['g_exp']:.4f}m/s2  L={self.L_cuerda}m",
                    (10,h-14),cv2.FONT_HERSHEY_SIMPLEX,0.52,g_col,1,cv2.LINE_AA)

    def _save_results(self,t,x_data,y_data,theta,r):
        res=self.result
        with open(os.path.join(self.exp_dir,"resultado.json"),"w") as f:
            json.dump({k:v for k,v in res.items() if not isinstance(v,np.ndarray)},f,indent=2,default=str)
        np.savetxt(os.path.join(self.exp_dir,"datos.csv"),
                   np.column_stack((t,x_data,y_data,theta,r["vx"],r["vy"])),
                   header="t(s),x(m),y(m),theta(rad),vx(m/s),vy(m/s)",
                   delimiter=",",fmt="%.6f",comments="")
        generate_fig1(t,x_data,y_data,r, os.path.join(self.exp_dir,"fig1_posicion_velocidad.png"))
        generate_fig2(t,theta,r,          os.path.join(self.exp_dir,"fig2_angulo.png"))
        generate_fig3(t,x_data,y_data,theta,self.L_cuerda, os.path.join(self.exp_dir,"fig3_trayectoria.png"))
        if self.pivot_trail:
            generate_fig4(t,self.pivot_trail,self.escala, os.path.join(self.exp_dir,"fig4_deriva_pivote.png"))
        frame_ev=self._get_frame(self.frame_inicio)
        if frame_ev is not None:
            save_evidence_image(frame_ev,res,os.path.basename(self.exp_dir),
                                os.path.join(self.exp_dir,"evidencia.png"))

    # ══════════════════════════════════════════
    #  GRÁFICAS
    # ══════════════════════════════════════════
    def _show_graficas(self):
        if not self.result: messagebox.showinfo("Sin datos","Ejecuta el tracking primero"); return
        win=tk.Toplevel(self); win.title("Gráficas — Péndulo")
        win.configure(bg=self.BG); win.geometry("900x650")
        nb=ttk.Notebook(win); nb.pack(fill="both",expand=True,padx=8,pady=8)
        figs=[("Pos/Vel","fig1_posicion_velocidad.png"),("Ángulo","fig2_angulo.png"),
              ("Trayectoria","fig3_trayectoria.png"),("Deriva Pivote","fig4_deriva_pivote.png")]
        for name,fname in figs:
            path=os.path.join(self.exp_dir,fname)
            if not os.path.exists(path): continue
            frm=tk.Frame(nb,bg=self.BG); nb.add(frm,text=name)
            img=Image.open(path); img.thumbnail((880,600))
            tk_img=ImageTk.PhotoImage(img)
            lbl=tk.Label(frm,image=tk_img,bg=self.BG); lbl.image=tk_img; lbl.pack(pady=6)

    # ══════════════════════════════════════════
    #  RESET
    # ══════════════════════════════════════════
    def _reset_all(self):
        if not messagebox.askyesno("Resetear","¿Resetear toda la calibración?"): return
        self.frame_inicio=None; self.frame_fin=None
        self.bob_lower=None; self.bob_upper=None
        self.eje_lower=None; self.eje_upper=None
        self.pivot_px=None; self.escala=0.0; self.punto_cal=None
        self.datos=[]; self.trail=[]; self.pivot_trail=[]; self.result={}
        for attr in ["rp_T","rp_A","rp_gamma","rp_err","rp_pts","rp_pivot","rp_escala","rp_piv_status"]:
            getattr(self,attr).config(text="—",fg=self.SUB)
        self.rp_g.config(text="—",fg=self.ACCENT)
        self._show_frame(); self._set_step("frames"); self._status("↺ Reseteado")

    # ══════════════════════════════════════════
    #  GIT
    # ══════════════════════════════════════════
    def _do_git_commit_auto(self):
        """Commit + Push automático al terminar tracking (sin intervención del usuario)."""
        if not self.result: return
        r = self.result
        msg = (f"[{os.path.basename(self.exp_dir)}] "
               f"L={r.get('L',0):.3f}m "
               f"T={r.get('T_fit',0):.4f}s "
               f"g={r.get('g_exp',0):.4f}m/s2 "
               f"err={r.get('error_pct',0):.2f}%")
        self._status("☁ Guardando en Git automáticamente...")

        def _c():
            ok, log = git_auto_commit(self.exp_dir, msg)
            status_msg = "✅ Git: commit OK" if ok else "⚠ Git: sin remote (solo commit local)"
            self.after(0, lambda: self._status(status_msg))

        threading.Thread(target=_c, daemon=True).start()

    def _do_git_commit(self):
        """Commit + Push manual (disponible si se reactiva el botón)."""
        if not self.result: self._status("⚠ Sin resultados"); return
        r=self.result
        msg=(f"[{os.path.basename(self.exp_dir)}] "
             f"L={r.get('L',0):.3f}m "
             f"T={r.get('T_fit',0):.4f}s "
             f"g={r.get('g_exp',0):.4f}m/s2 "
             f"err={r.get('error_pct',0):.2f}%")
        self._status("☁ Haciendo commit...")
        def _c():
            ok,log=git_auto_commit(self.exp_dir,msg)
            self.after(0,lambda: self._show_git_log(ok,log))
            self.after(0,lambda: self._status("✅ OK" if ok else "❌ Falló"))
        threading.Thread(target=_c,daemon=True).start()

    def _show_git_log(self,ok,log_txt):
        win=tk.Toplevel(self); win.title("Git"); win.configure(bg=self.BG); win.geometry("700x380")
        tk.Label(win,text="Git Log",font=("Courier New",10,"bold"),
                 bg=self.BG,fg=self.ACCENT if ok else self.RED).pack(pady=(10,4))
        frm=tk.Frame(win,bg=self.BG); frm.pack(fill="both",expand=True,padx=10)
        txt=tk.Text(frm,bg="#0a0a1a",fg="#ccffcc" if ok else "#ffaaaa",
                    font=("Courier New",8),wrap="word",relief="flat")
        sb=ttk.Scrollbar(frm,command=txt.yview); txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right",fill="y"); txt.pack(fill="both",expand=True)
        txt.insert("1.0",log_txt); txt.config(state="disabled")
        tk.Button(win,text="Cerrar",command=win.destroy,bg=self.CARD,fg=self.TEXT,
                  relief="flat",font=("Courier New",9),padx=14,pady=6).pack(pady=8)

    def _configure_remote(self):
        git=_find_git(); root=os.path.dirname(os.path.abspath(__file__))
        r=subprocess.run([git,"remote","get-url","origin"],cwd=root,capture_output=True,text=True)
        current=r.stdout.strip() if r.returncode==0 else ""
        win=tk.Toplevel(self); win.title("Remote"); win.configure(bg=self.BG)
        win.geometry("540x200"); win.resizable(False,False)
        tk.Label(win,text="🔗  Remote Git",font=("Courier New",11,"bold"),bg=self.BG,fg=self.ACCENT).pack(pady=(14,4))
        entry=tk.Entry(win,bg=self.CARD,fg=self.TEXT,insertbackground=self.ACCENT,
                       font=("Courier New",10),relief="flat",width=50)
        entry.insert(0,current); entry.pack(padx=20,pady=8,ipady=6)
        res=tk.Label(win,text="",font=("Courier New",8),bg=self.BG,fg=self.SUB); res.pack()
        def _apply():
            url=entry.get().strip()
            subprocess.run([git,"remote","remove","origin"],cwd=root,capture_output=True)
            r2=subprocess.run([git,"remote","add","origin",url],cwd=root,capture_output=True,text=True)
            res.config(text="✅ Configurado" if r2.returncode==0 else f"❌ {r2.stderr.strip()}")
        bf=tk.Frame(win,bg=self.BG); bf.pack(pady=8)
        tk.Button(bf,text="Aplicar",command=_apply,bg=self.ACCENT,fg="#000",
                  font=("Courier New",9,"bold"),relief="flat",padx=16,pady=6).pack(side="left",padx=4)
        tk.Button(bf,text="Cerrar",command=win.destroy,bg=self.CARD,fg=self.TEXT,
                  font=("Courier New",9),relief="flat",padx=16,pady=6).pack(side="left",padx=4)

    # ══════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════
    def _set_step(self,step):
        self.step=step
        idx=self.STEPS.index(step) if step in self.STEPS else -1
        names={"video":"1. Cargar Video","frames":"2. Marcar Frames",
               "bob":"3. ROI Bob","eje":"4. ROI Eje/Pivote",
               "calibracion":"5. Calibrar L","tracking":"6. Tracking",
               "resultados":"7. Resultados"}
        for i,(k,name) in enumerate(names.items()):
            si=self.STEPS.index(k)
            if si<idx:    col,pfx=self.ACCENT,"✓ "
            elif si==idx:  col,pfx=self.YELLOW,"▶ "
            else:          col,pfx=self.SUB,"  "
            self.step_labels[k].config(text=f"  {pfx}{name}",fg=col)

    def _status(self,msg):
        self.lbl_status.config(text=msg); self.update_idletasks()

    def on_close(self):
        if self.cap: self.cap.release()
        self.destroy()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app=PenduloApp()
    app.protocol("WM_DELETE_WINDOW",app.on_close)
    app.mainloop()