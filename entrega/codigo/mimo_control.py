"""
Funciones usadas por el cuaderno Miniproyecto_Control_Termico_MIMO.ipynb, tanto en el análisis como
en el código de los ensayos en el equipo:

    PlantFOPDT            planta 2x2 FOPDT en tiempo discreto (retardos explícitos, sin Padé)
    PI2                   dos PI independientes con anti-windup por back-calculation
    paso_control          ley de control Q = Q_OP + D·v, saturación y anti-windup (simulación y equipo)
    simular_lazo_cerrado  escenarios A/B sobre el modelo identificado
    simular_lazo_abierto  respuesta de G(s)·D ante escalones en v (verificación del desacoplador)
    indicadores           e_ss, t_s, sobreimpulso, IAE, interacción cruzada, esfuerzo y saturación
    Vigilante             protecciones en línea: calentador que no responde y lectura físicamente imposible

Convención: Gij = efecto de la entrada Qj sobre la salida Ti (fila = salida, columna = entrada).
"""
import numpy as np
import pandas as pd

_trapz = getattr(np, "trapezoid", None) or np.trapz


class PlantFOPDT:
    """Planta FOPDT 2x2 discreta: primer orden exacto y retardo entero de round(θ/Ts) muestras."""
    def __init__(self, K, tau, theta, Ts):
        self.K, self.Ts = np.asarray(K, float), Ts
        self.a = np.exp(-Ts / np.asarray(tau, float))
        self.d = np.round(np.asarray(theta, float) / Ts).astype(int)
        self.x = np.zeros((2, 2))
        self.hist = np.zeros((2, self.d.max() + 1))      # hist[j, m] = ΔQj[k-m]

    def output(self):
        return self.x.sum(axis=1)                         # ΔT1, ΔT2

    def update(self, dQ):
        self.hist = np.roll(self.hist, 1, axis=1)
        self.hist[:, 0] = dQ
        for i in range(2):
            for j in range(2):
                u = self.hist[j, self.d[i, j]]
                self.x[i, j] = self.a[i, j] * self.x[i, j] + self.K[i, j] * (1 - self.a[i, j]) * u


class PI2:
    """Dos PI independientes: v = Kp·e + I, con anti-windup por back-calculation (Tt = Ti/2).
    El integrador arranca en cero en cada ensayo (mismo estado inicial para A y B)."""
    def __init__(self, Kp, Ti):
        self.Kp, self.Ti = np.asarray(Kp, float), np.asarray(Ti, float)
        self.Tt = 0.5 * self.Ti
        self.I = np.zeros(2)
        self.e = np.zeros(2)

    def compute(self, e):
        self.e = np.asarray(e, float)
        return self.Kp * self.e + self.I

    def antiwindup(self, v, v_app, dt):
        self.I += dt * (self.Kp / self.Ti * self.e + (v_app - v) / self.Tt)


def paso_control(ctrl, D, Dinv, Q_op, qmin, qmax, r, y, dt):
    """Una muestra de la ley de control. Devuelve v, Q solicitada y Q aplicada."""
    v = ctrl.compute(r - y)
    q_cmd = Q_op + D @ v
    q_app = np.clip(q_cmd, qmin, qmax)
    ctrl.antiwindup(v, Dinv @ (q_app - Q_op), dt)
    return v, q_cmd, q_app


def simular_lazo_cerrado(K, TAU, THETA, D, Kp, Ti, ref, T_op, Q_op, qmin, qmax,
                         t_end, t_step, Ts=1.0, noise=0.0, seed=0):
    """Lazo cerrado sobre el modelo FOPDT en desviaciones alrededor de (T_op, Q_op)."""
    rng = np.random.default_rng(seed)
    plant = PlantFOPDT(K, TAU, THETA, Ts)
    ctrl = PI2(Kp, Ti)
    D = np.asarray(D, float); Dinv = np.linalg.inv(D)
    T_op, Q_op, ref = (np.asarray(a, float) for a in (T_op, Q_op, ref))
    n = int(round(t_end / Ts)); t = np.arange(n) * Ts
    out = {c: np.zeros((n, 2)) for c in ("T", "r_abs", "v", "Q_cmd", "Q")}
    for k in range(n):
        r = T_op + ref * (t[k] >= t_step)
        y = T_op + plant.output() + noise * rng.standard_normal(2)
        v, q_cmd, q_app = paso_control(ctrl, D, Dinv, Q_op, qmin, qmax, r, y, Ts)
        plant.update(q_app - Q_op)
        out["T"][k], out["r_abs"][k], out["v"][k], out["Q_cmd"][k], out["Q"][k] = y, r, v, q_cmd, q_app
    out["t"] = t
    return out


def simular_lazo_abierto(K, TAU, THETA, D, v_step, t_end, t_step, Ts=1.0):
    """Escalón en v (sin controlador): ΔQ = D·v y ΔT = G(s)·ΔQ."""
    plant = PlantFOPDT(K, TAU, THETA, Ts)
    n = int(round(t_end / Ts)); t = np.arange(n) * Ts
    dT, dQ = np.zeros((n, 2)), np.zeros((n, 2))
    for k in range(n):
        q = np.asarray(D, float) @ (np.asarray(v_step, float) * (t[k] >= t_step))
        dT[k], dQ[k] = plant.output(), q
        plant.update(q)
    return t, dT, dQ


def indicadores(t, T, r_abs, Q_cmd, Q_app, t_step, ref, qmin, qmax,
                band=0.02, band_alt=0.05, win=100.0, smooth_s=30.0):
    """Indicadores de la fase 6, con el mismo criterio para simulación y equipo.
    T se suaviza con un promedio móvil centrado de smooth_s segundos antes de calcular t_s y el
    sobreimpulso (con escalones de 5 °C la banda de ±2 % es ±0.1 °C, menor que el ruido del sensor)."""
    t, T, r_abs, Q_cmd, Q_app = (np.asarray(a, float) for a in (t, T, r_abs, Q_cmd, Q_app))
    if smooth_s > 0:
        w = max(1, int(round(smooth_s / np.median(np.diff(t)))))
        T = pd.DataFrame(T).rolling(w, center=True, min_periods=1).mean().to_numpy()
    m = t >= t_step
    out = {}
    for i in range(2):
        if abs(ref[i]) > 0:
            rf = r_abs[-1, i]
            for b, nombre in ((band, f"ts{i+1} ±{band*100:.0f}% [s]"), (band_alt, f"ts{i+1} ±{band_alt*100:.0f}% [s]")):
                fuera = np.where((np.abs(T[:, i] - rf) > b * abs(ref[i])) & m)[0]
                if len(fuera) and fuera[-1] == len(t) - 1:
                    out[nombre] = np.nan                    # no se estableció dentro del registro
                else:
                    out[nombre] = t[fuera[-1]] - t_step if len(fuera) else 0.0
            pico = (T[m, i].max() - rf) if ref[i] > 0 else (rf - T[m, i].min())
            out[f"e_ss{i+1} [°C]"] = rf - T[t >= t[-1] - win, i].mean()
            out[f"OS{i+1} [%]"] = max(0.0, pico) / abs(ref[i]) * 100
            out[f"IAE{i+1} [°C·s]"] = _trapz(np.abs(r_abs[m, i] - T[m, i]), t[m])
        else:
            pre = (t >= t_step - 60) & (t < t_step)
            T0 = T[pre, i].mean() if pre.any() else T[m, i][0]
            out[f"I_{i+1}<-{2-i} [°C]"] = np.max(np.abs(T[m, i] - T0))
    for j in range(2):
        out[f"Q{j+1} sol. máx/mín [%]"] = f"{Q_cmd[m, j].max():.1f} / {Q_cmd[m, j].min():.1f}"
        out[f"Q{j+1} apl. máx/mín [%]"] = f"{Q_app[m, j].max():.1f} / {Q_app[m, j].min():.1f}"
        lim = (Q_app[m, j] <= qmin + 1e-9) | (Q_app[m, j] >= qmax - 1e-9)
        out[f"% muestras en límite Q{j+1}"] = lim.mean() * 100
    return out


class Vigilante:
    """Protecciones en línea, ajustadas con los datos del 23-oct-2026 (detectan las fallas de los tres
    intentos de S1-A y no dan falsas alarmas en los ensayos E1 y E2):

    1. Calentador que no responde: Qi >= Q_MIN_RESP durante WIN_RESP s, error > E_MIN_RESP (15 °C) y Ti sube menos
       de DT_MIN_RESP °C en esa ventana.
    2. Lectura físicamente imposible: la pendiente de Ti (mediana de 9 muestras, ventana de 20 s)
       cae más rápido que el enfriamiento libre (Ti - T_AMB)/TAU_ENF + 0.05 °C/s, o sube más de
       1 °C/s, durante 3 muestras seguidas.
    """
    Q_MIN_RESP, WIN_RESP, DT_MIN_RESP, E_MIN_RESP = 15.0, 120.0, 3.0, 15.0
    MED_N, WIN_RATE, TAU_ENF, T_AMB, MARGEN, SUBIDA_MAX, N_SEGUIDAS = 9, 20.0, 200.0, 25.0, 0.05, 1.0, 3

    def __init__(self):
        self.t, self.T, self.Q, self.e = [], [], [], []
        self.cuenta = 0

    def revisar(self, t, T_crudo, Q, e):
        """Agrega una muestra; devuelve un texto con la anomalía o None."""
        self.t.append(t); self.T.append(np.asarray(T_crudo, float))
        self.Q.append(np.asarray(Q, float)); self.e.append(np.asarray(e, float))
        tt = np.asarray(self.t)
        # 1. calentador que no responde
        k0 = np.searchsorted(tt, t - self.WIN_RESP)
        if t - tt[0] >= self.WIN_RESP and k0 < len(tt) - 1:
            Tm = np.asarray(self.T)
            Qv = np.asarray(self.Q)[k0:]
            for i in range(2):
                ini = np.median(Tm[max(0, k0 - 2):k0 + 3, i]); fin = np.median(Tm[-5:, i])
                if Qv[:, i].min() >= self.Q_MIN_RESP and self.e[-1][i] > self.E_MIN_RESP and fin - ini < self.DT_MIN_RESP:
                    return (f"el calentador {i+1} no responde: Q{i+1} >= {self.Q_MIN_RESP:.0f} % durante "
                            f"{self.WIN_RESP:.0f} s y T{i+1} subió solo {fin - ini:.2f} °C")
        # 2. lectura físicamente imposible
        if len(self.T) > self.MED_N:
            Tm = np.asarray(self.T[-(self.MED_N + int(self.WIN_RATE) + 5):])
            tm = tt[-len(Tm):]
            med = np.column_stack([pd.Series(Tm[:, i]).rolling(self.MED_N, center=False).median()
                                   for i in range(2)])
            k1 = len(tm) - 1; k0 = np.searchsorted(tm, tm[-1] - self.WIN_RATE)
            if k0 >= self.MED_N - 1 and k1 > k0:
                s = (med[k1] - med[k0]) / (tm[k1] - tm[k0])
                lim = np.maximum(med[k1] - self.T_AMB, 0) / self.TAU_ENF + self.MARGEN
                mala = (s < -lim) | (s > self.SUBIDA_MAX)
                self.cuenta = self.cuenta + 1 if mala.any() else 0
                if self.cuenta >= self.N_SEGUIDAS:
                    i = int(np.argmax(mala))
                    return (f"lectura físicamente imposible en T{i+1}: pendiente {s[i]:+.2f} °C/s "
                            f"(límite físico −{lim[i]:.2f} / +{self.SUBIDA_MAX:.1f} °C/s)")
        return None
