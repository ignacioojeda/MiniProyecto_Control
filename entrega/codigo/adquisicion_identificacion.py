# Copia textual del código de adquisición usado para los ensayos de identificación E1 y E2 (23-oct-2026).
# Origen: celda del cuaderno 02_Accessing_TempLAB_UdeA.ipynb. La sección 2.0 del cuaderno fija HEATER, FILENAME y T_MAX antes de llamar a main().
"""
Adquisición de un ensayo MIMO por escalones en TCLab (ambos sensores medidos).

PERFIL (HEATER = 1, ensayo 1):
    0  - 1 min   : línea base         Q1 = 0 %,  Q2 = 0 %
    1  - 11 min  : escalón 1          Q1 = 15 %, Q2 = 0 %
    11 - 21 min  : escalón 2          Q1 = 50 %, Q2 = 0 %
    21 - 31 min  : escalón cruzado    Q1 = 50 %, Q2 = 35 %
Con HEATER = 2 (ensayo 2) se intercambian los papeles: Q2 hace 15 -> 50 % y Q1 da el escalón de 35 %.

QUÉ SE IDENTIFICA (convención Gij = efecto de la entrada Qj sobre la salida Ti):
    HEATER = 1 -> escalones 1 y 2 dan G11 (T1) y G21 (T2); el escalón cruzado da G12 (T1) y G22 (T2)
    HEATER = 2 -> escalones 1 y 2 dan G12 (T1) y G22 (T2); el escalón cruzado da G11 (T1) y G21 (T2)
Así, con los dos ensayos cada canal se identifica dos veces (una como canal principal y otra por
superposición), y se pueden comparar.

USO
1. Ajusta la sección CONFIGURACIÓN (HEATER, FILENAME, límites autorizados).
2. Ejecuta: `python tclab_escalones_mimo.py`. Se abre una ventana con la gráfica en tiempo real.
3. Para el ensayo 2 cambia HEATER = 2 y FILENAME, deja enfriar el equipo y vuelve a ejecutar.

DEPENDENCIAS: pip install tclab numpy pandas matplotlib scipy   (scipy solo para el ajuste final)

SEGURIDAD: los límites (potencia, temperatura, duración) deben ser los AUTORIZADOS por el
laboratorio. Si alguna temperatura supera T_MAX, se apagan ambos calentadores y el ensayo termina.
Antes del segundo ensayo, deja que el equipo se enfríe hasta una temperatura inicial comparable.
"""
import csv
import datetime
import os
import time

import matplotlib
try:
    import tkinter  # si está disponible, usa ventana emergente (TkAgg)
    matplotlib.use("TkAgg")
except Exception:
    pass  # si no, matplotlib elige su backend por defecto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tclab

# =====================================================================================
# CONFIGURACIÓN (lo único que normalmente debes tocar)
# =====================================================================================
HEATER = 2                            # entrada principal: 1 -> Q1 ; 2 -> Q2
FILENAME = "ensayo1_Q2_principal.csv"  # cambia el nombre en cada ejecución

BASELINE_S = 60.0                     # s, línea base con ambos calentadores en 0 %
SEG_S = 600.0                         # s, duración de cada escalón (10 min)
Q_MAIN_LEVELS = [5.0, 12.0]          # % niveles sucesivos de la entrada principal
Q_CROSS = 8.0                       # % escalón de la otra entrada (manteniendo el último nivel)

TS = 1.0                              # s, periodo de muestreo
T_MAX = 100                          # °C, límite de temperatura AUTORIZADO: si se supera, apaga y termina
T_MAX_N = 5                           # muestras: se compara con la MEDIANA de las últimas N lecturas
                                      # (así un pico aislado de ruido no corta el ensayo)

STEADY_WIN_S = 120.0                  # s, ventana para medir la pendiente al final de cada tramo
STEADY_SLOPE = 0.003                  # °C/s, pendiente máxima (abs) para considerar estacionario
SMOOTH_S = 15.0                       # s, ventana del suavizado (solo para los cálculos rápidos)
PRE_AVG_S = 30.0                      # s antes de cada escalón promediados para el valor inicial
SS_AVG_S = 60.0                       # s finales de cada tramo promediados para el valor final
DESPIKE_N = 5                         # muestras del filtro de mediana que quita picos ANTES del análisis
                                      # (el CSV guarda siempre los datos crudos)
DO_LSQ_FIT = True                     # ajuste final por mínimos cuadrados con superposición (scipy)
# =====================================================================================

OTHER = 2 if HEATER == 1 else 1


def construir_perfil():
    """Lista de tramos: (t_inicio, t_fin, Q1, Q2, nombre)."""
    tramos, t0 = [], 0.0
    q = {HEATER: 0.0, OTHER: 0.0}
    tramos.append((t0, BASELINE_S, q[1], q[2], "base"))
    t0 = BASELINE_S
    for n, nivel in enumerate(Q_MAIN_LEVELS, start=1):
        q[HEATER] = nivel
        tramos.append((t0, t0 + SEG_S, q[1], q[2], f"escalon{n}_Q{HEATER}"))
        t0 += SEG_S
    q[OTHER] = Q_CROSS
    tramos.append((t0, t0 + SEG_S, q[1], q[2], f"cruzado_Q{OTHER}"))
    return tramos


TRAMOS = construir_perfil()
DURACION_S = TRAMOS[-1][1]


def potencia_en(t_rel):
    """Devuelve (Q1, Q2, nombre_del_tramo, índice) para el instante t_rel."""
    for k, (ti, tf, q1, q2, nom) in enumerate(TRAMOS):
        if ti <= t_rel < tf:
            return q1, q2, nom, k
    ti, tf, q1, q2, nom = TRAMOS[-1]
    return q1, q2, nom, len(TRAMOS) - 1


def nombre_libre(path):
    """Evita sobrescribir un archivo existente."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    nuevo = f"{base}_{datetime.datetime.now():%Y%m%d_%H%M%S}{ext}"
    print(f"[aviso] '{path}' ya existe; se guardará como '{nuevo}'")
    return nuevo


# =====================================================================================
# ANÁLISIS
# =====================================================================================
def escalones(t_cambio):
    """Escalones aplicados: (t_real, entrada j (1 o 2), dQ, t_fin_tramo, nombre)."""
    out = []
    for k in range(1, len(TRAMOS)):
        if t_cambio[k] is None:
            continue
        _, tf, q1, q2, nom = TRAMOS[k]
        _, _, p1, p2, _ = TRAMOS[k - 1]
        for j, (a, b) in ((1, (p1, q1)), (2, (p2, q2))):
            if b != a:
                out.append((t_cambio[k], j, b - a, tf, nom))
    return out


def dos_puntos(t, y, t_s, dQ, t_fin, dt):
    """K, theta, tau, t63 y notas de un escalón (método de Smith, 28.3 % y 63.2 %)."""
    r = dict(K=np.nan, theta=np.nan, tau=np.nan, t63=np.nan, y0=np.nan, dT=np.nan, sigma=np.nan,
             pend_final=np.nan, nota="")
    pre = (t >= t_s - PRE_AVG_S) & (t < t_s)
    seg = (t >= t_s) & (t < t_fin)
    if pre.sum() < 3 or seg.sum() < 20:
        r["nota"] = "datos insuficientes"
        return r
    y0 = y[pre].mean()
    sig = y[pre].std(ddof=1)
    ts_, ys_ = t[seg], y[seg]
    n_ss = max(3, int(round(SS_AVG_S / dt)))
    dT = ys_[-n_ss:].mean() - y0
    r.update(y0=y0, dT=dT, sigma=sig)

    n_w = max(5, int(round(STEADY_WIN_S / dt)))
    if len(ts_) > n_w:
        r["pend_final"] = np.polyfit(ts_[-n_w:], ys_[-n_w:], 1)[0]
        if abs(r["pend_final"]) > STEADY_SLOPE:
            r["nota"] += f"no estacionario al final del tramo (pendiente {r['pend_final']:.4f} °C/s); "
    # pendiente antes del escalón (residuo del tramo anterior)
    pre_w = (t >= t_s - STEADY_WIN_S) & (t < t_s)
    if pre_w.sum() > 5:
        p_pre = np.polyfit(t[pre_w], y[pre_w], 1)[0]
        if abs(p_pre) > STEADY_SLOPE:
            r["nota"] += f"el tramo anterior no estaba estable (pendiente {p_pre:.4f} °C/s); "

    if abs(dT) < 5 * max(sig, 1e-3):
        r["nota"] += "respuesta no identificable (cambio comparable al ruido)"
        return r
    r["K"] = dT / dQ

    w = max(1, int(round(SMOOTH_S / dt)))
    ys = pd.Series(ys_ - y0).rolling(w, center=True, min_periods=1).mean().to_numpy()
    tr = ts_ - t_s
    s = np.sign(dT)

    def cruce(frac):
        cond = s * ys >= frac * abs(dT)
        if not cond.any():
            return np.nan
        k = int(np.argmax(cond))
        if k == 0:
            return tr[0]
        a, b = s * ys[k - 1], s * ys[k]
        f = 0.0 if b == a else (frac * abs(dT) - a) / (b - a)
        return tr[k - 1] + f * (tr[k] - tr[k - 1])

    t28, t63 = cruce(0.283), cruce(0.632)
    r["t63"] = t63
    if np.isfinite(t28) and np.isfinite(t63):
        tau = 1.5 * (t63 - t28)
        theta = t63 - tau
        r["tau"] = tau
        r["theta"] = max(theta, 0.0)
        if theta < 0:
            r["nota"] += "theta<0 (se fijó en 0); "
    return r


def analisis_rapido(t, T, t_cambio):
    t, T = np.asarray(t), np.asarray(T)
    dt = np.median(np.diff(t))
    filas = []
    for (t_s, j, dQ, t_fin, nom) in escalones(t_cambio):
        for i in (1, 2):
            r = dos_puntos(t, T[:, i - 1], t_s, dQ, t_fin, dt)
            filas.append(dict(tramo=nom, canal=f"G{i}{j}", tipo="directo" if i == j else "cruzado",
                              t_escalon_s=t_s, dQ_pct=dQ, T_inicial_C=r["y0"], dT_C=r["dT"],
                              K_C_por_pct=r["K"], theta_s=r["theta"], tau_s=r["tau"],
                              t63_s=r["t63"], sigma_C=r["sigma"], nota=r["nota"]))
    return pd.DataFrame(filas)


def quitar_picos(T):
    """Filtro de mediana centrado por columna: elimina picos aislados sin desfasar la respuesta."""
    if DESPIKE_N <= 1:
        return np.asarray(T, dtype=float)
    return pd.DataFrame(T).rolling(DESPIKE_N, center=True, min_periods=1).median().to_numpy()


def respuesta_fopdt(t, Q, K, tau, theta):
    """Respuesta de un FOPDT a una entrada escalonada (suma de escalones, superposición)."""
    y = np.zeros_like(t)
    dQ = np.diff(Q, prepend=Q[0])
    for k in np.nonzero(np.abs(dQ) > 1e-9)[0]:
        tt = np.maximum(t - t[k] - theta, 0.0)
        y += K * dQ[k] * (1 - np.exp(-tt / tau))
    return y


def ajuste_minimos_cuadrados(t, T, Q, res2p):
    """
    Para cada salida Ti ajusta a la vez Gi1 y Gi2 sobre TODO el registro:
        Ti(t) = Ti0 + Gi1*Q1(t) + Gi2*Q2(t)
    Corrige el sesgo del método de dos puntos cuando un tramo no alcanzó el estacionario.
    """
    from scipy.optimize import least_squares
    t, T, Q = np.asarray(t), np.asarray(T), np.asarray(Q)
    base = t < TRAMOS[0][1]
    filas, modelos = [], {}
    for i in (1, 2):
        y = T[:, i - 1]
        T0 = y[base].mean()
        p0 = []
        for j in (1, 2):
            sel = res2p[res2p["canal"] == f"G{i}{j}"]
            K0 = np.nanmean(sel["K_C_por_pct"]) if len(sel) else 0.1
            ta0 = np.nanmean(sel["tau_s"]) if len(sel) else 150.0
            th0 = np.nanmean(sel["theta_s"]) if len(sel) else 10.0
            p0 += [K0 if np.isfinite(K0) else 0.1, ta0 if np.isfinite(ta0) else 150.0,
                   th0 if np.isfinite(th0) else 10.0]

        def resid(p):
            return T0 + respuesta_fopdt(t, Q[:, 0], *p[0:3]) + respuesta_fopdt(t, Q[:, 1], *p[3:6]) - y

        lo, hi = [-np.inf, 1.0, 0.0] * 2, [np.inf, 5000.0, 300.0] * 2
        p0 = list(np.clip(p0, np.array(lo) + 1e-6, np.array(hi) - 1e-6))   # punto inicial dentro de límites
        sol = least_squares(resid, p0, bounds=(lo, hi))
        rmse = np.sqrt(np.mean(sol.fun ** 2))
        p = sol.x
        modelos[i] = (T0, p)
        for j in (1, 2):
            K, tau, th = p[3 * (j - 1): 3 * j]
            filas.append(dict(canal=f"G{i}{j}", K_C_por_pct=K, tau_s=tau, theta_s=th,
                              RMSE_salida_C=rmse,
                              nota="(1 solo escalón en este ensayo)" if j != HEATER else ""))
    return pd.DataFrame(filas), modelos


def imprimir_resultados(res2p, resls):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 90)
    print(f"RESULTADOS RÁPIDOS – método de dos puntos (Smith) | entrada principal: Q{HEATER}")
    print("=" * 90)
    cols = ["tramo", "canal", "tipo", "dQ_pct", "dT_C", "K_C_por_pct", "theta_s", "tau_s", "sigma_C"]
    print(res2p[cols].round(4).to_string(index=False))
    notas = res2p[res2p["nota"] != ""]
    for _, r in notas.iterrows():
        print(f"  NOTA {r['tramo']} / {r['canal']}: {r['nota']}")
    if resls is not None:
        print("\n" + "=" * 90)
        print("AJUSTE POR MÍNIMOS CUADRADOS (todo el registro, superposición) – valores recomendados")
        print("=" * 90)
        print(resls.round(4).to_string(index=False))
    print("\nSon estimaciones de un solo ensayo; compáralas con el ensayo de la otra entrada.")


# =====================================================================================
# ADQUISICIÓN
# =====================================================================================
def main():
    path = nombre_libre(FILENAME)
    base, _ = os.path.splitext(path)

    print("Perfil del ensayo:")
    for ti, tf, q1, q2, nom in TRAMOS:
        print(f"  {ti/60:5.1f} - {tf/60:5.1f} min  {nom:16s}  Q1 = {q1:4.0f} %  Q2 = {q2:4.0f} %")
    print(f"Duración total: {DURACION_S/60:.1f} min\n")

    # ---------------- Gráfica en tiempo real ----------------
    plt.ion()
    fig, (axT, axQ) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    lT1, = axT.plot([], [], label="T1", color="tab:blue")
    lT2, = axT.plot([], [], label="T2", color="tab:red")
    lQ1, = axQ.step([], [], where="post", label="Q1", color="tab:blue")
    lQ2, = axQ.step([], [], where="post", label="Q2", color="tab:red")
    axT.set_ylabel("Temperatura [°C]")
    axQ.set_ylabel("Potencia [%]")
    axQ.set_xlabel("Tiempo [s]")
    axT.set_title(f"Ensayo MIMO: Q{HEATER} {' → '.join(f'{x:.0f}' for x in Q_MAIN_LEVELS)} %, "
                  f"luego Q{OTHER} = {Q_CROSS:.0f} %")
    for ax in (axT, axQ):
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        ax.set_xlim(0, DURACION_S)
    axQ.set_ylim(-5, 105)
    fig.tight_layout()

    t, T1, T2, Q1, Q2, fase = [], [], [], [], [], []
    t_cambio = [None] * len(TRAMOS)   # instante real en que empezó cada tramo
    motivo = "perfil completo"
    n_picos = 0          # lecturas aisladas por encima de T_MAX (ruido)

    print("Iniciando TCLab...")
    with tclab.TCLab() as lab:
        lab.Q1(0)
        lab.Q2(0)
        print(f"T inicial: T1 = {lab.T1:.2f} °C, T2 = {lab.T2:.2f} °C. Ctrl+C para abortar.")

        f = open(path, "w", newline="", encoding="utf-8")
        wr = csv.writer(f)
        wr.writerow(["tiempo_s", "fase", "Q1_pct", "Q2_pct", "T1_C", "T2_C"])

        t0 = time.perf_counter()
        k = 0
        tramo_actual = -1
        try:
            while True:
                objetivo = k * TS
                restante = objetivo - (time.perf_counter() - t0)
                plt.pause(max(restante, 0.001))   # espera y procesa la ventana
                now = time.perf_counter() - t0

                if not plt.fignum_exists(fig.number):
                    motivo = "ventana cerrada por el usuario"
                    break
                if now >= DURACION_S:
                    break

                q1, q2, nom, idx = potencia_en(now)
                if idx != tramo_actual:
                    tramo_actual = idx
                    t_cambio[idx] = now
                    if idx > 0:
                        for ax in (axT, axQ):
                            ax.axvline(now, color="k", ls="--", lw=1)
                        print(f"[{now:7.1f} s] {nom}: Q1 = {q1:.0f} %, Q2 = {q2:.0f} %")
                lab.Q1(q1)
                lab.Q2(q2)

                a, b = lab.T1, lab.T2
                t.append(now); T1.append(a); T2.append(b)
                Q1.append(q1); Q2.append(q2); fase.append(nom)
                wr.writerow([f"{now:.3f}", nom, q1, q2, f"{a:.3f}", f"{b:.3f}"])
                f.flush()

                # Seguridad: mediana de las últimas T_MAX_N lecturas (robusta a picos de ruido)
                if max(a, b) >= T_MAX:
                    n_picos += 1
                    print(f"[{now:7.1f} s] [aviso] lectura >= {T_MAX:.0f} °C (T1={a:.2f}, T2={b:.2f}); "
                          f"se verifica con la mediana de las últimas {T_MAX_N} muestras")
                med1 = float(np.median(T1[-T_MAX_N:]))
                med2 = float(np.median(T2[-T_MAX_N:]))
                if max(med1, med2) >= T_MAX:
                    motivo = (f"LÍMITE DE TEMPERATURA ({T_MAX:.0f} °C) superado "
                              f"(mediana T1={med1:.2f} °C, T2={med2:.2f} °C)")
                    break

                # Gráfica
                lT1.set_data(t, T1); lT2.set_data(t, T2)
                lQ1.set_data(t, Q1); lQ2.set_data(t, Q2)
                axT.relim(); axT.autoscale_view(scalex=False)

                if k % max(1, int(30 / TS)) == 0:
                    falta = DURACION_S - now
                    print(f"[{now:7.1f} s] T1={a:6.2f} °C  T2={b:6.2f} °C  "
                          f"Q1={q1:3.0f} %  Q2={q2:3.0f} %  (faltan {falta/60:4.1f} min)")
                k += 1
        except KeyboardInterrupt:
            motivo = "abortado por el usuario (Ctrl+C)"
        except Exception as e:
            motivo = f"ERROR durante la adquisición ({type(e).__name__}: {e})"
        finally:
            try:
                lab.Q1(0)
                lab.Q2(0)       # apagado seguro SIEMPRE
            except Exception as e:
                print(f"[ATENCIÓN] no se pudo enviar el apagado al TCLab ({e}). Desconéctalo de la alimentación.")
            f.close()
            completo = motivo == "perfil completo"
            print("\n" + "#" * 78)
            print(f"#  FIN DEL ENSAYO a los {t[-1] if t else 0:.0f} s de {DURACION_S:.0f} s "
                  f"({'COMPLETO' if completo else 'INCOMPLETO'})")
            print(f"#  Motivo: {motivo}")
            if t:
                print(f"#  Última lectura: T1 = {T1[-1]:.2f} °C, T2 = {T2[-1]:.2f} °C, Q1 = {Q1[-1]:.0f} %, Q2 = {Q2[-1]:.0f} %")
            print(f"#  Picos aislados >= {T_MAX:.0f} °C ignorados (ruido): {n_picos}")
            print("#  Calentadores apagados.")
            print("#" * 78)
            print(f"Datos guardados en: {os.path.abspath(path)}")

    # ---------------- Cierre: metadatos, figura y análisis ----------------
    meta = [
        f"fecha: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"entrada_principal: Q{HEATER}", f"entrada_cruzada: Q{OTHER}",
        f"niveles_principal_pct: {Q_MAIN_LEVELS}", f"escalon_cruzado_pct: {Q_CROSS}",
        f"baseline_s: {BASELINE_S}", f"duracion_tramo_s: {SEG_S}", f"Ts_s: {TS}",
        f"t_inicio_tramos_s: {[None if x is None else round(x, 3) for x in t_cambio]}",
        f"T_max_C: {T_MAX}", f"T_max_mediana_N: {T_MAX_N}", f"picos_ignorados: {n_picos}",
        f"motivo_fin: {motivo}",
        "columnas: tiempo_s, fase, Q1_pct, Q2_pct, T1_C, T2_C",
    ]
    with open(base + "_meta.txt", "w", encoding="utf-8") as fm:
        fm.write("\n".join(meta) + "\n")
    try:
        fig.savefig(base + ".png", dpi=150)
    except Exception:
        pass

    if motivo != "perfil completo":
        print("\n[ATENCIÓN] El ensayo NO se completó: los parámetros que se calculen abajo son solo de los\n"
              "tramos registrados y los del último tramo (cortado) no son confiables.")
    if len(t) < BASELINE_S / TS + 20 or t_cambio[1] is None:
        print("No hay suficientes datos después del primer escalón para el análisis.")
    else:
        Tm = quitar_picos(np.column_stack([T1, T2]))   # el análisis usa datos sin picos
        Qm = np.column_stack([Q1, Q2])
        res2p = analisis_rapido(t, Tm, t_cambio)
        resls = None
        if DO_LSQ_FIT:
            try:
                resls, modelos = ajuste_minimos_cuadrados(t, Tm, Qm, res2p)
                # Superponer el modelo ajustado en la gráfica
                tt = np.asarray(t)
                for i, color in ((1, "navy"), (2, "darkred")):
                    T0, p = modelos[i]
                    y_hat = T0 + respuesta_fopdt(tt, Qm[:, 0], *p[0:3]) + respuesta_fopdt(tt, Qm[:, 1], *p[3:6])
                    axT.plot(tt, y_hat, "--", color=color, lw=1.2, label=f"modelo T{i}")
                axT.legend(loc="upper left")
                fig.savefig(base + ".png", dpi=150)
            except ImportError:
                print("[aviso] scipy no está instalado: se omite el ajuste por mínimos cuadrados.")
            except Exception as e:
                print(f"[aviso] el ajuste por mínimos cuadrados falló: {e}")
        imprimir_resultados(res2p, resls)
        res2p.to_csv(base + "_dos_puntos.csv", index=False)
        if resls is not None:
            resls.to_csv(base + "_minimos_cuadrados.csv", index=False)
        print(f"\nTablas guardadas en: {base}_dos_puntos.csv"
              + (f" y {base}_minimos_cuadrados.csv" if resls is not None else ""))

    if plt.fignum_exists(fig.number):
        plt.ioff()
        print("\nCierra la ventana de la gráfica para terminar.")
        plt.show()


if __name__ == "__main__":
    main()
