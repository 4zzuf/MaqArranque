"""Simulación del motor asíncrono trifásico orientada a analizar tiempos y corrientes
de arranque y frenado.

El código reproduce la lógica del script original de MATLAB, pero simplificado para
centrar el estudio en:
- Arranque a tensión plena.
- Frenado por desenergización (motor en rueda libre).
- Frenado por *plugging* (secuencia de tensión invertida en el eje q).
- Frenado dinámico por inyección de corriente continua en el estator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Parámetros de la máquina y de la simulación
# ---------------------------------------------------------------------------


@dataclass
class ParametrosSimulacion:
    """Constantes eléctricas, mecánicas y temporales del motor."""

    Rs: float = 3.7568
    Rr: float = 3.1329
    Rm: float = 2881.98
    lm_base: float = 0.569
    incremento_lr: float = 0.01105
    incremento_ls: float = 0.01624
    J: float = 0.00397
    D: float = 0.001764
    p: int = 2
    a: float = 1.2125
    V1: float = 230.0
    fb: float = 60.0
    Tm: float = 0.0
    ti: float = 0.0
    tf: float = 2.0
    dt: float = 1e-4

    @property
    def wb(self) -> float:
        """Frecuencia eléctrica síncrona (rad/s)."""

        return 2.0 * math.pi * self.fb

    @property
    def Vmax(self) -> float:
        """Valor máximo de la tensión de fase."""

        return math.sqrt(2.0) * self.V1


PerfilTension = Callable[[float, "ParametrosSimulacion"], tuple[np.ndarray, float]]


@dataclass
class MetricasEvento:
    """Magnitudes características registradas en un evento dinámico."""

    tiempo: Optional[float]
    corriente: Optional[float]
    velocidad: Optional[float]


@dataclass
class ResultadosSimulacion:
    """Variables principales a monitorear en cada escenario dinámico."""

    tiempo: np.ndarray
    corrientes_estator_dq: np.ndarray
    corrientes_estator_abc: np.ndarray
    torque: np.ndarray
    velocidad_rotor: np.ndarray
    razon_saturacion: Optional[np.ndarray] = None
    metricas: Optional[Dict[str, MetricasEvento]] = None


# ---------------------------------------------------------------------------
# Transformaciones y perfiles de tensión
# ---------------------------------------------------------------------------


def matriz_park(theta: float) -> np.ndarray:
    """Matriz de Park directa (de abc a dq0)."""

    base = np.array(
        [
            [math.cos(theta), math.cos(theta - 2.0 * math.pi / 3.0), math.cos(theta + 2.0 * math.pi / 3.0)],
            [math.sin(theta), math.sin(theta - 2.0 * math.pi / 3.0), math.sin(theta + 2.0 * math.pi / 3.0)],
            [0.5, 0.5, 0.5],
        ]
    )
    return (2.0 / 3.0) * base


def perf_tension_completa(t: float, params: ParametrosSimulacion) -> tuple[np.ndarray, float]:
    """Entrega la tensión trifásica balanceada a tensión plena."""

    frecuencia = params.wb
    vabc = params.Vmax * np.array(
        [
            math.sin(frecuencia * t),
            math.sin(frecuencia * t - 2.0 * math.pi / 3.0),
            math.sin(frecuencia * t + 2.0 * math.pi / 3.0),
        ]
    )
    return vabc, frecuencia


def perf_tension_nula(_: float, params: ParametrosSimulacion) -> tuple[np.ndarray, float]:
    """Perfile de desenergización: tensión cero en todas las fases."""

    return np.zeros(3), 0.0


def perf_tension_plugging(t: float, params: ParametrosSimulacion) -> tuple[np.ndarray, float]:
    """Perfil para frenado por plugging (inversión del componente q)."""

    frecuencia = params.wb
    theta = frecuencia * t
    matriz = matriz_park(theta)
    matriz_inv = np.linalg.inv(matriz)

    vabc_base = params.Vmax * np.array(
        [
            math.sin(frecuencia * t),
            math.sin(frecuencia * t - 2.0 * math.pi / 3.0),
            math.sin(frecuencia * t + 2.0 * math.pi / 3.0),
        ]
    )

    vqds = matriz @ vabc_base
    vqds[1] *= -1.0  # inversión del componente q para generar par contrario
    vabc = matriz_inv @ vqds
    return vabc, frecuencia


def perf_tension_inyeccion_cc(_: float, params: ParametrosSimulacion) -> tuple[np.ndarray, float]:
    """Perfil de frenado dinámico mediante inyección de tensión continua en d."""

    vd_dc = params.V1
    matriz = matriz_park(0.0)
    matriz_inv = np.linalg.inv(matriz)
    vabc = matriz_inv @ np.array([vd_dc, 0.0, 0.0])
    return vabc, 0.0


# ---------------------------------------------------------------------------
# Integradores numéricos (Runge-Kutta de cuarto orden)
# ---------------------------------------------------------------------------


def paso_rk4_corriente(I: np.ndarray, U: np.ndarray, R: np.ndarray, L_inv: np.ndarray, G: np.ndarray, h: float) -> np.ndarray:
    """Integra las ecuaciones eléctricas del modelo D-Q."""

    def di_dt(corriente: np.ndarray) -> np.ndarray:
        # Ecuación diferencial del modelo eléctrico:
        #   dI/dt = L^{-1} · (U - (R + G) · I)
        # que proviene de reorganizar U = (R + G) · I + L · (dI/dt).
        return L_inv @ (U - (R + G) @ corriente)

    k1 = h * di_dt(I)
    k2 = h * di_dt(I + 0.5 * k1)
    k3 = h * di_dt(I + 0.5 * k2)
    k4 = h * di_dt(I + k3)
    return I + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def paso_rk4_velocidad(wr: float, torque: float, params: ParametrosSimulacion, h: float) -> float:
    """Integra la ecuación mecánica del rotor."""

    def dwr_dt(velocidad: float) -> float:
        # Ecuación mecánica del rotor:
        #   J · dω_r/dt = T_e - T_m - D · ω_r
        # ⇒ dω_r/dt = (T_e - T_m - D · ω_r) / J.
        return (torque - params.Tm - params.D * velocidad) / params.J

    l1 = h * dwr_dt(wr)
    l2 = h * dwr_dt(wr + 0.5 * l1)
    l3 = h * dwr_dt(wr + 0.5 * l2)
    l4 = h * dwr_dt(wr + l3)
    nueva_velocidad = wr + (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0
    return max(nueva_velocidad, 0.0)


# ---------------------------------------------------------------------------
# Construcción del modelo y evaluación de eventos
# ---------------------------------------------------------------------------


def construir_matrices(
    params: ParametrosSimulacion,
    lm: float,
    ls: float,
    lr: float,
    w_sincrona: float,
    w_rotor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera las matrices R, L^{-1} y G del modelo eléctrico."""

    R = np.array(
        [
            [params.Rs, 0.0, 0.0, 0.0],
            [0.0, params.Rs, 0.0, 0.0],
            [0.0, 0.0, params.Rr, 0.0],
            [0.0, 0.0, 0.0, params.Rr],
        ]
    )

    L = np.array(
        [
            [ls, 0.0, lm, 0.0],
            [0.0, ls, 0.0, lm],
            [lm, 0.0, lr, 0.0],
            [0.0, lm, 0.0, lr],
        ]
    )

    velocidad_deslizamiento = w_sincrona - 0.5 * params.p * w_rotor
    G = -np.array(
        [
            [0.0, w_sincrona * ls, 0.0, w_sincrona * lm],
            [-w_sincrona * ls, 0.0, -w_sincrona * lm, 0.0],
            [0.0, lm * velocidad_deslizamiento, 0.0, lr * velocidad_deslizamiento],
            [-lm * velocidad_deslizamiento, 0.0, -lr * velocidad_deslizamiento, 0.0],
        ]
    )

    L_inv = np.linalg.inv(L)
    return R, L_inv, G


def calcular_metrica_evento(
    params: ParametrosSimulacion,
    tiempo: np.ndarray,
    velocidad_rotor: np.ndarray,
    corrientes: np.ndarray,
    tipo_evento: str,
) -> MetricasEvento:
    """Determina el tiempo característico y la corriente pico para un evento."""

    velocidad_sincrona = 2.0 * params.wb / params.p
    magnitud_corriente = np.linalg.norm(corrientes[:, :2], axis=1)

    if magnitud_corriente.size == 0:
        return MetricasEvento(tiempo=None, corriente=None, velocidad=None)

    velocidad_evento: Optional[float]

    if tipo_evento == "start":
        umbral = 0.95 * velocidad_sincrona
        indices = np.where(velocidad_rotor >= umbral)[0]
        if indices.size > 0:
            idx = int(indices[0])
            tiempo_evento = float(tiempo[idx])
            corriente_pico = float(np.max(magnitud_corriente[: idx + 1]))
            velocidad_evento = float(velocidad_rotor[idx])
        else:
            tiempo_evento = None
            corriente_pico = float(np.max(magnitud_corriente))
            velocidad_evento = None
    elif tipo_evento == "stop":
        indices = np.where(velocidad_rotor <= 0.0)[0]
        if indices.size > 0:
            idx = int(indices[0])
            if idx > 0:
                # Interpolación lineal para estimar el instante exacto en el que ω_r cruza cero
                t1, t2 = tiempo[idx - 1], tiempo[idx]
                w1, w2 = velocidad_rotor[idx - 1], velocidad_rotor[idx]
                if abs(w2 - w1) > 1e-12:
                    tiempo_evento = float(t1 + (0.0 - w1) * (t2 - t1) / (w2 - w1))
                else:
                    tiempo_evento = float(t2)
            else:
                tiempo_evento = float(tiempo[idx])
            corriente_pico = float(np.max(magnitud_corriente[: idx + 1]))
            velocidad_evento = 0.0
        else:
            # Si no se llegó a cero, usar el punto de menor velocidad registrado
            idx = int(np.argmin(velocidad_rotor))
            tiempo_evento = float(tiempo[idx]) if velocidad_rotor[idx] < velocidad_sincrona else None
            corriente_pico = float(np.max(magnitud_corriente[: idx + 1]))
            velocidad_evento = float(velocidad_rotor[idx]) if tiempo_evento is not None else None
    else:
        raise ValueError(f"Tipo de evento no soportado: {tipo_evento}")

    return MetricasEvento(tiempo=tiempo_evento, corriente=corriente_pico, velocidad=velocidad_evento)


# ---------------------------------------------------------------------------
# Núcleo de la simulación
# ---------------------------------------------------------------------------


def simular_iteracion(
    params: ParametrosSimulacion,
    perfil_saturacion: Optional[np.ndarray] = None,
    corriente_inicial: Optional[np.ndarray] = None,
    velocidad_inicial: float = 0.0,
    perfil_tension: PerfilTension = perf_tension_completa,
    permitir_actualizar_saturacion: bool = True,
    etiqueta_evento: str = "evento",
    tipo_evento: str = "start",
) -> ResultadosSimulacion:
    """Ejecuta una simulación temporal para un escenario concreto."""

    tiempo = np.arange(params.ti, params.tf + params.dt / 2.0, params.dt)
    pasos = tiempo.size

    corrientes_dq = np.zeros((pasos, 4))
    corrientes_abc = np.zeros((pasos, 3))
    torque = np.zeros(pasos)
    velocidad_rotor = np.zeros(pasos)
    razon_saturacion = (
        np.ones(pasos) if perfil_saturacion is None else perfil_saturacion.astype(float).copy()
    )

    I = np.zeros(4) if corriente_inicial is None else np.array(corriente_inicial, dtype=float)
    wr = float(velocidad_inicial)

    for idx, t in enumerate(tiempo):
        # 1) Ajuste de inductancias según el perfil de saturación disponible
        sat = razon_saturacion[idx]
        if perfil_saturacion is None and permitir_actualizar_saturacion:
            sat = max(sat, 1.0)
            razon_saturacion[idx] = sat

        lm = params.lm_base / sat
        ls = lm + params.incremento_ls
        lr = lm + params.incremento_lr

        # 2) Tensión aplicada y matrices del modelo eléctrico
        vabc_t, w_sincrona = perfil_tension(t, params)
        R, L_inv, G = construir_matrices(params, lm, ls, lr, w_sincrona, wr)

        # 3) Transformación de Park para obtener tensiones dq
        theta = w_sincrona * t
        Kqds = matriz_park(theta)
        Kqds_inv = np.linalg.inv(Kqds)
        vqds = Kqds @ vabc_t
        U = np.array([vqds[0], vqds[1], 0.0, 0.0])

        # 4) Integración de corrientes y velocidad.
        #    Se aplica RK4 sobre las ecuaciones:
        #      dI/dt = L^{-1}(U - (R + G)I)
        #      dω_r/dt = (T_e - T_m - D ω_r)/J
        I = paso_rk4_corriente(I, U, R, L_inv, G, params.dt)
        corrientes_dq[idx] = I

        torque[idx] = -1.5 * 0.5 * params.p * lm * (I[3] * I[0] - I[2] * I[1])
        wr = paso_rk4_velocidad(wr, torque[idx], params, params.dt)
        velocidad_rotor[idx] = wr

        # 5) Transformación inversa para obtener corrientes de fase del estator
        corrientes_abc[idx] = Kqds_inv @ np.array([I[0], I[1], 0.0])

        # 6) Actualización del modelo de saturación a partir de la corriente lineal
        if perfil_saturacion is None and permitir_actualizar_saturacion:
            magnitud_linea = abs(I[0] + 1j * I[1]) / math.sqrt(2.0)
            if magnitud_linea > 1e-9:
                lsat = (310.1 * magnitud_linea - 2.423 - 28.25) / 172.6
                razon_saturacion[idx] = max(lsat / magnitud_linea, 1.0) if lsat > 0 else 1.0
            else:
                razon_saturacion[idx] = 1.0

    metrica = calcular_metrica_evento(params, tiempo, velocidad_rotor, corrientes_dq, tipo_evento)
    metricas = {etiqueta_evento: metrica}

    return ResultadosSimulacion(
        tiempo=tiempo,
        corrientes_estator_dq=corrientes_dq,
        corrientes_estator_abc=corrientes_abc,
        torque=torque,
        velocidad_rotor=velocidad_rotor,
        razon_saturacion=razon_saturacion,
        metricas=metricas,
    )


def simular_motor(params: Optional[ParametrosSimulacion] = None) -> Dict[str, ResultadosSimulacion]:
    """Configura los tres escenarios solicitados y devuelve sus resultados."""

    if params is None:
        params = ParametrosSimulacion()

    arranque_lineal = simular_iteracion(
        params,
        etiqueta_evento="arranque_lineal",
        tipo_evento="start",
    )

    arranque_saturado = simular_iteracion(
        params,
        perfil_saturacion=arranque_lineal.razon_saturacion,
        permitir_actualizar_saturacion=False,
        etiqueta_evento="arranque_saturado",
        tipo_evento="start",
    )

    corriente_inicial = arranque_lineal.corrientes_estator_dq[-1]
    velocidad_inicial = arranque_lineal.velocidad_rotor[-1]

    frenado_desenergizado = simular_iteracion(
        params,
        perfil_saturacion=arranque_lineal.razon_saturacion,
        corriente_inicial=corriente_inicial,
        velocidad_inicial=velocidad_inicial,
        perfil_tension=perf_tension_nula,
        permitir_actualizar_saturacion=False,
        etiqueta_evento="frenado_desenergizado",
        tipo_evento="stop",
    )

    frenado_plugging = simular_iteracion(
        params,
        perfil_saturacion=arranque_lineal.razon_saturacion,
        corriente_inicial=corriente_inicial,
        velocidad_inicial=velocidad_inicial,
        perfil_tension=perf_tension_plugging,
        permitir_actualizar_saturacion=False,
        etiqueta_evento="frenado_plugging",
        tipo_evento="stop",
    )

    frenado_inyeccion_cc = simular_iteracion(
        params,
        perfil_saturacion=arranque_lineal.razon_saturacion,
        corriente_inicial=corriente_inicial,
        velocidad_inicial=velocidad_inicial,
        perfil_tension=perf_tension_inyeccion_cc,
        permitir_actualizar_saturacion=False,
        etiqueta_evento="frenado_inyeccion_cc",
        tipo_evento="stop",
    )

    return {
        "arranque_lineal": arranque_lineal,
        "arranque_saturado": arranque_saturado,
        "frenado_desenergizado": frenado_desenergizado,
        "frenado_plugging": frenado_plugging,
        "frenado_inyeccion_cc": frenado_inyeccion_cc,
    }


# ---------------------------------------------------------------------------
# Utilidades de graficación para corrientes y velocidad
# ---------------------------------------------------------------------------


def graficar_escenario(resultados: ResultadosSimulacion, titulo: str, etiqueta_evento: str) -> None:
    """Genera gráficos básicos de velocidad y corriente del estator."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib no está disponible. No se generarán gráficos.")
        return

    metricas = (resultados.metricas or {}).get(etiqueta_evento)
    magnitud_corriente = np.linalg.norm(resultados.corrientes_estator_dq[:, :2], axis=1)

    fig, (ax_velocidad, ax_corriente) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_velocidad.plot(resultados.tiempo, resultados.velocidad_rotor, label="ω_r (rad/s)")
    ax_velocidad.axhline(0.0, color="tab:gray", linestyle=":", linewidth=1.0, label="ω_r = 0")
    ax_velocidad.set_ylabel("Velocidad (rad/s)")
    ax_velocidad.set_title(f"{titulo} - Velocidad del rotor")

    ax_corriente.plot(resultados.tiempo, magnitud_corriente, color="tab:red", label="|I_s|")
    ax_corriente.set_ylabel("Corriente (A)")
    ax_corriente.set_xlabel("Tiempo (s)")
    ax_corriente.set_title(f"{titulo} - Corriente del estator")

    if metricas and metricas.tiempo is not None:
        for eje, texto in [
            (ax_velocidad, "Tiempo característico"),
            (ax_corriente, "Corriente característica"),
        ]:
            eje.axvline(metricas.tiempo, color="tab:green", linestyle="--", alpha=0.7)
            eje.text(
                metricas.tiempo,
                eje.get_ylim()[1] * 0.9,
                texto,
                rotation=90,
                verticalalignment="bottom",
                horizontalalignment="right",
                fontsize=9,
                backgroundcolor="white",
            )
        if metricas.velocidad is not None:
            ax_velocidad.scatter(metricas.tiempo, metricas.velocidad, color="tab:purple", zorder=5)
            ax_velocidad.text(
                metricas.tiempo,
                metricas.velocidad,
                "  ω_r en evento",
                color="tab:purple",
                verticalalignment="bottom",
            )

    ax_velocidad.legend()
    ax_corriente.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Ejecución desde la línea de comandos
# ---------------------------------------------------------------------------


def _imprimir_metricas(resultados: ResultadosSimulacion, nombre: str) -> None:
    """Muestra en consola las métricas principales de un escenario."""

    metricas = resultados.metricas or {}
    if not metricas:
        print(f"No hay métricas para {nombre}.")
        return

    print(f"\nEscenario: {nombre}")
    for etiqueta, valores in metricas.items():
        tiempo_txt = f"{valores.tiempo:.6f} s" if valores.tiempo is not None else "N/D"
        corriente_txt = f"{valores.corriente:.6f} A" if valores.corriente is not None else "N/D"
        velocidad_txt = f"{valores.velocidad:.6f} rad/s" if valores.velocidad is not None else "N/D"
        print(f"  Evento '{etiqueta}':")
        print(f"    Tiempo característico: {tiempo_txt}")
        print(f"    Corriente pico: {corriente_txt}")
        print(f"    Velocidad asociada: {velocidad_txt}")


def main() -> None:
    """Ejecuta la simulación y presenta los resultados más relevantes."""

    simulaciones = simular_motor()

    print("Simulación completada.")
    for nombre, resultados in simulaciones.items():
        _imprimir_metricas(resultados, nombre)

    # Graficar escenarios principales (si matplotlib está disponible)
    graficar_escenario(simulaciones["arranque_lineal"], "Arranque lineal", "arranque_lineal")
    graficar_escenario(
        simulaciones["frenado_desenergizado"], "Frenado por desenergización", "frenado_desenergizado"
    )
    graficar_escenario(simulaciones["frenado_plugging"], "Frenado por plugging", "frenado_plugging")
    graficar_escenario(
        simulaciones["frenado_inyeccion_cc"],
        "Frenado dinámico por inyección de CC",
        "frenado_inyeccion_cc",
    )


if __name__ == "__main__":
    main()
