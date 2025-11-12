"""Simulación del motor asíncrono trifásico orientada a analizar tiempos y corrientes de arranque y frenado.

El código reproduce la lógica del script original de MATLAB, pero simplificado para centrar el estudio en:
- Arranque a tensión plena.
- Frenado por desenergización.
- Frenado dinámico por contracorriente.
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
class SimulationParameters:
    """Constantes eléctricas, mecánicas y temporales del motor."""

    Rs: float = 3.7568
    Rr: float = 3.1329
    Rm: float = 2881.98
    lm_base: float = 0.569
    lr_offset: float = 0.01105
    ls_offset: float = 0.01624
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


VoltageProfile = Callable[[float, "SimulationParameters"], tuple[np.ndarray, float]]


@dataclass
class EventMetrics:
    """Magnitudes características registradas en un evento dinámico."""

    time: Optional[float]
    current: Optional[float]
    speed: Optional[float]


@dataclass
class SimulationResults:
    """Variables principales a monitorear en cada escenario dinámico."""

    time: np.ndarray
    stator_currents_dq: np.ndarray
    stator_currents_abc: np.ndarray
    torque: np.ndarray
    rotor_speed: np.ndarray
    saturation_ratio: Optional[np.ndarray] = None
    metrics: Optional[Dict[str, EventMetrics]] = None


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


def perf_tension_completa(t: float, params: SimulationParameters) -> tuple[np.ndarray, float]:
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


def perf_tension_nula(_: float, params: SimulationParameters) -> tuple[np.ndarray, float]:
    """Perfile de desenergización: tensión cero en todas las fases."""

    return np.zeros(3), 0.0


def perf_tension_invertida(t: float, params: SimulationParameters) -> tuple[np.ndarray, float]:
    """Perfil para frenado por contracorriente (inversión de fase)."""

    frecuencia = -params.wb
    vabc = params.Vmax * np.array(
        [
            math.sin(frecuencia * t),
            math.sin(frecuencia * t - 2.0 * math.pi / 3.0),
            math.sin(frecuencia * t + 2.0 * math.pi / 3.0),
        ]
    )
    return vabc, frecuencia


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


def paso_rk4_velocidad(wr: float, torque: float, params: SimulationParameters, h: float) -> float:
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
    params: SimulationParameters,
    lm: float,
    ls: float,
    lr: float,
    w_syn: float,
    w_r: float,
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

    velocidad_deslizamiento = w_syn - 0.5 * params.p * w_r
    G = -np.array(
        [
            [0.0, w_syn * ls, 0.0, w_syn * lm],
            [-w_syn * ls, 0.0, -w_syn * lm, 0.0],
            [0.0, lm * velocidad_deslizamiento, 0.0, lr * velocidad_deslizamiento],
            [-lm * velocidad_deslizamiento, 0.0, -lr * velocidad_deslizamiento, 0.0],
        ]
    )

    L_inv = np.linalg.inv(L)
    return R, L_inv, G


def calcular_metrica_evento(
    params: SimulationParameters,
    tiempo: np.ndarray,
    velocidad_rotor: np.ndarray,
    corrientes: np.ndarray,
    tipo_evento: str,
) -> EventMetrics:
    """Determina el tiempo característico y la corriente pico para un evento."""

    velocidad_sincrona = 2.0 * params.wb / params.p
    magnitud_corriente = np.linalg.norm(corrientes[:, :2], axis=1)

    if magnitud_corriente.size == 0:
        return EventMetrics(time=None, current=None)

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

    return EventMetrics(time=tiempo_evento, current=corriente_pico, speed=velocidad_evento)


# ---------------------------------------------------------------------------
# Núcleo de la simulación
# ---------------------------------------------------------------------------


def simular_iteracion(
    params: SimulationParameters,
    saturation_profile: Optional[np.ndarray] = None,
    initial_current: Optional[np.ndarray] = None,
    initial_speed: float = 0.0,
    voltage_profile: VoltageProfile = perf_tension_completa,
    allow_saturation_update: bool = True,
    event_label: str = "evento",
    event_type: str = "start",
) -> SimulationResults:
    """Ejecuta una simulación temporal para un escenario concreto."""

    tiempo = np.arange(params.ti, params.tf + params.dt / 2.0, params.dt)
    pasos = tiempo.size

    corrientes_dq = np.zeros((pasos, 4))
    corrientes_abc = np.zeros((pasos, 3))
    torque = np.zeros(pasos)
    velocidad_rotor = np.zeros(pasos)
    razon_saturacion = (
        np.ones(pasos) if saturation_profile is None else saturation_profile.astype(float).copy()
    )

    I = np.zeros(4) if initial_current is None else np.array(initial_current, dtype=float)
    wr = float(initial_speed)

    for idx, t in enumerate(tiempo):
        # 1) Ajuste de inductancias según el perfil de saturación disponible
        sat = razon_saturacion[idx]
        if saturation_profile is None and allow_saturation_update:
            sat = max(sat, 1.0)
            razon_saturacion[idx] = sat

        lm = params.lm_base / sat
        ls = lm + params.ls_offset
        lr = lm + params.lr_offset

        # 2) Tensión aplicada y matrices del modelo eléctrico
        vabc_t, w_syn = voltage_profile(t, params)
        R, L_inv, G = construir_matrices(params, lm, ls, lr, w_syn, wr)

        # 3) Transformación de Park para obtener tensiones dq
        theta = w_syn * t
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
        if saturation_profile is None and allow_saturation_update:
            magnitud_linea = abs(I[0] + 1j * I[1]) / math.sqrt(2.0)
            if magnitud_linea > 1e-9:
                lsat = (310.1 * magnitud_linea - 2.423 - 28.25) / 172.6
                razon_saturacion[idx] = max(lsat / magnitud_linea, 1.0) if lsat > 0 else 1.0
            else:
                razon_saturacion[idx] = 1.0

    metrica = calcular_metrica_evento(params, tiempo, velocidad_rotor, corrientes_dq, event_type)
    metricas = {event_label: metrica}

    return SimulationResults(
        time=tiempo,
        stator_currents_dq=corrientes_dq,
        stator_currents_abc=corrientes_abc,
        torque=torque,
        rotor_speed=velocidad_rotor,
        saturation_ratio=razon_saturacion,
        metrics=metricas,
    )


def simular_motor(params: Optional[SimulationParameters] = None) -> Dict[str, SimulationResults]:
    """Configura los tres escenarios solicitados y devuelve sus resultados."""

    if params is None:
        params = SimulationParameters()

    arranque_lineal = simular_iteracion(
        params,
        event_label="arranque_lineal",
        event_type="start",
    )

    arranque_saturado = simular_iteracion(
        params,
        saturation_profile=arranque_lineal.saturation_ratio,
        allow_saturation_update=False,
        event_label="arranque_saturado",
        event_type="start",
    )

    corriente_inicial = arranque_lineal.stator_currents_dq[-1]
    velocidad_inicial = arranque_lineal.rotor_speed[-1]

    frenado_desenergizado = simular_iteracion(
        params,
        saturation_profile=arranque_lineal.saturation_ratio,
        initial_current=corriente_inicial,
        initial_speed=velocidad_inicial,
        voltage_profile=perf_tension_nula,
        allow_saturation_update=False,
        event_label="frenado_desenergizado",
        event_type="stop",
    )

    frenado_contracorriente = simular_iteracion(
        params,
        saturation_profile=arranque_lineal.saturation_ratio,
        initial_current=corriente_inicial,
        initial_speed=velocidad_inicial,
        voltage_profile=perf_tension_invertida,
        allow_saturation_update=False,
        event_label="frenado_contracorriente",
        event_type="stop",
    )

    return {
        "arranque_lineal": arranque_lineal,
        "arranque_saturado": arranque_saturado,
        "frenado_desenergizado": frenado_desenergizado,
        "frenado_contracorriente": frenado_contracorriente,
    }


# ---------------------------------------------------------------------------
# Utilidades de graficación para corrientes y velocidad
# ---------------------------------------------------------------------------


def graficar_escenario(resultados: SimulationResults, titulo: str, etiqueta_evento: str) -> None:
    """Genera gráficos básicos de velocidad y corriente del estator."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib no está disponible. No se generarán gráficos.")
        return

    metricas = (resultados.metrics or {}).get(etiqueta_evento)
    magnitud_corriente = np.linalg.norm(resultados.stator_currents_dq[:, :2], axis=1)

    fig, (ax_velocidad, ax_corriente) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_velocidad.plot(resultados.time, resultados.rotor_speed, label="ω_r (rad/s)")
    ax_velocidad.axhline(0.0, color="tab:gray", linestyle=":", linewidth=1.0, label="ω_r = 0")
    ax_velocidad.set_ylabel("Velocidad (rad/s)")
    ax_velocidad.set_title(f"{titulo} - Velocidad del rotor")

    ax_corriente.plot(resultados.time, magnitud_corriente, color="tab:red", label="|I_s|")
    ax_corriente.set_ylabel("Corriente (A)")
    ax_corriente.set_xlabel("Tiempo (s)")
    ax_corriente.set_title(f"{titulo} - Corriente del estator")

    if metricas and metricas.time is not None:
        for eje, texto in [
            (ax_velocidad, "Tiempo característico"),
            (ax_corriente, "Corriente característica"),
        ]:
            eje.axvline(metricas.time, color="tab:green", linestyle="--", alpha=0.7)
            eje.text(
                metricas.time,
                eje.get_ylim()[1] * 0.9,
                texto,
                rotation=90,
                verticalalignment="bottom",
                horizontalalignment="right",
                fontsize=9,
                backgroundcolor="white",
            )
        if metricas.speed is not None:
            ax_velocidad.scatter(metricas.time, metricas.speed, color="tab:purple", zorder=5)
            ax_velocidad.text(
                metricas.time,
                metricas.speed,
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


def _imprimir_metricas(resultados: SimulationResults, nombre: str) -> None:
    """Muestra en consola las métricas principales de un escenario."""

    metricas = resultados.metrics or {}
    if not metricas:
        print(f"No hay métricas para {nombre}.")
        return

    print(f"\nEscenario: {nombre}")
    for etiqueta, valores in metricas.items():
        tiempo_txt = f"{valores.time:.6f} s" if valores.time is not None else "N/D"
        corriente_txt = f"{valores.current:.6f} A" if valores.current is not None else "N/D"
        velocidad_txt = f"{valores.speed:.6f} rad/s" if valores.speed is not None else "N/D"
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
    graficar_escenario(simulaciones["frenado_desenergizado"], "Frenado por desenergización", "frenado_desenergizado")
    graficar_escenario(
        simulaciones["frenado_contracorriente"], "Frenado por contracorriente", "frenado_contracorriente"
    )


if __name__ == "__main__":
    main()
