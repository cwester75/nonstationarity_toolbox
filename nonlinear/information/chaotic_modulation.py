import numpy as np


def chaotic_carrier(map_obj, x0, n):
    """Generate a chaotic carrier signal from a map."""
    carrier = []
    x = x0
    for _ in range(n):
        x = map_obj.f(x)
        carrier.append(x)
    return np.array(carrier)


def modulate(message_bits, carrier):
    """Additive chaotic modulation: embed binary message in carrier."""
    n = min(len(message_bits), len(carrier))
    signal = carrier[:n].copy()
    amplitude = 0.01
    for i in range(n):
        signal[i] += amplitude * message_bits[i]
    return signal


def demodulate(signal, carrier, threshold=0.005):
    """Recover binary message by subtracting carrier."""
    n = min(len(signal), len(carrier))
    residual = signal[:n] - carrier[:n]
    bits = []
    for r in residual:
        bits.append(1 if r > threshold else 0)
    return bits


def synchronize_maps(driver_map, response_map, x_driver, x_response, n, coupling=0.5):
    """Pecora-Carroll synchronization of two chaotic maps."""
    x_d = x_driver
    x_r = x_response
    errors = []
    for _ in range(n):
        x_d = driver_map.f(x_d)
        x_r = (1 - coupling) * response_map.f(x_r) + coupling * x_d
        errors.append(abs(x_d - x_r))
    return errors
