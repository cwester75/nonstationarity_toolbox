import numpy as np
from nonlinear.core.maps import LogisticMap
from nonlinear.information.chaotic_modulation import (
    chaotic_carrier,
    modulate,
    demodulate,
    synchronize_maps,
)


def test_chaotic_carrier_length():
    m = LogisticMap(4)
    carrier = chaotic_carrier(m, 0.3, 100)
    assert len(carrier) == 100


def test_modulate_demodulate_roundtrip():
    m = LogisticMap(4)
    carrier = chaotic_carrier(m, 0.3, 20)
    message = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    signal = modulate(message, carrier)
    recovered = demodulate(signal, carrier)
    assert recovered == message


def test_synchronize_converges():
    m = LogisticMap(4)
    errors = synchronize_maps(m, m, 0.3, 0.7, n=200, coupling=0.8)
    # After many steps, errors should decrease
    assert errors[-1] < errors[0]
