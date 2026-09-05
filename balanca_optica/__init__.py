"""Balança Óptica - motor de estimativa de peso bovino por visão computacional.

Fluxo: depth/RGB -> segmentação -> geometria métrica -> features -> regressor
-> agregação multi-frame -> intervalo de confiança -> calibração por rebanho.
"""

__version__ = "0.1.0"
