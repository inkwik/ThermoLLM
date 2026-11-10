"""Optional derived observations using the RBC's ISO Fanger PMV assumptions."""

from typing import Dict

import numpy as np


class CustomPMVObservations:
    """Append numeric PMV independently of occupancy; consumers apply masking."""

    def __init__(self, variables: Dict[str, str], observation_names: list):
        self.variables = dict(variables)
        if not self.variables:
            return
        if 'month' not in observation_names:
            raise ValueError('Custom PMV requires the month observation.')
        for name, temperature in self.variables.items():
            if name in observation_names:
                raise ValueError(f'Custom PMV observation already exists: {name}')
            if temperature not in observation_names:
                raise ValueError(f'Custom PMV temperature observation missing: {temperature}')
        # Load the dependency only when custom PMV is enabled.
        from pythermalcomfort.models import pmv_ppd_iso

        self.calculate = pmv_ppd_iso

    def augment(self, observations: Dict[str, float]) -> Dict[str, float]:
        if not self.variables:
            return observations
        temperatures = np.asarray(
            [observations[source] for source in self.variables.values()], dtype=float
        )
        pmv = self.calculate(
            tdb=temperatures,
            tr=temperatures,
            vr=0.15,
            rh=40,
            met=1.1,
            clo=0.5 if int(observations['month']) in {6, 7, 8, 9} else 1.0,
            model='7730-2005',
            limit_inputs=False,
            round_output=False,
        ).pmv
        return {**observations, **dict(zip(self.variables, np.asarray(pmv, dtype=float)))}
