"""Implementation of reward functions."""

from datetime import datetime
from math import exp
from typing import Any, Dict, List, Optional, Tuple, Union

from sinergym.utils.constants import LOG_REWARD_LEVEL, YEAR
from sinergym.utils.logger import TerminalLogger


class BaseReward(object):

    logger = TerminalLogger().getLogger(name='REWARD', level=LOG_REWARD_LEVEL)

    def __init__(self):
        """
        Base reward class.

        All reward functions should inherit from this class.

        Args:
            env (Env): Gym environment.
        """

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Method for calculating the reward function."""
        raise NotImplementedError("Reward class must have a `__call__` method.")


class LinearReward(BaseReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
    ):
        """
        Linear reward function.

        It considers the energy consumption and the absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * (max(T - T_{low}, 0) + max(T_{up} - T, 0))

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super().__init__()

        # Basic validations
        if not (0 <= energy_weight <= 1):
            self.logger.error(
                f'energy_weight must be between 0 and 1. Received: {energy_weight}'
            )
            raise ValueError
        if not all(
            isinstance(v, str) for v in temperature_variables + energy_variables
        ):
            self.logger.error('All variable names must be strings.')
            raise TypeError

        # Name of the variables
        self.temp_names = temperature_variables
        self.energy_names = energy_variables

        # Reward parameters
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature

        # Summer period
        self.summer_start = summer_start  # (month, day)
        self.summer_final = summer_final  # (month, day)

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy,
        }

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            List[float]: List with energy consumed in each energy variable.
        """
        return [obs_dict[v] for v in self.energy_names]

    def _get_temperature_violation(self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the temperature violation (ºC) in each observation's temperature variable.

        Returns:
            List[float]: List with temperature violation in each zone.
        """

        # Current datetime and summer period
        current_dt = datetime(
            YEAR, int(obs_dict['month']), int(obs_dict['day_of_month'])
        )
        summer_start_date = datetime(YEAR, *self.summer_start)
        summer_final_date = datetime(YEAR, *self.summer_final)

        temp_range = (
            self.range_comfort_summer
            if summer_start_date <= current_dt <= summer_final_date
            else self.range_comfort_winter
        )

        temp_values = [obs_dict[v] for v in self.temp_names]

        return [max(temp_range[0] - T, 0, T - temp_range[1]) for T in temp_values]

    def _get_reward(self) -> Tuple[float, ...]:
        """Compute the final reward value.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float, ...]: Total reward calculated and reward terms.
        """
        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * (1 - self.W_energy) * self.comfort_penalty
        reward = energy_term + comfort_term
        return reward, energy_term, comfort_term


class TwoZoneDataCenterReward(BaseReward):
    r"""Normalized energy and thermal-safety reward for the two-zone data centre.

    .. math::

       r_t = -\alpha\omega\frac{E_t}{E_{max}}
             -\beta(1-\omega)\frac{\sum_i\Delta T_{i,t}}{T_{max}}

    ``E_t`` is facility HVAC electrical demand in W. ``Delta T`` is the
    distance of each zone air temperature from the configured 20--26 C band.
    """

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        omega: float = 0.5,
        alpha: float = 1.0,
        beta: float = 5.0e-3,
        energy_max: float = 50_000.0,
        temperature_violation_max: float = 6.0,
    ):
        if len(temperature_variables) != 2:
            raise ValueError(
                'TwoZoneDataCenterReward requires East and West temperature variables.'
            )
        if not 0.0 <= omega <= 1.0:
            raise ValueError('omega must be between 0 and 1.')
        if alpha < 0.0 or beta < 0.0 or energy_max <= 0.0 or temperature_violation_max <= 0.0:
            raise ValueError('alpha, beta, energy_max, and temperature_violation_max must be positive.')
        if not all(isinstance(value, str) for value in temperature_variables + energy_variables):
            raise TypeError('All variable names must be strings.')

        super().__init__()
        self.temp_names = temperature_variables
        self.energy_names = energy_variables
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer
        self.summer_start = summer_start
        self.summer_final = summer_final
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.energy_max = energy_max
        self.temperature_violation_max = temperature_violation_max
        self.logger.info('Two-zone data-centre normalized reward initialized.')

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        energy = sum(obs_dict[name] for name in self.energy_names)
        current_dt = datetime(YEAR, int(obs_dict['month']), int(obs_dict['day_of_month']))
        summer_start = datetime(YEAR, *self.summer_start)
        summer_final = datetime(YEAR, *self.summer_final)
        comfort_range = (
            self.range_comfort_summer
            if summer_start <= current_dt <= summer_final
            else self.range_comfort_winter
        )
        violations = [
            max(0.0, obs_dict[name] - comfort_range[1])
            + max(0.0, comfort_range[0] - obs_dict[name])
            for name in self.temp_names
        ]
        total_violation = sum(violations)
        energy_term = -self.alpha * self.omega * (energy / self.energy_max)
        comfort_term = -self.beta * (1.0 - self.omega) * (
            total_violation / self.temperature_violation_max
        )
        reward = energy_term + comfort_term
        return reward, {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': -energy,
            'comfort_penalty': -total_violation,
            'total_power_demand': energy,
            'total_temperature_violation': total_violation,
            # Kept for compatibility with Sinergym loggers that expose the
            # energy/comfort trade-off under this conventional field name.
            'reward_weight': self.omega,
            'tradeoff_weight_omega': self.omega,
            'energy_max_w': self.energy_max,
            'temperature_violation_max_c': self.temperature_violation_max,
        }

# Add this to sinergym/reward_function.py

class TrueReward(BaseReward):
    """
    Occupancy-aware normalized reward for Sinergym.

    Reward formulation:
        R = - (w_energy * E_norm + w_comfort * V_norm)

    where:
        E_norm = clip(HVAC_electricity_demand_rate / hvac_demand_max, 0, 1)
        V_norm = clip(sum(zone_comfort_violations) / total_violation_max, 0, 1)

    Key ideas:
        - HVAC energy is one combined building-level value
        - Comfort violation is summed across all zones
        - Seasonal comfort bands are used
        - Occupied: comfort is prioritized
        - Unoccupied: energy is prioritized
    """

    def __init__(
        self,
        summer_months=(6, 7, 8, 9),
        summer_band=(23.0, 26.0),
        winter_band=(20.0, 23.5),
        max_violation_per_zone=8.0,
        hvac_demand_min=0.0,
        hvac_demand_max=25000.0,
        zone_names=None,
        temperature_variables: Optional[List[str]] = None,
        occupancy_variables: Optional[List[str]] = None,
        occupied_energy_weight: float = 0.1,
        occupied_comfort_weight: float = 0.9,
        unoccupied_energy_weight: float = 0.9,
        unoccupied_comfort_weight: float = 0.1,
    ):
        super().__init__()

        if zone_names is None:
            zone_names = ['SPACE1-1', 'SPACE2-1', 'SPACE3-1', 'SPACE4-1', 'SPACE5-1']
        self.zone_names = zone_names

        if temperature_variables is None:
            temperature_variables = [
                f'air_temperature_space{idx + 1}' for idx in range(len(self.zone_names))
            ]

        if occupancy_variables is None:
            occupancy_variables = [
                f'air_occ_space{idx + 1}' for idx in range(len(self.zone_names))
            ]

        if len(temperature_variables) != len(occupancy_variables):
            self.logger.error(
                'temperature_variables and occupancy_variables must have same length.'
            )
            raise ValueError

        self.temperature_variables = list(temperature_variables)
        self.occupancy_variables = list(occupancy_variables)

        self.summer_months = set(summer_months)
        self.summer_band = tuple(summer_band) if isinstance(summer_band, list) else summer_band
        self.winter_band = tuple(winter_band) if isinstance(winter_band, list) else winter_band

        self.max_violation_per_zone = float(max_violation_per_zone)
        self.total_violation_max = float(len(self.zone_names) * self.max_violation_per_zone)

        self.hvac_demand_min = float(hvac_demand_min)
        self.hvac_demand_max = float(hvac_demand_max)

        self.occupied_energy_weight = float(occupied_energy_weight)
        self.occupied_comfort_weight = float(occupied_comfort_weight)
        self.unoccupied_energy_weight = float(unoccupied_energy_weight)
        self.unoccupied_comfort_weight = float(unoccupied_comfort_weight)

        if abs((self.occupied_energy_weight + self.occupied_comfort_weight) - 1.0) > 1e-8:
            self.logger.error(
                'occupied_energy_weight + occupied_comfort_weight must equal 1.0'
            )
            raise ValueError

        if abs((self.unoccupied_energy_weight + self.unoccupied_comfort_weight) - 1.0) > 1e-8:
            self.logger.error(
                'unoccupied_energy_weight + unoccupied_comfort_weight must equal 1.0'
            )
            raise ValueError

        if self.hvac_demand_max <= self.hvac_demand_min:
            self.logger.error('hvac_demand_max must be greater than hvac_demand_min')
            raise ValueError

        if self.max_violation_per_zone <= 0:
            self.logger.error('max_violation_per_zone must be > 0')
            raise ValueError

        self.logger.info('TrueReward function initialized.')

    def _get_comfort_band(self, month: int) -> Tuple[float, float]:
        if int(month) in self.summer_months:
            return self.summer_band
        return self.winter_band

    @staticmethod
    def _clip_0_1(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        if max_val <= min_val:
            return 0.0
        norm = (value - min_val) / (max_val - min_val)
        return self._clip_0_1(norm)

    @staticmethod
    def _zone_violation(temp: float, t_min: float, t_max: float) -> float:
        if temp < t_min:
            return t_min - temp
        elif temp > t_max:
            return temp - t_max
        return 0.0

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        month = int(obs_dict.get('month', 1))
        day = int(obs_dict.get('day_of_month', 1))
        hour = int(obs_dict.get('hour', 0))

        t_min, t_max = self._get_comfort_band(month)

        zone_temperatures = []
        zone_occupancies = []
        zone_violations = []
        zone_violation_norms = []

        # Per-zone temperature and comfort violation
        for temp_key, occ_key in zip(self.temperature_variables, self.occupancy_variables):
            temp = float(obs_dict.get(temp_key, t_min))
            occ = float(obs_dict.get(occ_key, 0.0))

            violation = self._zone_violation(temp, t_min, t_max)
            violation = min(violation, self.max_violation_per_zone)  # cap per zone
            violation_norm = self._normalize(violation, 0.0, self.max_violation_per_zone)

            zone_temperatures.append(temp)
            zone_occupancies.append(occ)
            zone_violations.append(violation)
            zone_violation_norms.append(violation_norm)

        # Whole-building occupancy
        is_occupied = any(occ > 0 for occ in zone_occupancies)

        # Whole-building HVAC demand (single combined value)
        hvac_demand = float(
            obs_dict.get(
                'HVAC_electricity_demand_rate',
                obs_dict.get('hvac_electricity_demand_rate', self.hvac_demand_min)
            )
        )

        # Raw totals
        total_violation = sum(zone_violations)

        # Normalized totals
        energy_norm = self._normalize(
            hvac_demand, self.hvac_demand_min, self.hvac_demand_max
        )
        comfort_norm = self._normalize(
            total_violation, 0.0, self.total_violation_max
        )

        # Weight selection
        if is_occupied:
            w_energy = self.occupied_energy_weight
            w_comfort = self.occupied_comfort_weight
        else:
            w_energy = self.unoccupied_energy_weight
            w_comfort = self.unoccupied_comfort_weight

        # Unweighted penalty parts
        unweighted_energy_penalty = -energy_norm
        unweighted_comfort_penalty = -comfort_norm

        # Weighted reward parts
        energy_term = -w_energy * energy_norm
        comfort_term = -w_comfort * comfort_norm

        total_reward = energy_term + comfort_term

        # Bias / dominance tracking
        abs_energy_term = abs(energy_term)
        abs_comfort_term = abs(comfort_term)
        total_abs_term = abs_energy_term + abs_comfort_term

        energy_term_share = abs_energy_term / total_abs_term if total_abs_term > 0 else 0.0
        comfort_term_share = abs_comfort_term / total_abs_term if total_abs_term > 0 else 0.0

        dominant_term = (
            'energy'
            if abs_energy_term > abs_comfort_term
            else 'comfort'
            if abs_comfort_term > abs_energy_term
            else 'equal'
        )

        reward_terms = {
            # ===== Main reward outputs =====
            'reward_total': total_reward,
            'energy_term': energy_term,
            'comfort_term': comfort_term,

            # ===== Compatibility keys =====
            # Keep these names because wrappers/loggers may expect them
            'energy_penalty': unweighted_energy_penalty,
            'comfort_penalty': unweighted_comfort_penalty,
            'total_power_demand': hvac_demand,
            'total_temperature_violation': total_violation,

            # ===== Raw metrics =====
            'hvac_demand_raw': hvac_demand,
            'total_electricity_HVAC': float(obs_dict.get('total_electricity_HVAC', 0.0)),
            'comfort_violation_raw': total_violation,

            # ===== Normalized metrics =====
            'energy_norm': energy_norm,
            'comfort_norm': comfort_norm,

            # ===== Weights used this step =====
            'w_energy': w_energy,
            'w_comfort': w_comfort,
            'is_occupied': int(is_occupied),
            'num_occupied_zones': sum(1 for occ in zone_occupancies if occ > 0),

            # ===== Bias / dominance tracking =====
            'abs_energy_term': abs_energy_term,
            'abs_comfort_term': abs_comfort_term,
            'energy_term_share': energy_term_share,
            'comfort_term_share': comfort_term_share,
            'dominant_term_energy': 1 if dominant_term == 'energy' else 0,
            'dominant_term_comfort': 1 if dominant_term == 'comfort' else 0,
            'dominant_term_equal': 1 if dominant_term == 'equal' else 0,

            # ===== Unweighted contribution tracking =====
            'unweighted_energy_penalty': unweighted_energy_penalty,
            'unweighted_comfort_penalty': unweighted_comfort_penalty,

            # ===== Time / seasonal context =====
            'month': month,
            'day': day,
            'hour': hour,
            'comfort_band_low': float(t_min),
            'comfort_band_high': float(t_max),

            # ===== Zone summary stats =====
            'mean_zone_temp': sum(zone_temperatures) / len(zone_temperatures) if zone_temperatures else 0.0,
            'max_zone_temp': max(zone_temperatures) if zone_temperatures else 0.0,
            'min_zone_temp': min(zone_temperatures) if zone_temperatures else 0.0,
            'mean_temp_violation': sum(zone_violations) / len(zone_violations) if zone_violations else 0.0,
            'max_temp_violation': max(zone_violations) if zone_violations else 0.0,
            'mean_temp_violation_norm': sum(zone_violation_norms) / len(zone_violation_norms) if zone_violation_norms else 0.0,
            'max_temp_violation_norm': max(zone_violation_norms) if zone_violation_norms else 0.0,
        }

        # Optional per-zone tracking
        for i, (temp, occ, violation, violation_norm) in enumerate(
            zip(zone_temperatures, zone_occupancies, zone_violations, zone_violation_norms),
            start=1
        ):
            reward_terms[f'zone_{i}_temp'] = temp
            reward_terms[f'zone_{i}_occupancy'] = occ
            reward_terms[f'zone_{i}_violation'] = violation
            reward_terms[f'zone_{i}_violation_norm'] = violation_norm

        return total_reward, reward_terms

class MyReward(BaseReward):
    """
    Custom reward function for Sinergym:
    - Penalizes temperature violations per zone (seasonal comfort bands)
    - Penalizes total HVAC electricity demand rate (Watt)
    - Occupancy-aware weighting (rho)
    - All normalization is [0, 1]
    """

    def __init__(self,
                 summer_months=(6, 7, 8, 9),
                 summer_band=(23.0, 26.0),
                 winter_band=(20.0, 23.5),
                 temp_violation_min=0.0,
                 temp_violation_max=8.0,
                 hvac_demand_min=0.0,
                 hvac_demand_max=15000.0,
                 zone_names=None,
                 temperature_variables: Optional[List[str]] = None,
                 occupancy_variables: Optional[List[str]] = None,
                 rho_occupied: float = 1.1,
                 rho_unoccupied: float = 0.1):
        # Zone names as per your environment config
        if zone_names is None:
            # Default typical 5-zone naming used in 5ZoneAutoDXVAV zonal configs
            zone_names = ['SPACE1-1', 'SPACE2-1', 'SPACE3-1', 'SPACE4-1', 'SPACE5-1']
        self.zone_names = zone_names
        # Explicit variable names for temperature and occupancy signals
        if temperature_variables is None:
            temperature_variables = [
                f"air_temperature_space{idx + 1}" for idx in range(len(self.zone_names))
            ]
        if occupancy_variables is None:
            occupancy_variables = [
                f"air_occ_space{idx + 1}" for idx in range(len(self.zone_names))
            ]
        if len(temperature_variables) != len(occupancy_variables):
            raise ValueError("temperature_variables and occupancy_variables must have same length")
        self.temperature_variables = list(temperature_variables)
        self.occupancy_variables = list(occupancy_variables)
        self.summer_months = set(summer_months)
        # Ensure bands are tuples (YAML passes them as lists)
        self.summer_band = tuple(summer_band) if isinstance(summer_band, list) else summer_band
        self.winter_band = tuple(winter_band) if isinstance(winter_band, list) else winter_band
        self.temp_violation_min = float(temp_violation_min)
        self.temp_violation_max = float(temp_violation_max)
        self.hvac_demand_min = float(hvac_demand_min)
        self.hvac_demand_max = float(hvac_demand_max)
        self.rho_occupied = float(rho_occupied)
        self.rho_unoccupied = float(rho_unoccupied)

    def _get_comfort_band(self, month):
        if int(month) in self.summer_months:
            return self.summer_band
        else:
            return self.winter_band

    def _normalize(self, value, min_val, max_val):
        norm = (value - min_val) / (max_val - min_val)
        return min(max(norm, 0.0), 1.0)

    def __call__(self, observation, reward_info=None):
        # Get current month for comfort band
        month = observation.get('month', 1)
        Tmin, Tmax = self._get_comfort_band(month)

        total_reward = 0.0
        zone_terms = []
        rho_terms = []
        temp_norm_terms = []
        temp_violation_terms = []

        temp_keys = list(self.temperature_variables)
        occ_keys = list(self.occupancy_variables)

        # Zone-wise temperature violation and occupancy
        for temp_key, occ_key in zip(temp_keys, occ_keys):

            T = observation.get(temp_key, Tmin)  # fallback to Tmin if missing
            occupancy = observation.get(occ_key, 0)

            # Temperature violation
            temp_violation = max(Tmin - T, 0, T - Tmax)
            norm_temp_violation = self._normalize(temp_violation, self.temp_violation_min, self.temp_violation_max)

            # Occupancy weighting
            rho = self.rho_occupied if occupancy and float(occupancy) > 0 else self.rho_unoccupied
            temp_violation_terms.append(temp_violation)
            temp_norm_terms.append(norm_temp_violation)
            rho_terms.append(rho)
            zone_terms.append(rho * norm_temp_violation)
            total_reward -= rho * norm_temp_violation

        # HVAC electricity demand rate (cumulative for all zones)
        # Note: observation key name is case sensitive in Sinergym
        hvac_demand = observation.get('HVAC_electricity_demand_rate',
                                      observation.get('hvac_electricity_demand_rate', self.hvac_demand_min))
        norm_hvac_demand = self._normalize(hvac_demand, self.hvac_demand_min, self.hvac_demand_max)

        total_reward -= norm_hvac_demand
    
    
    
        # Calculate compatibility terms for wrappers (they expect these keys)
        comfort_penalty_sum = sum(zone_terms)  # Raw comfort penalty (without negative sign)
        comfort_term = -comfort_penalty_sum     # Comfort term for reward
        energy_term = -norm_hvac_demand         # Energy term for reward
        
        reward_terms = {
            # Required keys for LoggerWrapper/WandBLogger compatibility
            'comfort_term': comfort_term,
            'energy_term': energy_term,
            'comfort_penalty': -comfort_penalty_sum,  # Negative of sum
            'energy_penalty': -hvac_demand,           # Raw HVAC demand (negative)
            'total_temperature_violation': sum(temp_violation_terms),
            'total_power_demand': hvac_demand,
            
            # Your custom scalar metrics (safe for WandB logging)
            'comfort_component': comfort_term,
            'hvac_norm': norm_hvac_demand,
            'hvac_demand_raw': hvac_demand,
            'month': int(month),
            'day': int(observation.get('day_of_month', 1)),
            'hour': int(observation.get('hour', 0)),
            'total_electricity_HVAC': float(observation.get('total_electricity_HVAC', 0.0)),
            
            # Zone-wise statistics (scalars derived from lists)
            'mean_zone_penalty': sum(zone_terms) / len(zone_terms) if zone_terms else 0.0,
            'max_zone_penalty': max(zone_terms) if zone_terms else 0.0,
            'mean_temp_violation': sum(temp_violation_terms) / len(temp_violation_terms) if temp_violation_terms else 0.0,
            'max_temp_violation': max(temp_violation_terms) if temp_violation_terms else 0.0,
            'mean_rho': sum(rho_terms) / len(rho_terms) if rho_terms else 0.0,
            'num_occupied_zones': sum(1 for r in rho_terms if r == self.rho_occupied),
        }

        return total_reward, reward_terms


class MyPMV(BaseReward):
    """
    Paper-style PMV reward adapted for the 5-zone setup.

    Reward formulation:
        R = - (sum_i rho_i * Norm(|PMV_i|) + Norm(E_HVAC_kWh))

    Notes:
        - PMV is still evaluated per zone.
        - HVAC energy is a single building-level meter, so the energy term is added
          once per timestep, not once per zone.
        - rho is occupancy-aware: occupied zones get higher comfort weight.
    """

    def __init__(self,
                 pmv_min=0.0,
                 pmv_max=4.0,
                 energy_kwh_min=0.0,
                 energy_kwh_max=4.5,
                 zone_names=None,
                 pmv_variables: Optional[List[str]] = None,
                 occupancy_variables: Optional[List[str]] = None,
                 energy_variable: str = 'total_electricity_HVAC',
                 rho_occupied: float = 1.1,
                 rho_unoccupied: float = 0.1):
        super().__init__()
        if zone_names is None:
            zone_names = ['SPACE1-1', 'SPACE2-1', 'SPACE3-1', 'SPACE4-1', 'SPACE5-1']
        self.zone_names = zone_names

        if pmv_variables is None:
            pmv_variables = [
                f"pmv_space{idx + 1}" for idx in range(len(self.zone_names))
            ]
        if occupancy_variables is None:
            occupancy_variables = [
                f"air_occ_space{idx + 1}" for idx in range(len(self.zone_names))
            ]
        if len(pmv_variables) != len(occupancy_variables):
            raise ValueError("pmv_variables and occupancy_variables must have same length")

        self.pmv_variables = list(pmv_variables)
        self.occupancy_variables = list(occupancy_variables)
        self.energy_variable = str(energy_variable)
        self.pmv_min = float(pmv_min)
        self.pmv_max = float(pmv_max)
        self.energy_kwh_min = float(energy_kwh_min)
        self.energy_kwh_max = float(energy_kwh_max)
        self.rho_occupied = float(rho_occupied)
        self.rho_unoccupied = float(rho_unoccupied)

    def _normalize(self, value, min_val, max_val):
        if max_val <= min_val:
            raise ValueError("max_val must be greater than min_val for normalization")
        norm = (value - min_val) / (max_val - min_val)
        return min(max(norm, 0.0), 1.0)

    def _get_pmv_penalty_value(self, pmv):
        return abs(pmv)

    def __call__(self, observation, reward_info=None):
        total_reward = 0.0
        zone_terms = []
        rho_terms = []
        pmv_abs_terms = []
        pmv_norm_terms = []
        pmv_raw_terms = []

        pmv_keys = list(self.pmv_variables)
        occ_keys = list(self.occupancy_variables)

        for pmv_key, occ_key in zip(pmv_keys, occ_keys):
            pmv = float(observation.get(pmv_key, 0.0))
            occupancy = observation.get(occ_key, 0)
            abs_pmv = abs(pmv)
            pmv_penalty_value = self._get_pmv_penalty_value(pmv)
            norm_abs_pmv = self._normalize(pmv_penalty_value, self.pmv_min, self.pmv_max)

            rho = self.rho_occupied if occupancy and float(occupancy) > 0 else self.rho_unoccupied
            pmv_raw_terms.append(pmv)
            pmv_abs_terms.append(abs_pmv)
            pmv_norm_terms.append(norm_abs_pmv)
            rho_terms.append(rho)
            zone_terms.append(rho * norm_abs_pmv)
            total_reward -= rho * norm_abs_pmv

        hvac_energy_joules = float(observation.get(self.energy_variable, 0.0))
        hvac_energy_kwh = hvac_energy_joules / 3.6e6
        norm_hvac_energy = self._normalize(
            hvac_energy_kwh, self.energy_kwh_min, self.energy_kwh_max
        )

        total_reward -= norm_hvac_energy

        comfort_penalty_sum = sum(zone_terms)
        comfort_term = -comfort_penalty_sum
        energy_term = -norm_hvac_energy

        reward_terms = {
            'comfort_term': comfort_term,
            'energy_term': energy_term,
            'comfort_penalty': -comfort_penalty_sum,
            'energy_penalty': -hvac_energy_kwh,
            'total_temperature_violation': sum(pmv_abs_terms),
            'total_pmv_abs': sum(pmv_abs_terms),
            'total_power_demand': hvac_energy_kwh,
            'total_energy_kwh': hvac_energy_kwh,
            'comfort_component': comfort_term,
            'energy_norm': norm_hvac_energy,
            'total_hvac_energy_joules': hvac_energy_joules,
            'total_hvac_energy_kwh': hvac_energy_kwh,
            'month': int(observation.get('month', 1)),
            'day': int(observation.get('day_of_month', 1)),
            'hour': int(observation.get('hour', 0)),
            'total_electricity_HVAC': hvac_energy_joules,
            'pmv_abs_min': float(self.pmv_min),
            'pmv_abs_max': float(self.pmv_max),
            'energy_kwh_min': float(self.energy_kwh_min),
            'energy_kwh_max': float(self.energy_kwh_max),
            'mean_zone_penalty': sum(zone_terms) / len(zone_terms) if zone_terms else 0.0,
            'max_zone_penalty': max(zone_terms) if zone_terms else 0.0,
            'mean_pmv': sum(pmv_raw_terms) / len(pmv_raw_terms) if pmv_raw_terms else 0.0,
            'max_pmv': max(pmv_raw_terms) if pmv_raw_terms else 0.0,
            'min_pmv': min(pmv_raw_terms) if pmv_raw_terms else 0.0,
            'mean_abs_pmv': sum(pmv_abs_terms) / len(pmv_abs_terms) if pmv_abs_terms else 0.0,
            'max_abs_pmv': max(pmv_abs_terms) if pmv_abs_terms else 0.0,
            'mean_abs_pmv_norm': sum(pmv_norm_terms) / len(pmv_norm_terms) if pmv_norm_terms else 0.0,
            'max_abs_pmv_norm': max(pmv_norm_terms) if pmv_norm_terms else 0.0,
            'mean_rho': sum(rho_terms) / len(rho_terms) if rho_terms else 0.0,
            'num_occupied_zones': sum(1 for r in rho_terms if r == self.rho_occupied),
        }

        for i, (pmv, occupancy, abs_pmv, abs_pmv_norm) in enumerate(
            zip(pmv_raw_terms, occ_keys, pmv_abs_terms, pmv_norm_terms),
            start=1
        ):
            reward_terms[f'zone_{i}_pmv'] = pmv
            reward_terms[f'zone_{i}_abs_pmv'] = abs_pmv
            reward_terms[f'zone_{i}_abs_pmv_norm'] = abs_pmv_norm
            reward_terms[f'zone_{i}_occupancy'] = float(observation.get(occupancy, 0.0))

        return total_reward, reward_terms


class TruePMV(MyPMV):
    """
    MyPMV variant that penalizes only PMV violation outside the [-0.5, 0.5] band.

    Uses custom ISO Fanger observations (new_pmv_space1..5) by default.
    Explicit pmv_variables can select another observation source.
    Energy penalty, occupancy weights, normalization bounds, and reward terms are
    inherited from MyPMV.
    """

    def __init__(self,
                 pmv_min=0.0,
                 pmv_max=4.0,
                 energy_kwh_min=0.0,
                 energy_kwh_max=4.5,
                 zone_names=None,
                 pmv_variables: Optional[List[str]] = None,
                 occupancy_variables: Optional[List[str]] = None,
                 energy_variable: str = 'total_electricity_HVAC',
                 rho_occupied: float = 1.1,
                 rho_unoccupied: float = 0.1):
        if pmv_variables is None:
            zone_count = len(zone_names) if zone_names is not None else 5
            pmv_variables = [f'new_pmv_space{i}' for i in range(1, zone_count + 1)]
        super().__init__(
            pmv_min=pmv_min,
            pmv_max=pmv_max,
            energy_kwh_min=energy_kwh_min,
            energy_kwh_max=energy_kwh_max,
            zone_names=zone_names,
            pmv_variables=pmv_variables,
            occupancy_variables=occupancy_variables,
            energy_variable=energy_variable,
            rho_occupied=rho_occupied,
            rho_unoccupied=rho_unoccupied,
        )

    def _get_pmv_penalty_value(self, pmv):
        return max(abs(pmv) - 0.5, 0.0)


class OccupancyAwareLinearReward(BaseReward):

    def __init__(
        self,
        temperature_variables: List[str],
        heating_setpoint_variables: List[str],
        cooling_setpoint_variables: List[str],
        occupancy_variables: List[str],
        energy_variable: str,
        temperature_violation_range: Tuple[float, float] = (0.0, 5.0),
        power_range: Tuple[float, float] = (0.0, 11000.0),
        occupied_weight: float = 1.1,
        unoccupied_weight: float = 0.1,
        comfort_weight: float = 1.0,
        energy_weight: float = 1.0,
    ):
        """Reward combining per-zone comfort penalties with HVAC power usage.

        Args:
            temperature_variables (List[str]): Zone air temperature variable names.
            heating_setpoint_variables (List[str]): Zone heating setpoint variable names.
            cooling_setpoint_variables (List[str]): Zone cooling setpoint variable names.
            occupancy_variables (List[str]): Zone occupancy variable names.
            energy_variable (str): Name of the HVAC power variable.
            temperature_violation_range (Tuple[float, float], optional): Min/max expected temperature violation used for normalization. Defaults to (0.0, 5.0).
            power_range (Tuple[float, float], optional): Min/max expected HVAC power used for normalization. Defaults to (0.0, 11000.0).
            occupied_weight (float, optional): Comfort weight multiplier when the zone is occupied. Defaults to 4.0.
            unoccupied_weight (float, optional): Comfort weight multiplier when the zone is empty. Defaults to 0.1.
            comfort_weight (float, optional): Weight applied to the aggregated comfort penalty. Defaults to 1.0.
            energy_weight (float, optional): Weight applied to the energy penalty. Defaults to 1.0.
        """

        super().__init__()

        sizes = {
            len(temperature_variables),
            len(heating_setpoint_variables),
            len(cooling_setpoint_variables),
            len(occupancy_variables),
        }
        if len(sizes) != 1:
            self.logger.error('Zone-related variable lists must have the same length.')
            raise ValueError
        if energy_weight < 0 or comfort_weight < 0:
            self.logger.error('comfort_weight and energy_weight must be non-negative.')
            raise ValueError
        if power_range[0] >= power_range[1]:
            self.logger.error('power_range must define a valid min and max.')
            raise ValueError
        if temperature_violation_range[0] >= temperature_violation_range[1]:
            self.logger.error(
                'temperature_violation_range must define a valid min and max.'
            )
            raise ValueError

        self.temp_names = temperature_variables
        self.htg_names = heating_setpoint_variables
        self.clg_names = cooling_setpoint_variables
        self.occ_names = occupancy_variables
        self.energy_name = energy_variable

        self.temp_violation_min, self.temp_violation_max = temperature_violation_range
        self.power_min, self.power_max = power_range
        self.occupied_weight = occupied_weight
        self.unoccupied_weight = unoccupied_weight
        self.comfort_weight = comfort_weight
        self.energy_weight = energy_weight

        self.logger.info('Occupancy-aware reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        zone_penalties: List[float] = []

        for temp_name, htg_name, clg_name, occ_name in zip(
            self.temp_names, self.htg_names, self.clg_names, self.occ_names
        ):
            temperature = obs_dict[temp_name]
            heating_sp = obs_dict[htg_name]
            cooling_sp = obs_dict[clg_name]
            occupancy = obs_dict[occ_name]

            violation = 0.0
            if temperature < heating_sp:
                violation = heating_sp - temperature
            elif temperature > cooling_sp:
                violation = temperature - cooling_sp

            violation_norm = self._min_max_norm(
                violation, self.temp_violation_min, self.temp_violation_max
            )
            weight = (
                self.occupied_weight if occupancy > 0 else self.unoccupied_weight
            )
            zone_penalties.append(weight * violation_norm)

        comfort_component = (
            sum(zone_penalties) / len(zone_penalties) if zone_penalties else 0.0
        )

        power = obs_dict[self.energy_name]
        energy_component = self._min_max_norm(power, self.power_min, self.power_max)

        total_penalty = (
            self.energy_weight * energy_component
            + self.comfort_weight * comfort_component
        )
        reward = -total_penalty

        reward_terms = {
            'energy_norm': energy_component,
            'comfort_norm': comfort_component,
            'zone_penalties': zone_penalties,
            'power_raw': power,
            'total_penalty': total_penalty,
        }

        return reward, reward_terms

    @staticmethod
    def _min_max_norm(value: float, min_value: float, max_value: float) -> float:
        if max_value == min_value:
            return 0.0
        normalized = (value - min_value) / (max_value - min_value)
        return min(max(normalized, 0.0), 1.0)


class EnergyCostLinearReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        energy_cost_variables: List[str],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.4,
        temperature_weight: float = 0.4,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
        lambda_energy_cost: float = 1.0,
    ):
        """
        Linear reward function with the addition of the energy cost term.

        Considers energy consumption, absolute difference to thermal comfort and energy cost.

        .. math::
            R = - W_E * lambda_E * power - W_T * lambda_T * (max(T - T_{low}, 0) + max(T_{up} - T, 0)) - (1 - W_P - W_T) * lambda_EC * power_cost

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.4.
            temperature_weight (float, optional): Weight given to the temperature term. Defaults to 0.4.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1.0.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
            lambda_energy_cost (float, optional): Constant for removing dimensions from temperature(1/E). Defaults to 1.0.
        """

        super().__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight,
            lambda_energy,
            lambda_temperature,
        )

        self.energy_cost_names = energy_cost_variables
        self.W_temperature = temperature_weight
        self.lambda_energy_cost = lambda_energy_cost

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Energy cost calculation
        energy_cost_values = self._get_money_spent(obs_dict)
        self.total_energy_cost = sum(energy_cost_values)
        self.energy_cost_penalty = -self.total_energy_cost

        # Weighted sum of terms
        reward, energy_term, comfort_term, energy_cost_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_cost_term': energy_cost_term,
            'reward_energy_weight': self.W_energy,
            'reward_temperature_weight': self.W_temperature,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'energy_cost_penalty': self.energy_cost_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'money_spent': self.total_energy_cost,
        }

        return reward, reward_terms

    def _get_money_spent(self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the total money spent in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            List[float]: List with money spent in each energy cost variable.
        """
        return [v for k, v in obs_dict.items() if k in self.energy_cost_names]

    def _get_reward(self) -> Tuple[float, ...]:
        """It calculates reward value using the negative absolute comfort, energy penalty and energy cost penalty calculates previously.

        Returns:
            Tuple[float, ...]: Total reward calculated, reward term for energy, reward term for comfort and reward term for energy cost.
        """
        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * self.W_temperature * self.comfort_penalty
        energy_cost_term = (
            self.lambda_energy_cost
            * (1 - self.W_energy - self.W_temperature)
            * self.energy_cost_penalty
        )

        reward = energy_term + comfort_term + energy_cost_term
        return reward, energy_term, comfort_term, energy_cost_term


class ExpReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
    ):
        """
        Reward considering exponential absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * exp( (max(T - T_{low}, 0) + max(T_{up} - T, 0)) )

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super().__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight,
            lambda_energy,
            lambda_temperature,
        )

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        # Exponential Penalty
        self.comfort_penalty = -sum(
            exp(violation) for violation in temp_violations if violation > 0
        )

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy,
        }

        return reward, reward_terms


class HourlyLinearReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        default_energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
        range_comfort_hours: tuple = (9, 19),
    ):
        """
        Linear reward function with a time-dependent weight for consumption and energy terms.

        Args:
            temperature_variables (List[str]]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            default_energy_weight (float, optional): Default weight given to the energy term when thermal comfort is considered. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
            range_comfort_hours (tuple, optional): Hours where thermal comfort is considered. Defaults to (9, 19).
        """

        super(HourlyLinearReward, self).__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            default_energy_weight,
            lambda_energy,
            lambda_temperature,
        )

        # Reward parameters
        self.range_comfort_hours = range_comfort_hours
        self.default_energy_weight = default_energy_weight

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Determine reward weight depending on the hour
        self.W_energy = (
            self.default_energy_weight
            if self.range_comfort_hours[0]
            <= obs_dict['hour']
            <= self.range_comfort_hours[1]
            else 1.0
        )

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy,
        }

        return reward, reward_terms


class NormalizedLinearReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        max_energy_penalty: float = 8,
        max_comfort_penalty: float = 12,
    ):
        """
        Linear reward function with a time-dependent weight for consumption and energy terms.

        Args:
            temperature_variables (List[str]]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Default weight given to the energy term when thermal comfort is considered. Defaults to 0.5.
            max_energy_penalty (float, optional): Maximum energy penalty value. Defaults to 8.
            max_comfort_penalty (float, optional): Maximum comfort penalty value. Defaults to 12.
        """

        super().__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight,
        )

        # Reward parameters
        self.max_energy_penalty = max_energy_penalty
        self.max_comfort_penalty = max_comfort_penalty

    def _get_reward(self) -> Tuple[float, ...]:
        """It calculates reward value using energy consumption and grades of temperature out of comfort range. Applying normalization

        Returns:
            Tuple[float, ...]: total reward calculated, reward term for energy and reward term for comfort.
        """
        # Update max energy and comfort
        self.max_energy_penalty = max(self.max_energy_penalty, self.energy_penalty)
        self.max_comfort_penalty = max(self.max_comfort_penalty, self.comfort_penalty)

        # Calculate normalization
        energy_norm = (
            self.energy_penalty / self.max_energy_penalty
            if self.max_energy_penalty
            else 0
        )
        comfort_norm = (
            self.comfort_penalty / self.max_comfort_penalty
            if self.max_comfort_penalty
            else 0
        )

        # Calculate reward terms with norm values
        energy_term = self.W_energy * energy_norm
        comfort_term = (1 - self.W_energy) * comfort_norm
        reward = energy_term + comfort_term

        return reward, energy_term, comfort_term


class MultiZoneReward(BaseReward):

    def __init__(
        self,
        energy_variables: List[str],
        temperature_and_setpoints_conf: Dict[str, str],
        comfort_threshold: float = 0.5,
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
    ):
        """
        A linear reward function for environments with different comfort ranges in each zone. Instead of having
        a fixed and common comfort range for the entire building, each zone has its own comfort range, which is
        directly obtained from the setpoints established in the building. This function is designed for buildings
        where temperature setpoints are not controlled directly but rather used as targets to be achieved, while
        other actuators are controlled to reach these setpoints. A setpoint observation variable can be assigned
        per zone if it is available in the specific building. It is also possible to assign the same setpoint
        variable to multiple air temperature zones.

        Args:
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            temperature_and_setpoints_conf (Dict[str, str]): Dictionary with the temperature variable name (key) and the setpoint variable name (value) of the observation space.
            comfort_threshold (float, optional): Comfort threshold for temperature range (+/-). Defaults to 0.5.
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super().__init__()

        # Name of the variables
        self.energy_names = energy_variables
        self.comfort_configuration = temperature_and_setpoints_conf
        self.comfort_threshold = comfort_threshold

        # Reward parameters
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature
        self.comfort_ranges = {}

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy,
            'comfort_threshold': self.comfort_threshold,
        }

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            List[float]: List with energy consumed in each energy variable.
        """
        return [obs_dict[v] for v in self.energy_names]

    def _get_temperature_violation(self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the total temperature violation (ºC) in the current observation.

        Returns:
           List[float]: List with temperature violation (ºC) in each zone.
        """
        # Calculate current comfort range for each zone
        self._get_comfort_ranges(obs_dict)

        temp_violations = [
            (
                max(0.0, min(abs(T - comfort_range[0]), abs(T - comfort_range[1])))
                if T < comfort_range[0] or T > comfort_range[1]
                else 0.0
            )
            for temp_var, comfort_range in self.comfort_ranges.items()
            if (T := obs_dict[temp_var])
        ]

        return temp_violations

    def _get_comfort_ranges(self, obs_dict: Dict[str, Any]):
        """Calculate the comfort range for each zone in the current observation.

        Returns:
            Dict[str, Tuple[float, float]]: Comfort range for each zone.
        """
        # Calculate current comfort range for each zone
        self.comfort_ranges = {
            temp_var: (
                setpoint - self.comfort_threshold,
                setpoint + self.comfort_threshold,
            )
            for temp_var, setpoint_var in self.comfort_configuration.items()
            if (setpoint := obs_dict[setpoint_var]) is not None
        }

    def _get_reward(self) -> Tuple[float, ...]:
        """Compute the final reward value.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float, ...]: Total reward calculated and reward terms.
        """
        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * (1 - self.W_energy) * self.comfort_penalty
        reward = energy_term + comfort_term
        return reward, energy_term, comfort_term
