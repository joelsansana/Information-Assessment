"""
First-principles white-box model of a biodiesel (transesterification)
reactor.

The model follows the structure of the BDsim benchmark: a well-mixed
jacketed reactor with two inlet streams (oil and methanol) carrying out
the three-step transesterification of triglycerides (TG) into
biodiesel (E) via diglyceride (G) and monoglyceride (M) intermediates.

State vector (length 5):
    x[0] -- triglyceride mole fraction
    x[1] -- monoglyceride mole fraction
    x[2] -- diglyceride mole fraction
    x[3] -- biodiesel (ester) mole fraction
    x[4] -- reactor temperature [K]

Input vector (length 4):
    u[0] -- oil mass flow [kg/h]
    u[1] -- methanol mass flow [kg/h]
    u[2] -- oil inlet temperature [K]
    u[3] -- methanol inlet temperature [K]

The :class:`Reactor` class wraps ODE integration, full-trajectory
simulation and identification of the kinetic parameter(s).
"""

import numpy as np

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error

_INITIAL_STATE = np.array([0.0031, 0.4235, 0.1432, 0.4302, 333.5500])
_DEFAULT_KINETIC_PAR = 0.046


class Reactor:
    """
    Train and simulate the biodiesel reactor model.

    The class exposes three operations:

    * :meth:`predict` -- simulate state trajectories for a given
      parameter and input sequence.
    * :meth:`train` -- identify a single global kinetic parameter by
      least-squares matching of the simulated ester mole fraction
      against measured data.
    * :meth:`dyn_rates` -- identify a time-varying kinetic parameter
      by re-fitting one sample step at a time. Useful for diagnosing
      kinetic drift but rarely used in the main assessment pipeline.
    """

    def ode_eval(self, tspan, par, u, x0):
        """
        Integrate the reactor ODE over a single time span.

        Parameters
        ----------
        tspan : array-like of shape (2,)
            Integration interval ``[t0, t1]``.
        par : float or array-like
            Kinetic parameter(s) consumed by :func:`kinetics`.
        u : array-like of shape (4,)
            Constant input vector on ``tspan``.
        x0 : array-like of shape (5,)
            Initial state.

        Returns
        -------
        sol : ``scipy.integrate.OdeResult``
            Result of the LSODA integration.
        """
        sol = solve_ivp(reactor, t_span=(tspan[0], tspan[-1]), y0=x0, \
        method='LSODA', args=(par, u,), dense_output=False)
        return sol

    def predict(self, time, par, u):
        """
        Simulate the full reactor trajectory for a given input sequence.

        The integrator is initialised with a hard-coded steady-state
        vector and then propagated sample-by-sample so that the state
        at the end of each interval becomes the initial condition for
        the next one.

        Parameters
        ----------
        time : pandas.Series of length ``samples``
            Time stamps in seconds. The underlying ``.values`` are
            used as integration breakpoints.
        par : float or array-like
            Kinetic parameter(s) passed to :func:`kinetics`.
        u : ndarray of shape (4, samples)
            Input matrix -- rows are the four input variables and
            columns correspond to samples.

        Returns
        -------
        svR : ndarray of shape (samples, 5)
            Simulated state trajectory (composition fractions plus
            reactor temperature in K).
        """
        par = np.array(par)
        samples = len(time)
        tspan = time.values
        x0 = _INITIAL_STATE.copy()

        sol = self.ode_eval(tspan, par, u[:, 0], x0)
        x0 = sol.y[:, -1]

        svR = np.array([])
        for i in range(samples-1):
            svR = np.append(svR, x0)
            sol = self.ode_eval(tspan[i:i+2], par, u[:, i], x0)
            x0 = sol.y[:, -1]
        svR = np.append(svR, sol.y[:, -1])
        svR = svR.reshape(samples, len(x0))
        return svR

    def train(self, time, u, xvals, par0=_DEFAULT_KINETIC_PAR):
        """
        Identify a single kinetic parameter by non‑linear least squares.

        The objective matches the simulated ester mole fraction
        (``svR[:, 3]``) against the measured values ``xvals``.

        Parameters
        ----------
        time : pandas.Series
            Time stamps of the measurements.
        u : ndarray of shape (4, samples)
            Input matrix aligned with ``time``.
        xvals : array-like of shape (samples,)
            Measured ester mole fraction.
        par0 : float, default=0.046
            Initial guess for the kinetic parameter.

        Returns
        -------
        par : ndarray
            Identified kinetic parameter(s).
        """
        def _obj(par, time, u, xvals):
            svR = self.predict(time, par, u)
            MSE = mean_squared_error(xvals, svR[:, 3], squared=True)
            return MSE
        solution = least_squares(
            fun=_obj,
            x0=par0,
            method='lm',
            args=(time, u, xvals),
            verbose=False,
            )
        par = solution.x
        print('Reaction rate:', par)
        return par

    def dyn_rates(self, time, u, xvals, par0=_DEFAULT_KINETIC_PAR):
        """
        Identify a sample-by-sample time-varying kinetic parameter.

        For each pair of consecutive samples a new ``par`` is fitted
        using the previously propagated state as the initial
        condition. The result is a per-step estimate of the reaction
        rate that can be used to inspect kinetic drift.

        Parameters
        ----------
        time : pandas.Series
            Time stamps of the measurements.
        u : ndarray of shape (4, samples)
            Input matrix aligned with ``time``.
        xvals : array-like of shape (samples,)
            Measured ester mole fraction.
        par0 : float, default=0.046
            Initial guess for the kinetic parameter at every step.

        Returns
        -------
        rates : ndarray
            Concatenated array of identified kinetic parameters, one
            per integration interval.
        """
        samples = len(time)
        tspan = time.values
        x0 = _INITIAL_STATE.copy()

        def _obj_dyn(par, tspan, u, x0, xvals):
            sol = self.ode_eval(tspan, par, u, x0)
            y_pred = np.append(x0[3], sol.y[3, -1])
            MSE = mean_squared_error(xvals, y_pred, squared=True)
            return MSE

        rates = np.array([])
        for i in range(1, samples):
            solution = least_squares(
            fun=_obj_dyn,
            x0=par0,
            method='lm',
            args=(tspan[i-1:i+1], u[:, i], x0, xvals[i-1:i+1]),
            verbose=False,
            )
            par = solution.x
            rates = np.append(rates, par)
            sol = self.ode_eval(tspan[i-1:i+1], par, u[:, i], x0)
            x0 = sol.y[:, -1]
        return rates


def reactor(t, x, par, u):
    """
    Right-hand side of the biodiesel reactor ODE.

    Implements mass balances for the four pseudo-components
    ``[TG, M, G, E]`` and an energy balance for the reactor
    temperature. Component and physical-property data (molar mass,
    density, heat capacity) are taken from the BDsim benchmark and
    are hard-coded as constants.

    Parameters
    ----------
    t : float
        Time (unused, present for ``solve_ivp`` compatibility).
    x : ndarray of shape (5,)
        Current state vector.
    par : float or array-like
        Kinetic parameter(s) consumed by :func:`kinetics`.
    u : array-like of shape (4,)
        Constant input vector on the current interval.

    Returns
    -------
    dxdt : ndarray of shape (5,)
        Time derivative of the state vector.
    """
    M = np.array([0.853, 0.032, 0.092, 0.286,])
    ro = np.array([954.0, 757.0, 1340.0, 844.0,])
    cp = np.array([2110.0, 2785.0, 2556.0, 2146.0,])
    nc = 4
    Mo = M[0,]
    Mm = M[1,]
    xo = np.array([1, 0, 0, 0,])
    xm = np.array([0, 1, 0, 0,])
    VR = 20.
    vmol = M / ro
    cpmol = cp * M
    cpmolo = cpmol[0]
    cpmolm = cpmol[1]

    dHr = -6309.

    xR = x[:nc].T
    TR = x[nc]

    No = u[0]/Mo/3600
    Nm = u[1]/Mm/3600
    To = u[2]
    Tm = u[3]

    nR = VR / np.sum(vmol * xR)
    cpmolR = np.sum(xR * cpmol)

    r = kinetics(par)

    rx = np.array([-r, -3*r, r, 3*r,]).reshape(-1,)

    dxRdt = (Nm*(xm - xR) + No*(xo - xR) + rx*VR) / nR

    dTRdt = (Nm*cpmolm*(Tm - TR) + No*cpmolo*(To - TR) \
        + VR*np.sum(-dHr * r)) / (nR * cpmolR)

    return np.append(dxRdt, dTRdt)


def kinetics(par):
    """
    Kinetic model.

    Placeholder implementation returning ``par`` unchanged so that the
    identification routines can be exercised end-to-end. Replace with
    a temperature-dependent Arrhenius expression when a richer
    kinetic description is required.
    """
    return par
