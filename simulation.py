import numpy as np
import matplotlib.pyplot as plt


def membrane_drift(V, tau, mu):
    """
    Drift b(V) = -(V - mu) / tau for a leaky integrate-and-fire neuron.
    """
    return -(V - mu) / tau

def membrane_solution(V0, t, tau, mu):
    """
    Exact solution of dV/dt = b(V) with initial condition V(0) = V0.
    """
    return mu + (V0 - mu) * np.exp(-t / tau)


def intensity_f(V, alpha, v_th):
    """
    State-dependent intensity f(V) = alpha * max(V - v_th, 0).
    """
    return alpha * np.maximum(V - v_th, 0.0)


def simulate_if_neuron(
    T=1.0,
    v0=0.0,
    tau=0.02,
    mu=1.2,
    alpha=50.0,
    v_th=1.0,
    A=None,
    dt_record=1e-3,
    seed=None,
):
    """
    Simulate an Integrate-and-Fire neuron driven by a Poisson random measure:

        V_t = v0 + ∫_0^t b(V_s) ds
              - ∫_0^t ∫_R+ V_{s-} 1_{u <= f(V_{s-})} Π(ds, du)

    where
        b(V)  = -(V - mu) / tau
        f(V)  = alpha * max(V - v_th, 0)
        jumps: V -> 0 at spike times.

    Algorithm:
        - Use a dominating rate A >= sup f(V_t) (Poisson thinning).
        - Propose event times with Exp(A).
        - Between events, solve dV/dt = b(V) exactly.
        - At each proposal time s, draw u ~ Uniform(0, A):
            * if u <= f(V_{s-}) -> spike, reset V_s = 0
            * else -> no spike.

    Parameters
    ----------
    T : float
        Final simulation time.
    v0 : float
        Initial membrane potential V_0.
    tau : float
        Membrane time constant.
    mu : float
        Equilibrium/resting potential for the drift.
    alpha : float
        Gain of the intensity function f(V).
    v_th : float
        Threshold parameter inside f(V).
    A : float or None
        Upper bound for f(V). If None, it is chosen automatically.
    dt_record : float
        Time step at which to record V_t for plotting (does NOT affect dynamics).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    t_grid : np.ndarray
        Time grid at which V was recorded.
    V_grid : np.ndarray
        Membrane potential values at times t_grid.
    spike_times : list of float
        Times at which spikes occurred.
    """
    rng = np.random.default_rng(seed)

    # Automatic choice of dominating rate A if not provided
    if A is None:
        f_max = alpha * max(mu - v_th, 0.0)
        A = 1.2 * f_max if f_max > 0 else 1.0  # safety factor

    t = 0.0
    V = v0

    # Recording arrays
    t_grid = [0.0]
    V_grid = [V]

    next_record = dt_record
    spike_times = []

    while t < T:
        # 1. Propose next event time via exponential with rate A
        u = rng.random()
        delta = -np.log(u) / A      # Exp(A)
        t_prop = t + delta

        # If the proposed time exceeds T, just integrate deterministically to T
        if t_prop >= T:
            # Record up to T
            while next_record <= T:
                dt = next_record - t
                V = membrane_solution(V, dt, tau, mu)
                t = next_record
                t_grid.append(t)
                V_grid.append(V)
                next_record += dt_record
            break

        # 2. Integrate between t and t_prop, recording at dt_record steps
        while next_record <= t_prop:
            dt = next_record - t
            V = mu + (V - mu) * np.exp(-dt / tau)
            t = next_record
            t_grid.append(t)
            V_grid.append(V)
            next_record += dt_record

        # Now move exactly from current time t to t_prop (no recording here)
        dt_exact = t_prop - t
        V = mu + (V - mu) * np.exp(-dt_exact / tau)
        t = t_prop

        # 3. Thinning: accept/reject spike
        lam = intensity_f(V, alpha=alpha, v_th=v_th)
        u2 = rng.random() * A  # u2 ~ Uniform(0, A)
        if u2 <= lam:
            # Spike occurs
            spike_times.append(t)
            V = 0.0  # reset after spike

        # Record value right after event if we "cross" a record time
        # (optional; here we just wait until next_record in next loop)

    return np.array(t_grid), np.array(V_grid), spike_times


if __name__ == "__main__":
    # Example usage
    T = 1.0          # total time (s)
    v0 = 0.0
    tau = 0.02       # 20 ms
    mu = 1.2         # equilibrium potential
    alpha = 80.0     # intensity gain
    v_th = 1.0       # threshold inside f(V)
    dt_record = 1e-3

    t_grid, V_grid, spike_times = simulate_if_neuron(
        T=T,
        v0=v0,
        tau=tau,
        mu=mu,
        alpha=alpha,
        v_th=v_th,
        dt_record=dt_record,
        seed=42
    )

    # Plot membrane potential and spikes
    plt.figure()
    plt.plot(t_grid, V_grid, label="V(t)")
    for s in spike_times:
        plt.axvline(s, linestyle="--", alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel("Membrane potential V(t)")
    plt.title("Integrate-and-Fire neuron with Poisson random measure")
    plt.legend()
    plt.tight_layout()
    plt.show()
