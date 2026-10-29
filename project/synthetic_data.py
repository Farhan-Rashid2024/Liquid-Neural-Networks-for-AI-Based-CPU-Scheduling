import numpy as np
import pandas as pd
import os

def generate_poisson_events(rate, duration, seed=42):
    """Generates a homogeneous Poisson process."""
    np.random.seed(seed)
    # Expected number of events
    n_expected = int(rate * duration)
    # Generate more just in case, then filter
    inter_arrivals = np.random.exponential(1.0 / rate, size=int(n_expected * 1.5))
    arrival_times = np.cumsum(inter_arrivals)
    arrival_times = arrival_times[arrival_times < duration]

    df = pd.DataFrame({
        'timestamp': arrival_times,
        'type': 'poisson'
    })
    return df

def generate_mmpp_events(lambda_low=1.0, lambda_high=10.0,
                         p_low_to_high=0.01, p_high_to_low=0.1,
                         duration=1000, seed=42):
    """
    Generates a Markov-Modulated Poisson Process (MMPP).
    We discretize time into small steps (dt) and simulate the Markov chain.
    """
    np.random.seed(seed)

    # We will simulate the state continuous time Markov chain
    # State 0: low rate, State 1: high rate
    # Transition rates:
    q_01 = p_low_to_high
    q_10 = p_high_to_low

    current_time = 0.0
    state = 0
    arrival_times = []

    while current_time < duration:
        # Time until next state transition
        rate_transition = q_01 if state == 0 else q_10
        t_transition = np.random.exponential(1.0 / rate_transition)

        # Time until next arrival in current state
        lambda_current = lambda_low if state == 0 else lambda_high
        t_arrival = np.random.exponential(1.0 / lambda_current)

        if t_arrival < t_transition:
            # Arrival happens before state transition
            current_time += t_arrival
            arrival_times.append(current_time)
        else:
            # State transition happens before arrival
            current_time += t_transition
            state = 1 - state # Flip state

    # Filter out events beyond duration
    arrival_times = [t for t in arrival_times if t < duration]

    df = pd.DataFrame({
        'timestamp': arrival_times,
        'type': 'bursty_mmpp'
    })
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Generating Poisson dataset...")
    df_poisson = generate_poisson_events(rate=2.0, duration=10000)
    df_poisson.to_csv("data/synthetic_poisson.csv", index=False)

    print("Generating Bursty (MMPP) dataset...")
    df_bursty = generate_mmpp_events(lambda_low=0.5, lambda_high=10.0,
                                     p_low_to_high=0.02, p_high_to_low=0.1,
                                     duration=10000)
    df_bursty.to_csv("data/synthetic_bursty.csv", index=False)

    print(f"Poisson events: {len(df_poisson)}")
    print(f"Bursty events: {len(df_bursty)}")
