import numpy as np
import pandas as pd


SEED = 202
N_ROWS = 7000
N_SERVERS = 12

OUTPUT_FILE = "independent_test_set_v2.csv"

rng = np.random.default_rng(SEED)


roles = [
    "Web Server",
    "Database Server",
    "App Server"
]

servers = []

for i in range(1, N_SERVERS + 1):
    server_id = f"SRV-{i:03d}"
    hostname = f"server-{i:03d}"

    role = roles[(i - 1) % len(roles)]

    servers.append({
        "server_id": server_id,
        "hostname": hostname,
        "role": role
    })


timestamps = pd.date_range(
    start="2026-01-01 00:00:00",
    periods=N_ROWS,
    freq="5min"
)


def clip_non_negative(x):
    return np.maximum(x, 0.0)


def add_autocorrelated_noise(
    n,
    scale=1.0,
    phi=0.75
):
    """
    Generate correlated noise rather than independent
    random noise for every row.
    """
    eps = rng.normal(0, scale, n)

    noise = np.zeros(n)

    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + eps[i]

    return noise


def work_hour_factor(hours):
    """
    Smooth workload pattern:
    - Low activity at night
    - Gradual increase before 09:00
    - High during working hours
    - Gradual decrease after 18:00
    """
    factor = np.zeros(len(hours))

    for i, hour in enumerate(hours):
        if hour < 6:
            factor[i] = 0.25
        elif hour < 9:
            factor[i] = 0.25 + (hour - 6) / 3 * 0.65
        elif hour < 18:
            factor[i] = 0.90
        elif hour < 21:
            factor[i] = 0.90 - (hour - 18) / 3 * 0.55
        else:
            factor[i] = 0.35

    return factor


server_indices = np.arange(N_ROWS) % N_SERVERS

server_id_array = np.array(
    [servers[i]["server_id"] for i in server_indices], dtype=object
)

hostname_array = np.array(
    [servers[i]["hostname"] for i in server_indices], dtype=object
)

role_array = np.array(
    [servers[i]["role"] for i in server_indices], dtype=object
)

periods_needed = -(-N_ROWS // N_SERVERS)
base_time = pd.date_range(
    start="2026-01-01 00:00:00",
    periods=periods_needed,
    freq="5min"
)

row_time_idx = np.arange(N_ROWS) // N_SERVERS
timestamp_array = base_time.to_numpy()[row_time_idx]


hours = pd.DatetimeIndex(timestamp_array).hour
minutes = pd.DatetimeIndex(timestamp_array).minute
days = pd.DatetimeIndex(timestamp_array).dayofweek

hour_decimal = hours + minutes / 60.0

work_factor = work_hour_factor(hour_decimal)


cpu_base = np.zeros(N_ROWS)
memory_base = np.zeros(N_ROWS)
network_base = np.zeros(N_ROWS)
response_base = np.zeros(N_ROWS)

for i, role in enumerate(role_array):

    if role == "Web Server":
        cpu_base[i] = 25 + 42 * work_factor[i]
        memory_base[i] = 48 + 18 * work_factor[i]
        network_base[i] = 70 + 180 * work_factor[i]
        response_base[i] = 35 + 18 * work_factor[i]

    elif role == "Database Server":
        cpu_base[i] = 35 + 38 * work_factor[i]
        memory_base[i] = 62 + 20 * work_factor[i]
        network_base[i] = 45 + 110 * work_factor[i]
        response_base[i] = 55 + 28 * work_factor[i]

    elif role == "App Server":
        cpu_base[i] = 30 + 45 * work_factor[i]
        memory_base[i] = 55 + 22 * work_factor[i]
        network_base[i] = 55 + 140 * work_factor[i]
        response_base[i] = 45 + 23 * work_factor[i]


server_cpu_offset = {
    f"SRV-{i:03d}": rng.uniform(-5, 5)
    for i in range(1, N_SERVERS + 1)
}

server_memory_offset = {
    f"SRV-{i:03d}": rng.uniform(-4, 4)
    for i in range(1, N_SERVERS + 1)
}

server_network_factor = {
    f"SRV-{i:03d}": rng.uniform(0.75, 1.30)
    for i in range(1, N_SERVERS + 1)
}

server_response_offset = {
    f"SRV-{i:03d}": rng.uniform(-5, 8)
    for i in range(1, N_SERVERS + 1)
}


for i, sid in enumerate(server_id_array):
    cpu_base[i] += server_cpu_offset[sid]
    memory_base[i] += server_memory_offset[sid]
    network_base[i] *= server_network_factor[sid]
    response_base[i] += server_response_offset[sid]


cpu_noise = add_autocorrelated_noise(
    N_ROWS,
    scale=1.8,
    phi=0.70
)

memory_noise = add_autocorrelated_noise(
    N_ROWS,
    scale=0.9,
    phi=0.85
)

network_noise = add_autocorrelated_noise(
    N_ROWS,
    scale=8.0,
    phi=0.65
)

response_noise = add_autocorrelated_noise(
    N_ROWS,
    scale=2.5,
    phi=0.75
)


daily_cycle = np.sin(
    2 * np.pi * hour_decimal / 24
)

weekly_factor = np.where(
    days >= 5,
    0.75,
    1.0
)


cpu_usage = (
    cpu_base
    + cpu_noise
    + daily_cycle * 2.0
)

memory_usage = (
    memory_base
    + memory_noise
    + daily_cycle * 1.0
)

network_in = (
    network_base
    * weekly_factor
    + network_noise
)

network_out = (
    network_in * rng.uniform(0.35, 0.75, N_ROWS)
    + rng.normal(0, 6, N_ROWS)
)

response_time = (
    response_base
    + response_noise
    + np.maximum(cpu_usage - 75, 0) * 0.35
)


cpu_usage = np.clip(cpu_usage, 0, 89)
memory_usage = np.clip(memory_usage, 0, 95)

network_in = clip_non_negative(network_in)
network_out = clip_non_negative(network_out)

response_time = clip_non_negative(response_time)


df = pd.DataFrame({
    "server_id": server_id_array,
    "hostname": hostname_array,
    "role": role_array,
    "timestamp": pd.to_datetime(timestamp_array),
    "cpu_usage": cpu_usage.astype(float),
    "memory_usage": memory_usage.astype(float),
    "network_in": network_in.astype(float),
    "network_out": network_out.astype(float),
    "response_time": response_time.astype(float),
    "is_anomaly": np.zeros(N_ROWS, dtype=int),
    "anomaly_type": np.full(
        N_ROWS,
        "NONE",
        dtype=object
    )
})


anomaly_types = [
    "CPU_SPIKE",
    "NETWORK_SURGE",
    "RESPONSE_DEGRADATION"
]

target_anomaly_rows = int(
    N_ROWS * rng.uniform(0.02, 0.05)
)

used_ranges = []

current_anomaly_rows = 0


def ranges_overlap(start, end, ranges):
    for s, e in ranges:
        if not (end < s or start > e):
            return True
    return False


while current_anomaly_rows < target_anomaly_rows:

    episode_length = int(
        rng.integers(5, 21)
    )

    if current_anomaly_rows + episode_length > target_anomaly_rows:
        episode_length = target_anomaly_rows - current_anomaly_rows

    if episode_length < 5:
        break

    start = int(
        rng.integers(
            0,
            N_ROWS - episode_length
        )
    )

    end = start + episode_length - 1

    if ranges_overlap(start, end, used_ranges):
        continue

    used_ranges.append((start, end))

    anomaly_type = rng.choice(anomaly_types)

    positions = np.arange(
        start,
        end + 1
    )

    progress = np.linspace(
        0,
        1,
        episode_length
    )

    if anomaly_type == "CPU_SPIKE":

        spike_strength = (
            20
            + 12 * progress
            + rng.normal(0, 2, episode_length)
        )

        df.loc[
            positions,
            "cpu_usage"
        ] = np.clip(
            82 + spike_strength,
            90,
            100
        )

        df.loc[
            positions,
            "response_time"
        ] += np.linspace(
            10,
            35,
            episode_length
        )

    elif anomaly_type == "NETWORK_SURGE":

        surge_factor_in = rng.uniform(
            3.0,
            6.0
        )

        surge_factor_out = rng.uniform(
            2.5,
            5.0
        )

        ramp = (
            1.0
            + 0.25 * progress
        )

        df.loc[
            positions,
            "network_in"
        ] *= surge_factor_in * ramp

        df.loc[
            positions,
            "network_out"
        ] *= surge_factor_out * ramp

        df.loc[
            positions,
            "response_time"
        ] += np.linspace(
            5,
            25,
            episode_length
        )

    elif anomaly_type == "RESPONSE_DEGRADATION":

        degradation = (
            np.linspace(
                15,
                180,
                episode_length
            )
            + rng.normal(
                0,
                3,
                episode_length
            )
        )

        df.loc[
            positions,
            "response_time"
        ] += degradation

        df.loc[
            positions,
            "cpu_usage"
        ] += np.linspace(
            2,
            12,
            episode_length
        )

    df.loc[
        positions,
        "is_anomaly"
    ] = 1

    df.loc[
        positions,
        "anomaly_type"
    ] = anomaly_type

    current_anomaly_rows += episode_length


df["cpu_usage"] = np.clip(
    df["cpu_usage"],
    0,
    100
)

df["memory_usage"] = np.clip(
    df["memory_usage"],
    0,
    100
)

df["network_in"] = np.maximum(
    df["network_in"],
    0
)

df["network_out"] = np.maximum(
    df["network_out"],
    0
)

df["response_time"] = np.maximum(
    df["response_time"],
    0
)


df["is_anomaly"] = df["is_anomaly"].astype(int)

df["anomaly_type"] = df["anomaly_type"].astype(str)


expected_columns = [
    "server_id",
    "hostname",
    "role",
    "timestamp",
    "cpu_usage",
    "memory_usage",
    "network_in",
    "network_out",
    "response_time",
    "is_anomaly",
    "anomaly_type"
]

assert list(df.columns) == expected_columns

assert pd.api.types.is_object_dtype(df["server_id"]) or pd.api.types.is_string_dtype(df["server_id"])
assert pd.api.types.is_object_dtype(df["hostname"]) or pd.api.types.is_string_dtype(df["hostname"])
assert pd.api.types.is_object_dtype(df["role"]) or pd.api.types.is_string_dtype(df["role"])
assert pd.api.types.is_datetime64_any_dtype(
    df["timestamp"]
)

assert pd.api.types.is_float_dtype(
    df["cpu_usage"]
)

assert pd.api.types.is_float_dtype(
    df["memory_usage"]
)

assert pd.api.types.is_float_dtype(
    df["network_in"]
)

assert pd.api.types.is_float_dtype(
    df["network_out"]
)

assert pd.api.types.is_float_dtype(
    df["response_time"]
)

assert pd.api.types.is_integer_dtype(
    df["is_anomaly"]
)

assert df["is_anomaly"].isin([0, 1]).all()

assert df["anomaly_type"].isin([
    "NONE",
    "CPU_SPIKE",
    "NETWORK_SURGE",
    "RESPONSE_DEGRADATION"
]).all()

assert not df.isnull().any().any()

assert (df["cpu_usage"] >= 0).all()
assert (df["cpu_usage"] <= 100).all()
assert (df["memory_usage"] >= 0).all()
assert (df["memory_usage"] <= 100).all()
assert (df["network_in"] >= 0).all()
assert (df["network_out"] >= 0).all()
assert (df["response_time"] >= 0).all()


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("=" * 60)
print("Independent Test Set Generated")
print("=" * 60)

print(f"File: {OUTPUT_FILE}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print()
print("Anomaly distribution:")
print(
    df["is_anomaly"]
    .value_counts()
    .sort_index()
)

print()
print("Anomaly types:")
print(
    df["anomaly_type"]
    .value_counts()
)

print()
print("Anomaly percentage:")
print(
    f"{df['is_anomaly'].mean() * 100:.2f}%"
)

print()
print("Schema:")
print(df.dtypes)

print()
print("File saved successfully.")