import numpy as np


class FruitFlyBrain:

    def __init__(self, n_hidden=96, seed=7):
        self.rng = np.random.default_rng(seed)
        self.n_hidden = n_hidden

        self.w_in = self.rng.normal(0.0, 0.35, (16, n_hidden)).astype(np.float32)
        self.w_rec = self.rng.normal(
            0.0, 0.08, (n_hidden, n_hidden)
        ).astype(np.float32)
        self.w_out = self.rng.normal(0.0, 0.25, (n_hidden, 2)).astype(np.float32)

        self.v = np.zeros(n_hidden, dtype=np.float32)
        self.refractory = np.zeros(n_hidden, dtype=np.float32)
        self.activity = np.zeros(n_hidden, dtype=np.float32)

        self.v_rest = -0.65
        self.v_reset = -0.85
        self.v_threshold = 0.55
        self.tau = 0.010
        self.refractory_time = 0.001

        self.motor_state = np.zeros(2, dtype=np.float32)
        self.memory = np.zeros(4, dtype=np.float32)

    def reset(self):
        self.v.fill(self.v_rest)
        self.refractory.fill(0.0)
        self.activity.fill(0.0)
        self.motor_state.fill(0.0)
        self.memory.fill(0.0)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def _encode_senses(self, obs):
        joints = np.asarray(obs["joints"], dtype=np.float32)
        fly = np.asarray(obs["fly"], dtype=np.float32)
        contacts = np.asarray(obs["contact_forces"], dtype=np.float32)

        joint_angles = joints[0]
        joint_vel = joints[1]
        body_vel = fly[1]
        body_rot_vel = fly[3]

        contact_mag = np.linalg.norm(contacts, axis=1) if contacts.size else np.zeros(1)
        contact_left = float(contact_mag[: len(contact_mag) // 2].mean())
        contact_right = float(contact_mag[len(contact_mag) // 2 :].mean())

        forward_speed = float(body_vel[0])
        lateral_speed = float(body_vel[1])
        yaw_rate = float(body_rot_vel[2])

        joint_mean = float(np.mean(joint_angles))
        joint_speed = float(np.mean(np.abs(joint_vel)))

        left_motor_bias = float(np.mean(joint_vel[:21]))
        right_motor_bias = float(np.mean(joint_vel[21:]))

        vision = np.asarray(obs.get("vision", np.zeros((2, 1, 2))))
        if vision.ndim == 3:
            left_vision = float(vision[0].mean()) / 255.0
            right_vision = float(vision[1].mean()) / 255.0
        else:
            left_vision = right_vision = 0.0

        odor = np.asarray(obs.get("odor_intensity", 0.0), dtype=np.float32)
        odor_signal = float(np.nan_to_num(odor, nan=0.0).mean())
        target = np.asarray(obs.get("target_vector", (0.0, 0.0)))

        return np.array(
            [
                np.tanh(forward_speed / 10.0),
                np.tanh(lateral_speed / 10.0),
                np.tanh(yaw_rate / 10.0),
                np.tanh(contact_left / 20.0),
                np.tanh(contact_right / 20.0),
                np.tanh(joint_mean),
                np.tanh(joint_speed / 10.0),
                np.tanh(left_motor_bias / 10.0),
                np.tanh(right_motor_bias / 10.0),
                1.0 - left_vision,
                1.0 - right_vision,
                np.tanh(odor_signal),
                np.tanh(float(target[0])),
                np.tanh(float(target[1])),
                float(self.memory.mean()),
                1.0,
            ],
            dtype=np.float32,
        )

    def step(self, obs, dt):
        target = np.asarray(obs.get("target_vector", (0.0, 0.0)))
        self.memory *= np.float32(np.exp(-dt / 0.5))
        self.memory[0] += np.float32(np.tanh(target[0])) * dt
        self.memory[1] += np.float32(np.tanh(target[1])) * dt
        x = self._encode_senses(obs)

        recurrent_drive = self.activity @ self.w_rec
        drive = x @ self.w_in + recurrent_drive

        active = self.refractory <= 0.0
        self.v[active] += (dt / self.tau) * (
            -self.v[active] + np.tanh(drive[active])
        )

        self.refractory = np.maximum(0.0, self.refractory - dt)

        spikes = (self.v >= self.v_threshold) & active
        self.v[spikes] = self.v_reset
        self.refractory[spikes] = self.refractory_time

        self.activity *= np.float32(np.exp(-dt / 0.008))
        self.activity[spikes] += 1.0

        raw_motor = np.tanh(self.activity @ self.w_out)

        self.motor_state += (dt / 0.012) * (raw_motor - self.motor_state)
        self.motor_state = np.clip(self.motor_state, -1.0, 1.0)

        return self.motor_state.copy()

    def learn(self, reward):
        self.w_out += np.float32(0.0001 * reward) * self.activity[:, None]
        self.w_out = np.clip(self.w_out, -0.5, 0.5)

    @property
    def firing_rate_proxy(self):
        return float(np.mean(self.activity))

    @property
    def active_neurons(self):
        return int(np.count_nonzero(self.activity > 0.05))
