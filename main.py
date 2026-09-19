import time

import cv2
import numpy as np

from brain import FruitFlyBrain
from flygym_gymnasium import Camera, Fly, SingleFlySimulation
from flygym_gymnasium.arena import FlatTerrain
from flygym_gymnasium.examples.locomotion import CPGNetwork, PreprogrammedSteps


TIME = 1e-4
RUN_TIME = 60.0
DISPLAY_FPS = 2500
RENDER_SIZE = (320, 180)
WINDOW_NAME = "hell."
NEURAL_WINDOW_NAME = "brain"


class FeatureTerrain(FlatTerrain):
    def __init__(self):
        super().__init__()
        self.odor_source = np.array([12.0, 0.0, 0.0], dtype=np.float32)
        self.obstacle_positions = np.array(
            [[6.0, 2.5], [13.0, -2.0], [20.0, 2.0]], dtype=np.float32
        )
        self.target_positions = np.array(
            [[8.0, 0.0, 0.25], [16.0, 5.0, 0.25], [24.0, -3.0, 0.25]],
            dtype=np.float32,
        )
        self.target_index = 0
        self.dark = False
        self._set_white_materials()

    def _set_white_materials(self):
        for texture in self.root_element.asset.find_all("texture"):
            if getattr(texture, "type", None) == "skybox":
                texture.builtin = "flat"
                texture.rgb1 = "1 1 1"
                texture.rgb2 = "1 1 1"
        for material in self.root_element.asset.find_all("material"):
            material.rgba = "1 1 1 1"
            material.reflectance = 0.0
            material.shininess = 0.0
            material.specular = 0.0

    @property
    def odor_dimensions(self):
        return 1

    def get_olfaction(self, antennae_pos):
        distances = np.linalg.norm(antennae_pos[:, :2] - self.odor_source[:2], axis=1)
        intensity = np.exp(-distances / 8.0).astype(np.float32)
        return intensity.reshape(1, -1)

    def target_vector(self, position):
        target = self.target_positions[self.target_index]
        delta = target[:2] - position[:2]
        distance = float(np.linalg.norm(delta))
        if distance < 1.0 and self.target_index < len(self.target_positions) - 1:
            self.target_index += 1
            target = self.target_positions[self.target_index]
            delta = target[:2] - position[:2]
            distance = float(np.linalg.norm(delta))
        return np.array([delta[0] / 20.0, delta[1] / 20.0], dtype=np.float32), distance

    def obstacle_features(self, position):
        offsets = self.obstacle_positions - position[:2]
        distances = np.linalg.norm(offsets, axis=1)
        nearest = int(np.argmin(distances))
        return offsets[nearest] / 10.0, float(distances[nearest])

    def toggle_light(self, physics):
        self.dark = not self.dark
        brightness = 0.12 if self.dark else 0.8
        physics.model.vis.headlight.ambient[:] = brightness
        physics.model.vis.headlight.diffuse[:] = brightness
        physics.model.vis.headlight.specular[:] = 0.0


class BrainDrivenFly:
    def __init__(self, brain, cpg, preprogrammed_steps):
        self.brain = brain
        self.cpg = cpg
        self.preprogrammed_steps = preprogrammed_steps
        self.last_brain_output = np.zeros(2, dtype=np.float32)
        self.drive_vector = np.ones(6, dtype=np.float64)
        self.smoothed_drive = np.ones(2, dtype=np.float64)
        self.last_target_distance = 0.0

    def reset(self):
        self.brain.reset()
        self.cpg.reset()
        self.last_brain_output.fill(0.0)
        self.smoothed_drive.fill(1.0)

    def step(self, sim, obs):
        output = np.asarray(self.brain.step(obs, sim.timestep), dtype=np.float32)
        self.last_brain_output = np.resize(output, 2)

        target = np.asarray(obs.get("target_vector", (0.0, 0.0)))
        obstacle = np.asarray(obs.get("obstacle_vector", (0.0, 0.0)))
        touch = float(obs.get("touch_level", 0.0))
        vision = np.asarray(obs.get("vision", np.zeros((2, 1, 2))))
        visual_turn = 0.0
        if vision.ndim == 3 and vision.shape[0] == 2:
            visual_turn = float(vision[0].mean() - vision[1].mean()) / 255.0

        speed = np.clip(
            1.9 + 0.22 * self.last_brain_output[0] - 0.15 * touch,
            1.0,
            2.2,
        )
        turn = np.clip(
            0.10 * self.last_brain_output[1]
            + 0.20 * target[1]
            - 0.20 * obstacle[1] / max(abs(float(obstacle[0])), 1.0)
            + 0.08 * visual_turn,
            -0.35,
            0.35,
        )
        requested_drive = np.array([speed - turn, speed + turn])
        self.smoothed_drive += 0.35 * (requested_drive - self.smoothed_drive)
        self.drive_vector[:3] = self.smoothed_drive[0]
        self.drive_vector[3:] = self.smoothed_drive[1]
        self.cpg.intrinsic_amps = np.clip(self.drive_vector, 0.9, 1.6)
        self.cpg.step()

        joints = []
        adhesion = []
        for index, leg in enumerate(self.preprogrammed_steps.legs):
            phase = self.cpg.curr_phases[index]
            magnitude = self.cpg.curr_magnitudes[index]
            joints.append(
                self.preprogrammed_steps.get_joint_angles(
                    leg, phase, magnitude
                )
            )
            adhesion.append(
                self.preprogrammed_steps.get_adhesion_onoff(leg, phase)
            )

        return {
            "joints": np.concatenate(joints),
            "adhesion": np.asarray(adhesion, dtype=np.int32),
        }

    def learn(self, reward):
        self.brain.learn(reward)


class AlertSystem:
    def __init__(self):
        self.message = ""
        self.expires_at = 0.0

    def trigger(self, message, frequency=700):
        self.message = message
        self.expires_at = time.perf_counter() + 1.0

    def active_message(self):
        return self.message if time.perf_counter() < self.expires_at else ""


def build_cpg(timestep):
    phase_biases = np.pi * np.array(
        [
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
        ]
    )
    return CPGNetwork(
        timestep=timestep,
        intrinsic_freqs=np.ones(6) * 180.0,
        intrinsic_amps=np.ones(6),
        phase_biases=phase_biases,
        coupling_weights=(phase_biases > 0) * 10.0,
        convergence_coefs=np.ones(6) * 20.0,
        init_magnitudes=np.ones(6),
    )


class CameraController:
    def __init__(self):
        self.azimuth = -np.pi / 2.0
        self.distance = 12.0
        self.height = 8.0
        self.pitch = 0.35
        self.dragging = False
        self.last_mouse = None

    def mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.last_mouse = None
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            old_x, old_y = self.last_mouse
            self.azimuth += (x - old_x) * 0.012
            self.pitch = np.clip(self.pitch - (y - old_y) * 0.008, -1.2, 1.2)
            self.last_mouse = (x, y)

    def update(self, camera, simulation):
        target = simulation.physics.data.qpos[:3].copy()
        target[2] += 0.2
        horizontal = self.distance * np.cos(self.pitch)
        camera_position = target + np.array(
            [
                horizontal * np.cos(self.azimuth),
                horizontal * np.sin(self.azimuth),
                self.height + self.distance * np.sin(self.pitch),
            ]
        )
        direction = target - camera_position
        direction /= np.linalg.norm(direction)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(direction, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, direction)
        binding = simulation.physics.bind(camera._cam)
        binding.xpos = camera_position
        binding.xmat = np.column_stack((right, up, -direction)).flatten()


def add_brain_features(obs, terrain, fly_position):
    target_vector, distance = terrain.target_vector(fly_position)
    obstacle_vector, obstacle_distance = terrain.obstacle_features(fly_position)
    obs["target_vector"] = target_vector
    obs["obstacle_vector"] = obstacle_vector
    obs["obstacle_distance"] = np.float32(obstacle_distance)
    obs["touch_level"] = np.float32(
        np.linalg.norm(np.asarray(obs["contact_forces"], dtype=np.float32))
    )
    return distance, obstacle_distance


def show_frame(rendered):
    if isinstance(rendered, (list, tuple)):
        rendered = rendered[0] if rendered else None
    if rendered is None:
        return cv2.waitKeyEx(1) & 0xFF
    frame = np.asarray(rendered)
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(WINDOW_NAME, frame)
    return cv2.waitKeyEx(1) & 0xFF


def show_neural_activity(brain, reward_history):
    width, height = 720, 480
    diagram = np.full((height, width, 3), 255, dtype=np.uint8)
    hidden_count = brain.n_hidden
    hidden_columns = 8
    hidden_points = []
    for index in range(hidden_count):
        x = 180 + (index % hidden_columns) * 48
        y = 90 + (index // hidden_columns) * 72
        hidden_points.append((x, y))
    input_points = [(45, 55 + index * 24) for index in range(16)]
    output_points = [(650, 180), (650, 300)]

    for index, point in enumerate(hidden_points):
        input_index = index % len(input_points)
        weight = abs(float(brain.w_in[input_index, index]))
        color = (200, 200, 200) if weight < 0.2 else (80, 150, 220)
        cv2.line(diagram, input_points[input_index], point, color, 1)
        output_index = index % 2
        output_weight = abs(float(brain.w_out[index, output_index]))
        color = (180, 180, 180) if output_weight < 0.2 else (60, 120, 220)
        cv2.line(diagram, point, output_points[output_index], color, 1)

    for index, point in enumerate(input_points):
        cv2.circle(diagram, point, 5, (80, 80, 80), -1)
    for index, point in enumerate(hidden_points):
        activity = float(np.clip(brain.activity[index], 0.0, 1.0))
        cv2.circle(diagram, point, 10, (40, int(220 - 160 * activity), 255), -1)
    for point in output_points:
        cv2.circle(diagram, point, 14, (40, 80, 220), -1)

    if reward_history:
        values = np.asarray(reward_history[-180:], dtype=np.float32)
        values = np.clip(values, -1.0, 1.0)
        points = []
        for index, value in enumerate(values):
            points.append((index * 3 + 180, 445 - int(value * 35)))
        if len(points) > 1:
            cv2.polylines(diagram, [np.asarray(points, dtype=np.int32)], False, (20, 120, 220), 2)
    cv2.imshow(NEURAL_WINDOW_NAME, diagram)


def main():
    terrain = FeatureTerrain()
    fly = Fly(
        name="fly_1",
        enable_vision=True,
        enable_olfaction=True,
        enable_adhesion=True,
        draw_adhesion=False,
        vision_refresh_rate=200,
        init_pose="stretch",
        control="position",
        spawn_pos=(0.0, 0.0, 0.0),
    )
    camera = Camera(
        terrain.root_element.worldbody,
        camera_name="overview",
        camera_parameters={"mode": "fixed", "pos": (-12, -14, 9), "euler": (1.1, 0, -0.7), "fovy": 45},
        window_size=RENDER_SIZE,
        fps=DISPLAY_FPS,
        timestamp_text=False,
        play_speed_text=False,
    )
    simulation = SingleFlySimulation(
        fly=fly,
        cameras=[camera],
        arena=terrain,
        timestep=TIME,
    )
    simulation.physics.model.mat_shininess[:] = 0.0
    simulation.physics.model.mat_reflectance[:] = 0.0
    simulation.physics.model.vis.map.znear = 0.01
    simulation.physics.model.vis.map.zfar = 20.0

    controller = BrainDrivenFly(
        FruitFlyBrain(n_hidden=32, seed=7),
        build_cpg(TIME),
        PreprogrammedSteps(),
    )
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *RENDER_SIZE)
    cv2.namedWindow(NEURAL_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(NEURAL_WINDOW_NAME, 720, 480)
    camera_controller = CameraController()
    cv2.setMouseCallback(WINDOW_NAME, camera_controller.mouse_callback)
    alert = AlertSystem()
    previous_target_index = terrain.target_index
    last_frame_time = time.perf_counter()
    displayed_fps = 0.0
    reward_history = []

    try:
        observation, _ = simulation.reset(seed=7)
        controller.reset()
        total_steps = int(RUN_TIME / 4e-4)
        render_every = 5
        for step_index in range(total_steps):
            distance, obstacle_distance = add_brain_features(
                observation,
                terrain,
                observation["fly"][0],
            )
            action = controller.step(simulation, observation)
            observation, _, terminated, truncated, _ = simulation.step(action)
            touch_level = float(np.linalg.norm(observation["contact_forces"]))
            reward = -0.001 * distance - 0.002 * touch_level
            controller.learn(reward)
            reward_history.append(reward)
            if obstacle_distance < 1.5:
                alert.trigger("obstacle nearby")
            if terrain.target_index != previous_target_index:
                alert.trigger("target reached")
                previous_target_index = terrain.target_index

            if step_index % render_every == 0:
                camera_controller.update(camera, simulation)
                rendered = simulation.render()
                now = time.perf_counter()
                elapsed = max(now - last_frame_time, 1e-6)
                displayed_fps = 0.9 * displayed_fps + 0.1 / elapsed
                last_frame_time = now
                key = show_frame(rendered)
                show_neural_activity(controller.brain, reward_history)
                if key == ord("q"):
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if terminated or truncated:
                break
    except Exception as error:
        import traceback

        traceback.print_exc()
        input(f"\nSimulation stopped: {error}\nPress Enter to close...")
    finally:
        simulation.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
