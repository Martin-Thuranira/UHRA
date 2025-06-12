
import time
import numpy as np
from MCLlocalisation import MCL  

def main():
    # Known landmarks
    map_landmarks = {
        1: (10.0, 10.0),
        2: (10.0, 20.0),
        3: (20.0, 10.0),
        4: (20.0, 20.0)
    }

    # Initialize MCL
    mcl = MCL(
        num_particles=1000,
        map_size=(30, 30),
        initial_pose=(5.0, 5.0, 0.0)
    )

    # Fake "true" pose , initialises car position.
    true_pose = [5.0, 5.0, 0.0]

    for step in range(100):
        # Simulate motion in a circleish pattern
        delta_x = 0.5
        delta_y = 0.0
        delta_theta = 0.05
        true_pose[0] += delta_x * np.cos(true_pose[2])
        true_pose[1] += delta_x * np.sin(true_pose[2])
        true_pose[2] += delta_theta
        true_pose[2] %= 2 * np.pi

        # Generate simulated "observations" to landmarks from true pose
        observations = []
        for lid, (lx, ly) in map_landmarks.items():
            dx = lx - true_pose[0]
            dy = ly - true_pose[1]
            range_ = np.sqrt(dx**2 + dy**2) + np.random.normal(0, 0.1)
            bearing = (np.arctan2(dy, dx) - true_pose[2]) % (2 * np.pi) + np.random.normal(0, 0.05)
            observations.append((lid, range_, bearing))

        # Run MCL steps
        mcl.motion_model(delta_x, delta_y, delta_theta)
        mcl.measurement_model(observations, map_landmarks)
        mcl.resample()
        estimated_pose = mcl.estimate_pose()
        print(f"Step {step}: Estimated = {estimated_pose}, True = {true_pose}")

        # Visualization in a plot
        mcl.visualize(true_pose=true_pose, map_landmarks=map_landmarks)
        time.sleep(0.05)

        if step > 20 and np.std([p.x for p in mcl.particles]) < 0.1:
            print("Converged")
            break

if __name__ == "__main__":
    main()
