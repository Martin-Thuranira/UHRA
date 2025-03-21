import numpy as np
import matplotlib.pyplot as plt
from uefs_simulator import Vehicle, Lidar  # Assuming UEFS provides these classes

class Particle:
    def __init__(self, x, y, theta, weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.landmarks = {}  # Dictionary to store landmark positions

class FastSLAM:
    def __init__(self, num_particles, initial_pose):
        self.num_particles = num_particles
        self.particles = [Particle(*initial_pose, 1.0 / num_particles) for _ in range(num_particles)]
        self.landmarks = {}  # Global map of landmarks

    def motion_model(self, velocity, steering_angle, dt):
        for particle in self.particles:
            # Update particle's pose based on motion model
            particle.x += velocity * np.cos(particle.theta) * dt
            particle.y += velocity * np.sin(particle.theta) * dt
            particle.theta += steering_angle * dt

    def measurement_model(self, observations):
        for particle in self.particles:
            for obs in observations:
                landmark_id, range_, bearing = obs
                if landmark_id not in particle.landmarks:
                    # Initialize new landmark
                    particle.landmarks[landmark_id] = (particle.x + range_ * np.cos(particle.theta + bearing),
                                                       particle.y + range_ * np.sin(particle.theta + bearing)
                else:
                    # Update existing landmark using EKF or other methods
                    pass

    def resample(self):
        weights = [particle.weight for particle in self.particles]
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / self.num_particles] * self.num_particles
        else:
            weights = [w / total_weight for w in weights]
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=weights)
        self.particles = [self.particles[i] for i in indices]

    def update(self, velocity, steering_angle, dt, observations):
        self.motion_model(velocity, steering_angle, dt)
        self.measurement_model(observations)
        self.resample()

    def get_best_particle(self):
        return max(self.particles, key=lambda p: p.weight)

def main():
    # Initialize UEFS simulator
    vehicle = Vehicle()
    lidar = Lidar()

    # Initial pose of the vehicle
    initial_pose = (0.0, 0.0, 0.0)  # x, y, theta

    # Initialize FastSLAM
    fastslam = FastSLAM(num_particles=100, initial_pose=initial_pose)

    # Simulation loop
    for _ in range(1000):  # Run for 1000 time steps
        # Get control inputs (velocity, steering angle)
        velocity, steering_angle = vehicle.get_control_inputs()

        # Get sensor observations (landmarks)
        observations = lidar.get_observations()

        # Update FastSLAM
        fastslam.update(velocity, steering_angle, dt=0.1, observations=observations)

        # Get the best particle (estimated pose)
        best_particle = fastslam.get_best_particle()
        print(f"Estimated Pose: ({best_particle.x}, {best_particle.y}, {best_particle.theta})")

        # Visualize the map and trajectory
        plt.clf()
        for particle in fastslam.particles:
            plt.plot(particle.x, particle.y, 'ro', markersize=1)
        for landmark_id, (lx, ly) in best_particle.landmarks.items():
            plt.plot(lx, ly, 'bo')
        plt.pause(0.01)

if __name__ == "__main__":
    main()
