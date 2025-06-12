class Particle:
    def __init__(self, x, y, theta, weight=1.0):
        # Represents a hypothesis of the robot's pose
        self.x = x              # X-position
        self.y = y              # Y-position
        self.theta = theta      # Orientation in radians
        self.weight = weight    # Importance weight

class MCL:
    def __init__(self, num_particles, map_size, initial_pose=None):
        """
        Initialize the MCL algorithm.
        - num_particles: number of particles used in the filter
        - map_size: tuple (width, height) of the environment
        - initial_pose: optional (x, y, theta) if initial guess is known
        """
        self.num_particles = num_particles
        self.map_width, self.map_height = map_size

        # Initialize particles around initial_pose or uniformly across the map
        if initial_pose:
            self.particles = [Particle(*initial_pose) for _ in range(num_particles)]
        else:
            self.particles = [
                Particle(
                    x=np.random.uniform(0, self.map_width),
                    y=np.random.uniform(0, self.map_height),
                    theta=np.random.uniform(0, 2 * np.pi)
                ) for _ in range(num_particles)
            ]

        self.normalize_weights()  # Ensure valid probability distribution

        # Motion model noise parameters
        self.alpha1 = 0.1  # Noise for first rotation
        self.alpha2 = 0.1  # Noise for translation
        self.alpha3 = 0.1  # Noise for second rotation
        self.alpha4 = 0.1  # Additional translation noise

    def normalize_weights(self):
        """Normalize all particle weights so they sum to 1"""
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight

    def motion_model(self, delta_x, delta_y, delta_theta):
        """
        Apply noisy motion model to each particle based on robot's odometry.
        - delta_x, delta_y, delta_theta: movement since last step
        """
        for p in self.particles:
            # Add Gaussian noise to simulate uncertainty in motion
            noisy_delta_x = delta_x + np.random.normal(0, self.alpha2 * abs(delta_x) + self.alpha4 * abs(delta_theta))
            noisy_delta_y = delta_y + np.random.normal(0, self.alpha2 * abs(delta_y) + self.alpha4 * abs(delta_theta))
            noisy_delta_theta = delta_theta + np.random.normal(0, self.alpha1 * abs(delta_theta) + self.alpha3 * (abs(delta_x) + abs(delta_y)))

            # Update pose using noisy motion
            p.x += noisy_delta_x * np.cos(p.theta) - noisy_delta_y * np.sin(p.theta)
            p.y += noisy_delta_x * np.sin(p.theta) + noisy_delta_y * np.cos(p.theta)
            p.theta = (p.theta + noisy_delta_theta) % (2 * np.pi)

            # Constrain to map boundaries
            p.x = np.clip(p.x, 0, self.map_width)
            p.y = np.clip(p.y, 0, self.map_height)

    def measurement_model(self, observations, map_landmarks):
        """
        Update particle weights based on sensor observations.
        - observations: list of (landmark_id, measured_range, measured_bearing)
        - map_landmarks: dictionary of true landmark positions {id: (x, y)}
        """
        for p in self.particles:
            p.weight = 1.0  # Reset weight

            for obs in observations:
                landmark_id, measured_range, measured_bearing = obs

                if landmark_id in map_landmarks:
                    lx, ly = map_landmarks[landmark_id]
                    
                    # Compute expected range and bearing from particle to landmark
                    dx = lx - p.x
                    dy = ly - p.y
                    expected_range = np.sqrt(dx**2 + dy**2)
                    expected_bearing = (np.arctan2(dy, dx) - p.theta) % (2 * np.pi)

                    # Likelihood from Gaussian distribution
                    range_prob = norm.pdf(measured_range - expected_range, 0, 0.1)
                    bearing_prob = norm.pdf(measured_bearing - expected_bearing, 0, 0.05)

                    # Combine probabilities
                    p.weight *= range_prob * bearing_prob

        # Normalize particle weights
        self.normalize_weights()

    def resample(self):
        """
        Resample particles using low-variance resampling.
        Focuses particles on high-probability areas.
        """
        new_particles = []
        step = 1.0 / self.num_particles
        r = np.random.uniform(0, step)
        c = self.particles[0].weight
        i = 0

        for _ in range(self.num_particles):
            u = r + _ * step
            while u > c:
                i += 1
                c += self.particles[i].weight
            # Create a new particle copy
            new_particles.append(Particle(
                self.particles[i].x,
                self.particles[i].y,
                self.particles[i].theta,
                1.0  # Reset weight to uniform
            ))

        self.particles = new_particles

    def estimate_pose(self):
        """
        Estimate robot's current pose using weighted average of particles.
        Returns a tuple (x, y, theta).
        """
        x = sum(p.x * p.weight for p in self.particles)
        y = sum(p.y * p.weight for p in self.particles)

        # Use circular mean for orientation
        sin_theta = sum(np.sin(p.theta) * p.weight for p in self.particles)
        cos_theta = sum(np.cos(p.theta) * p.weight for p in self.particles)
        theta = np.arctan2(sin_theta, cos_theta)

        return x, y, theta

    def visualize(self, true_pose=None, map_landmarks=None):
        """
        Plot particles, estimated pose, and optionally the true pose and landmarks.
        - true_pose: actual robot pose (x, y, theta)
        - map_landmarks: dictionary of landmark locations
        """
        plt.clf()

        # Plot all particles
        xs = [p.x for p in self.particles]
        ys = [p.y for p in self.particles]
        plt.scatter(xs, ys, c='blue', s=5, alpha=0.3, label='Particles')

        # Plot estimated pose
        est_x, est_y, est_theta = self.estimate_pose()
        plt.plot(est_x, est_y, 'go', markersize=10, label='Estimated Pose')

        # Plot true pose if known
        if true_pose:
            plt.plot(true_pose[0], true_pose[1], 'ro', markersize=10, label='True Pose')

        # Plot landmarks if provided
        if map_landmarks:
            for landmark_id, (lx, ly) in map_landmarks.items():
                plt.plot(lx, ly, 'k*', markersize=10)
                plt.text(lx, ly, str(landmark_id))

        # Set plot boundaries and labels
        plt.xlim(0, self.map_width)
        plt.ylim(0, self.map_height)
        plt.grid(True)
        plt.legend()
        plt.pause(0.01)  # For live updates
