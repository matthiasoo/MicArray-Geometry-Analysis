from pathlib import Path

import acoular as ac
import numpy as np
import matplotlib.pyplot as plt

# import traits.api to enforce data types in object parameters
from traits.api import cached_property, Int, List, Property


class DroneSignalGenerator(ac.NoiseGenerator):
    """
    Class for generating a synthetic multicopter drone signal.
    This is just a basic example class for demonstration purposes
    with only few settable and some arbitrary fixed parameters.
    It is not intended to create perfectly realistic signals.
    """

    # List with rotor speeds (for each rotor independently)
    # Default: 1 rotor, 15000 rpm
    rpm_list = List([15000, ])

    # Number of blades per rotor
    # Default: 2
    num_blades_per_rotor = Int(2)

    # internal identifier
    digest = Property(depends_on=['rms', 'seed', 'sample_freq', 'num_samples', 'rpm_list', 'num_blades_per_rotor'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def signal(self):
        """
        function that returns the full signal
        """
        # initialize a random generator for noise generation
        rng = np.random.default_rng(seed=self.seed)
        # use 1/f² broadband noise as basis for the signal
        wn = rng.standard_normal(self.num_samples)  # normal distributed values
        wnf = np.fft.rfft(wn)  # transform to freq domain
        wnf /= (np.linspace(0.1, 1, len(wnf)) * 5) ** 2  # spectrum ~ 1/f²
        sig = np.fft.irfft(wnf)  # transform to time domain

        # vector with all time instances
        t = np.arange(self.num_samples, dtype=float) / self.sample_freq

        # iterate over all rotors
        for rpm in self.rpm_list:
            f_base = rpm / 60  # rotor speed in Hz

            # randomly set phase of rotor
            phase = rng.uniform() * 2 * np.pi

            # calculate higher harmonics up to 50 times the rotor speed
            for n in np.arange(50) + 1:
                # if we're looking at a blade passing frequency, make it louder
                if n % self.num_blades_per_rotor == 0:
                    amp = 1
                else:
                    amp = 0.2

                # exponentially decrease amplitude for higher freqs with arbitrary factor
                amp *= np.exp(-n / 10)

                # add harmonic signal component to existing signal
                sig += amp * np.sin(2 * np.pi * n * f_base * t + phase)

                # return signal normalized to given RMS value
        return sig * self.rms / np.std(sig)

# length of signal
t_msm = 10.5 # s
# sampling frequency
f_sample = 44100 # Hz

drone_signal = DroneSignalGenerator(rpm_list = [15010,14962,13536,13007],
                                    num_blades_per_rotor = 2,
                                    sample_freq = f_sample,
                                    num_samples = f_sample*t_msm)

# If you're running the example in an interactive environment, you might want
# to listen to the pure signal by uncommenting the two following lines:
#from IPython.display import Audio
#display(Audio(drone_signal.signal(),rate = f_sample))

geom_path=Path('../geom/spiral_64.xml')
m = ac.MicGeom(from_file=geom_path)

flight_speed = 16 # m/s

# 11 seconds trajectory, which is a little more than we have signal for
ts = np.arange(12)

# initialize a random generator for path deviations
rng = np.random.default_rng(seed = 23)

# Set one waypoint each second,
waypoints = { t : ((t-5.5)*flight_speed,       # vary
                     0.0, # randomly vary y position up to ±0.2 m around 6 m
                    10) # randomly vary z position up to ±0.3 m around 10 m height
              for t in ts }

traj = ac.Trajectory(points = waypoints)

# We'll keep the environment simple for now: just air at standard conditions with speed of sound 343 m/s
e = ac.Environment(c=343.)

# Define point source
p = ac.MovingPointSourceDipole(signal = drone_signal, # the signal of the source
                               trajectory = traj,     # set trajectory
                               conv_amp = True,       # take into account convective amplification
                               mics = m,              # set the "array" with which to measure the sound field
                               start = 0.5,           # observation starts 0.5 seconds after signal starts at drone
                               env = e)               # the environment the source is moving in

# Copy the waypoints from the original source into a new trajectory, but with inverted z
waypoints_reflection = { time : (x, y, -z) for time, (x, y, z) in waypoints.items() }
traj_reflection = ac.Trajectory(points = waypoints_reflection)

# Define a mirror source with the mirrored trajectory
p_reflection = ac.MovingPointSourceDipole(signal = drone_signal,        # the same signal as above
                                          trajectory = traj_reflection, # set trajectory of mirror source
                                          conv_amp = True,
                                          mics = m,
                                          start = 0.5,
                                          env = e)

# Mix the original source and the mirror source
drone_above_ground = ac.SourceMixer( sources = [p, p_reflection] )

# Write data stream onto disk for later re-use. This step is not necessary if runtime isn't an issue.


# Prepare wav output.
# If you don't need caching, you can directly put "source = drone_above_ground" here.
all_channels = list(range(m.num_mics))
output = ac.WriteWAV(name = 'drone_flyby_with_ground_reflection.wav',
                     source = p,
                     channels = all_channels) # export both channels as stereo

# Start the actual export
output.save()