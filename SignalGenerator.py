import copy
import acoular as ac
import numpy as np
from IPython.core.pylabtools import figsize
from acoular.internal import digest
from traits.api import cached_property, Int, List, Property
from pathlib import Path
import matplotlib.pyplot as plt

ac.config.global_caching = 'none'

class DroneSignalGenerator(ac.NoiseGenerator):
    # defaults
    rpm_list = List([15000, ])
    num_blades_per_rotor = Int(2)

    digest = Property(depends_on=['rms', 'seed', 'sample_freq', 'num_samples', 'rpm_list', 'num_blades_per_rotor'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def signal(self):
        # initialize a random generator for noise generation
        rng = np.random.default_rng(seed=self.seed)
        # use 1/f² broadband noise as basis for the signal
        wn = rng.standard_normal(self.num_samples)  # WHITE NOISE
        wnf = np.fft.rfft(wn)  # to freq domain
        wnf /= (np.linspace(0.1, 1, len(wnf)) * 5) ** 2  # RED NOISE
        sig = np.fft.irfft(wnf)  # to time domain

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

class SignalRecorder:
    def __init__(self, geom_path, signal_src):
        self.geom_path = Path(geom_path)
        if not self.geom_path.exists():
            raise FileNotFoundError(f"Geom file not found: {geom_path}")
        self.mics = ac.MicGeom(from_file=geom_path)
        self.signal = signal_src
        self.env = ac.Environment()
        self.output_dir = Path('signal')
        self.output_name = self.geom_path.stem

    def run_static(self, loc=(0, 0, 10)) :
        p = ac.PointSourceDipole(
            signal=self.signal,
            mics=self.mics,
            env=self.env,
            loc=loc,
        )

        save_path = self.output_dir / 'static'
        save_path.mkdir(parents=True, exist_ok=True)
        all_channels = list(range(self.mics.num_mics))

        file_wav = save_path / f"{self.output_name}_static.wav"
        output = ac.WriteWAV(source=p, name=str(file_wav), channels=all_channels)
        output.save()
        print(f"WAV saved: {file_wav}")

    def run_dynamic(self, speed=10.0, height=10.0, start_offset=0.5):
        t_traj = np.arange(10) # traj duration

        waypoints = { t : ((t - 5) * speed, 0.0, height) for t in t_traj }
        traj = ac.Trajectory(points=waypoints)

        p = ac.MovingPointSourceDipole(
            signal=self.signal,
            trajectory=traj,
            mics=self.mics,
            env=self.env,
            conv_amp=True,
            start=start_offset,
            direction=(0, 0, 1)
        )

        waypoints_reflection = {time: (x, y, -z) for time, (x, y, z) in waypoints.items()}
        traj_reflection = ac.Trajectory(points=waypoints_reflection)

        # mirror src
        p_reflection = ac.MovingPointSourceDipole(
            signal=self.signal,
            trajectory=traj_reflection,
            conv_amp=True,
            mics=self.mics,
            start=0.5,
            env=self.env,
            direction=(0, 0, -1)
        )

        wn_gen = ac.WNoiseGenerator(
            sample_freq=self.signal.sample_freq,
            num_samples=self.signal.num_samples,
            seed=100,
            rms=0.05
        )

        n = ac.UncorrelatedNoiseSource(
            signal=wn_gen,
            mics=self.mics
        )

        drone_above_ground = ac.SourceMixer(sources=[p, p_reflection, n])

        save_path = self.output_dir / 'dynamic'
        save_path.mkdir(parents=True, exist_ok=True)
        all_channels = list(range(self.mics.num_mics))

        file_wav = save_path / f"{self.output_name}_dynamic.wav"
        output = ac.WriteWAV(source=drone_above_ground, name=str(file_wav), channels=all_channels)
        output.save()
        print(f"WAV saved: {file_wav}")