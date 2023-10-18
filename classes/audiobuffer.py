import pyaudio
import scipy.signal as ss
import numpy as np
import librosa
import time
from queue import Queue
import threading
import math


class Stream(threading.Thread):
    def __init__(self, path, chunk_size):
        self.track_data, self.track_rate = librosa.load(path, sr=44.1e3, dtype=np.float64, mono=False)
        self.last_coords_queue = Queue()
        self.last_coords_queue.put((0, 0, 0, 0))
        self.queue = Queue()
        self.buffer_alive = True
        self.chunk_size = chunk_size
        self.empty_chunk = np.zeros((2, chunk_size))
        self.sample_length = int(chunk_size//4)
        self.hop_length = int(self.sample_length//10)
        threading.Thread.__init__(self)
        
    def join(self):
        self.buffer_alive = False
        super().join()
        
    def run(self):
        remaining_frames = np.zeros((2, self.sample_length))
        last_coords = (0, 0)
        while self.buffer_alive:
            if self.queue.qsize() < 100:
                remaining_frames, last_coords = self._populate_chunk(remaining_frames, last_coords)
            else:
                time.sleep(0.05)
                
    def _populate_chunk(self, remaining_frames, last_coords):        
        samples_per_chunk = 0
        current_coords = (0, 0, 0, 0) # self.settings.coords[index]
        current_stack_length = 0
        chunk = self.empty_chunk.copy()
        
        while current_stack_length < self.chunk_size:
            if current_stack_length == 0 and np.any(remaining_frames):
                chunk[:, :remaining_frames.shape[1]] += remaining_frames
                remaining_frames = np.zeros((2, self.sample_length))
            else:
                read_from = np.random.randint(0, self.track_data.shape[1] - self.sample_length)
                sample = self.track_data[:, read_from:read_from + self.sample_length]
                if (current_stack_length + self.sample_length <= self.chunk_size):
                    chunk[:, current_stack_length:current_stack_length + self.sample_length] += sample
                else:
                    chunk[:, current_stack_length:] += sample[:, :self.chunk_size - current_stack_length]
                    remaining_frames[:, :self.sample_length - (self.chunk_size - current_stack_length)] += sample[:, self.chunk_size - current_stack_length:]
                    
                current_stack_length += self.hop_length
            samples_per_chunk += 1

        self.queue.put(chunk)
        return remaining_frames, current_coords

class AudioBuffer(threading.Thread):
    def __init__(self, settings):
        self.tail = Queue()
        self.settings = settings
        self.stream_array = []
        self.chunk_size = int(44100//2)
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 1-lluvia 1.wav', self.chunk_size))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 2-lluvia 2.wav', self.chunk_size))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 3-lluvia 3.wav', self.chunk_size))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 4-lluvia 4.wav', self.chunk_size))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 5-lluvia bajo lona.wav', self.chunk_size))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 6-lluvia bosque.wav', self.chunk_size))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 7-lluvia dentro de furgon.wav', self.chunk_size))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 8-lluvia dentro del furgon.wav', self.chunk_size))
        # instantiate PyAudio (1)
        self.p = pyaudio.PyAudio()
        self.first_IR, self.IR_rate = librosa.load('audios/IRs/208-R1_LargeRoom.wav', sr=44.1e3, dtype=np.float64, mono=False)
        track1_frame = self.stream_array[0].track_data[:,0 : self.chunk_size]
        track1 = ss.fftconvolve(track1_frame, self.first_IR, mode="full", axes=1)
        self.tail.put(np.zeros(track1.shape))
        self.people_counter = 0
        self.empty_chunk = np.zeros((2, self.chunk_size))
        self.doubled_empty_chunk = np.zeros((self.chunk_size * 2))
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=int(44100),
            output=True,
            stream_callback=self.get_callback(),
            frames_per_buffer=self.chunk_size
        )
        threading.Thread.__init__(self)

    def process_queue(self, index):
        queue = self.stream_array[index].queue
        frames = queue.get()
        frames = frames * self.compute_velocity_from_entropy(index)
        self.stream_array[index].last_coords_queue.put(self.settings.coords[index])
        return frames

    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            track = self.empty_chunk.copy()
            current_n_people = self.people_counter
            print("current_n_people", current_n_people)
            for i in range(current_n_people):
                track += self.process_queue(i) # * (1/current_n_people)
            actual_combinated_chunk = self._intercalate_channels(track)
            ret_data = actual_combinated_chunk.astype(np.float32).tobytes()
            return (ret_data, pyaudio.paContinue)
        return callback
    
    def start(self):
        for i in range(self.settings.max_people_counter):
            stream = self.stream_array[i]
            stream.start()
        super().start()

    def run(self):
        self.stream.start_stream()
        
    def join(self):
        for i in range(self.settings.max_people_counter):
            stream = self.stream_array[i]
            stream.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        super().join()
    
    def _intercalate_channels(self, chunk):
        return np.ravel(np.column_stack((chunk[0, :], chunk[1, :])))

    def _apply_stereo_panning(self, chunk, last_coords, current_coords):
        ramp = np.linspace(last_coords[0], current_coords[0], chunk.shape[1])
        chunk_l = np.multiply(chunk[1, :], (1 / self.settings.x_screen_size) * ramp)
        chunk_r = np.multiply(chunk[0, :], 1-(1 / self.settings.x_screen_size) * ramp)
        
        return np.array((chunk_l, chunk_r))

    def _sum_distances(self, index):
        current = self.settings.coords[index]
        total = 0
        for coords in self.settings.coords:
            total += self.calculate_distance(current, coords)
        return total
            
    def compute_velocity_from_entropy(self, index):
        value = 0.3
        if self.settings.people_counter > 1:
            max_value = self.calculate_distance((0, 0), (self.settings.x_screen_size, self.settings.y_screen_size)) * (self.settings.people_counter - 1)
            value = math.pow(1 - (self._sum_distances(index) / max_value), 3)
        return value

    def calculate_distance(self, A, B):
        return math.sqrt(pow((A[0] - B[0]), 2) + pow((A[1] - B[1]), 2))
