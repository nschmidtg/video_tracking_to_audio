import pyaudio
import scipy.signal as ss
import numpy as np
import librosa
import time
from queue import Queue
import threading
import math
import os


class Control:
    def __init__(self):
        self.last_value = [0, 0, 0, 0]
        self.queue = Queue(1)

    def add_value(self, value):
        if self.queue.qsize() > 0:
            self.queue.get()
        self.queue.put(value)

    def get_position_for_coordinate(self, chunk_size, width):
        current_value = []
        if self.queue.qsize() > 0:
            current_value = self.queue.get()
        else:
            current_value = self.last_value
        current_coordinate = current_value[0]
        last_coordinate = self.last_value[0]
        max_jump = int(width/14)
        
        if np.abs(current_coordinate - last_coordinate) > max_jump:
            current_coordinate = last_coordinate + np.sign(current_coordinate - last_coordinate) * max_jump
        array = np.linspace(last_coordinate, current_coordinate, chunk_size)
        self.last_value = [current_coordinate, current_value[1], current_value[2], current_value[3]]
        return array


class RampHandler():
    def __init__(self, ramp_length):
        self.ramp_length = ramp_length
        self.ramp_up = np.linspace(0, 1, self.ramp_length)
        self.ramp_down = np.linspace(1, 0, self.ramp_length)
        self.current_y_value = 0
        self.current_fading = 'none'

    def get_next_fade(self, chunk_size, fading):
        ramp = np.zeros(chunk_size)
        if fading == 'in':
            current_x_position = int(self.current_y_value * self.ramp_length)
            if current_x_position + chunk_size >= self.ramp_length:
                ramp = np.concatenate([self.ramp_up[current_x_position:], np.ones(chunk_size - (self.ramp_length - current_x_position))], axis=0)
                self.current_fading = 'playing'
            else:
                ramp = self.ramp_up[current_x_position:current_x_position + chunk_size]
                self.current_fading = 'in'
            self.current_y_value = ramp[-1]
        elif fading == 'out':
            current_x_position = int((1 - self.current_y_value) * self.ramp_length)
            if current_x_position + chunk_size >= self.ramp_length:
                ramp = np.concatenate([self.ramp_down[current_x_position:], np.zeros(chunk_size - (self.ramp_length - current_x_position))], axis=0)
                self.current_fading = 'none'
            else:
                ramp = self.ramp_down[current_x_position:current_x_position + chunk_size]
                self.current_fading = 'out'
            self.current_y_value = ramp[-1]
        elif fading == 'playing':
            ramp = np.ones(chunk_size)
        elif fading == 'none':
            ramp = np.zeros(chunk_size)
            
        self.current_y_value = ramp[-1]
            
        return ramp


class Stream(threading.Thread):
    def __init__(self, path, chunk_size, screen_width, screen_height, linear=False, static_ambient=False):
        self.static_ambient = static_ambient
        self.track_data, self.track_rate = librosa.load(path, sr=44.1e3, dtype=np.float64, mono=False)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.coordinates = Control()
        self.queue = Queue()
        self.buffer_alive = True
        self.chunk_size = chunk_size
        self.ramp_handler = RampHandler(self.chunk_size * 60)
        self.empty_chunk = np.zeros((2, chunk_size))
        self.sample_length = int(chunk_size//2)
        self.empty_sample = np.zeros((2, self.sample_length))
        self.fade_length = 0
        self.hop_length = self.sample_length - self.fade_length
        self.populate_function = self._populate_linear_chunk if linear else self._populate_random_chunk
        self.read_from = 0
        self.ramp_last_value = self.screen_width//2
        self.random = np.random.RandomState(0)
        threading.Thread.__init__(self)
        
    def join(self):
        self.buffer_alive = False
        super().join()
        
    def reset_random(self):
        self.read_from = self.random.randint(0, self.track_data.shape[1] - self.chunk_size)
        
    def run(self):
        remaining_frames = self.empty_sample.copy()
        self.reset_random()
        while self.buffer_alive:
            if self.queue.qsize() < 5:
                remaining_frames = self.populate_function(remaining_frames)
            else:
                time.sleep(0.01)
                
    def _populate_random_chunk(self, remaining_frames):        
        samples_per_chunk = 0
        current_stack_length = 0
        chunk = self.empty_chunk.copy()
        
        while current_stack_length < self.chunk_size:
            if current_stack_length == 0 and np.any(remaining_frames):
                chunk[:, :remaining_frames.shape[1]] += remaining_frames
                remaining_frames = self.empty_sample.copy()
            else:
                read_from = self.random.randint(0, self.track_data.shape[1] - self.sample_length)
                sample = self.track_data[:, read_from:read_from + self.sample_length]
                if (current_stack_length + self.sample_length <= self.chunk_size):
                    chunk[:, current_stack_length:current_stack_length + self.sample_length] += sample
                else:
                    chunk[:, current_stack_length:] += sample[:, :self.chunk_size - current_stack_length]
                    remaining_frames[:, :self.sample_length - (self.chunk_size - current_stack_length)] += sample[:, self.chunk_size - current_stack_length:]
                    
                current_stack_length += self.hop_length
            samples_per_chunk += 1
            
        chunk = self._apply_stereo_panning(chunk)

        self.queue.put(chunk)
        return remaining_frames
    
    def _populate_linear_chunk(self, remaining_frames):        
        current_stack_length = 0
        chunk = self.empty_chunk.copy()
        if self.read_from + self.chunk_size > self.track_data.shape[1]:
            self.reset_random()
        read_from = self.read_from
        while current_stack_length < self.chunk_size:
            if current_stack_length == 0 and np.any(remaining_frames):
                chunk[:, :remaining_frames.shape[1]] += remaining_frames
                remaining_frames = self.empty_sample.copy()
            else:
                sample = self.track_data[:, read_from:read_from + self.sample_length]
                if (current_stack_length + self.sample_length <= self.chunk_size):
                    chunk[:, current_stack_length:current_stack_length + self.sample_length] += sample
                else:
                    chunk[:, current_stack_length:] += sample[:, :self.chunk_size - current_stack_length]
                    remaining_frames[:, :self.sample_length - (self.chunk_size - current_stack_length)] += sample[:, self.chunk_size - current_stack_length:]
                    
                current_stack_length += self.sample_length
                read_from += self.sample_length

        self.queue.put(chunk)
        self.read_from = read_from
        return remaining_frames
    
    def _apply_stereo_panning(self, chunk):
        if self.ramp_handler.current_fading == 'playing' or self.ramp_handler.current_fading == 'in':
            ramp = self.coordinates.get_position_for_coordinate(chunk.shape[1], self.screen_width)
            chunk_r = np.multiply(chunk[1, :], (1 / self.screen_width) * ramp)
            chunk_l = np.multiply(chunk[0, :], 1-(1 / self.screen_width) * ramp)
            self.ramp_last_value = ramp[-1]
        else:
            chunk_r = np.multiply(chunk[1, :], (1 / self.screen_width) * self.ramp_last_value)
            chunk_l = np.multiply(chunk[0, :], 1-(1 / self.screen_width) * self.ramp_last_value)
            self.coordinates.last_value[0] = self.ramp_last_value
        
        return np.array((chunk_l, chunk_r))
    
    def compute_area(self):
        coordinates = self.coordinates.last_value
        return coordinates[2] * coordinates[3]
    
    def compute_centroid(self):
        coordinates = self.coordinates.last_value
        return (coordinates[0] + coordinates[2]/2, coordinates[1] + coordinates[3]/2)
    
class AudioBuffer(threading.Thread):
    def __init__(self, screen_width, screen_height, max_n_people):
        self.tail = Queue()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_n_people = max_n_people
        self.stream_array = []
        self.chunk_size = int(1100*4) # 4400 100ms chord 
        self.base_stream = Stream('audios/Final/base1.wav', self.chunk_size, screen_width, screen_height, linear=True, static_ambient=True)
        self.stream_array.append(Stream('audios/Final/A1.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/Final/G3.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/Final/E3.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/Final/C3.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/Final/A3.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/Final/G2.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/Final/E2.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/Final/base2.wav', self.chunk_size, screen_width, screen_height, linear=True, static_ambient=False))
        self.stream_array.append(Stream('audios/Final/C2.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/Final/A2.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/Final/G1.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/Final/E1.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/Final/C1.wav', self.chunk_size, screen_width, screen_height))
        
        self.p = pyaudio.PyAudio()
        self.first_IR, self.IR_rate = librosa.load('audios/IRs/301-LargeHall.wav', sr=44.1e3, dtype=np.float64, mono=False)
        track1_frame = self.stream_array[0].track_data[:,0 : self.chunk_size]
        self.track1 = ss.fftconvolve(track1_frame, self.first_IR, mode="full", axes=1)
        self.reverb_empty_track = np.zeros((2, self.track1.shape[1]))
        self.tail.put(np.zeros(self.track1.shape))
        self.people_counter = 0
        self.empty_chunk = np.zeros((2, self.chunk_size))
        self.doubled_empty_chunk = np.zeros((self.chunk_size * 2))
        self.last_velocity_value = 0
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=int(44100),
            output=True,
            stream_callback=self.get_callback(),
            frames_per_buffer=self.chunk_size,
            # output_device_index=5,
        )
        threading.Thread.__init__(self)

    def process_queue(self, stream, fading):
        frames = stream.queue.get()
        if stream.static_ambient is False:
            frames = frames * stream.ramp_handler.get_next_fade(self.chunk_size, fading)
        return frames
    
    def _apply_reverb(self, chunk, wet_level=0.2):
        track_rev = ss.fftconvolve(chunk, self.first_IR, mode="full", axes=1)
        dry_track_with_zeros = self.reverb_empty_track.copy()
        dry_track_with_zeros[:, :self.chunk_size] = chunk
        tail = self.tail.get()
        track = np.multiply(track_rev, wet_level) + np.multiply(dry_track_with_zeros, 1-wet_level)
        tail_plus_track = tail + track
        actual_tail = tail_plus_track[:, self.chunk_size:]
        actual_chunk = tail_plus_track[:, :self.chunk_size]
        empty_tail = self.reverb_empty_track.copy()
        empty_tail[:,:actual_tail.shape[1]] = actual_tail
        self.tail.put(empty_tail)
        return actual_chunk
        
    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            track = self.empty_chunk.copy()
            current_n_people = self.people_counter
            for i in range(self.max_n_people):
                current_stream = self.stream_array[i]
                fading = current_stream.ramp_handler.current_fading
                if i < current_n_people:
                    if current_stream.ramp_handler.current_fading == 'none':
                        fading = 'in'
                else:
                    if current_stream.ramp_handler.current_fading != 'none':
                        fading = 'out'
                if fading != 'none':
                    track += self.process_queue(current_stream, fading)
            track += self.process_queue(self.base_stream, 'playing')
            track = self.apply_velocity_from_entropy2(track)
            track = self._apply_reverb(track, 0.5)
            actual_combinated_chunk = self._intercalate_channels(track)
            ret_data = actual_combinated_chunk.astype(np.float32).tobytes()
            return (ret_data, pyaudio.paContinue)
        return callback
    
    def start(self):
        self.base_stream.start()
        for i in range(self.max_n_people):
            stream = self.stream_array[i]
            stream.start()
        super().start()

    def run(self):
        self.stream.start_stream()
        
    def join(self):
        for i in range(self.max_n_people):
            stream = self.stream_array[i]
            stream.join()
        self.base_stream.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        super().join()
    
    def _intercalate_channels(self, chunk):
        return np.ravel(np.column_stack((chunk[0, :], chunk[1, :])))
            
    def compute_velocity_from_entropy(self):
        value = 0.3
        playing = 1
        for stream in self.stream_array:
            if stream.ramp_handler.current_fading != 'none':
                playing += 1
        return max(math.pow(playing / self.max_n_people + value,3), 1)
    
    def apply_velocity_from_entropy2(self, track):
        points = []
        for stream in self.stream_array:
            if stream.ramp_handler.current_fading != 'none':
                points.append(stream.compute_centroid())
        total_distance = self.sum_of_distances(points)
        if len(points) == 0 or total_distance == 0:
            current_velocity = 0.2
            track = np.multiply(track, np.linspace(self.last_velocity_value, current_velocity, track.shape[1]))
            self.last_velocity_value = current_velocity
            return track
        max_distance = self.max_sum_of_distances(len(points), self.screen_width, self.screen_height)
        current_velocity = math.pow(min(1 - total_distance / max_distance, 1), 2)
        track = np.multiply(track, np.linspace(self.last_velocity_value, current_velocity, track.shape[1]))
        self.last_velocity_value = current_velocity
        return track
        
    def max_sum_of_distances(self, N, w, h):
        # Calculate the perimeter of the square
        perimeter = 2 * (w + h)
        # Calculate the distance between each pair of points
        distance_between_points = perimeter / N
        # Calculate the sum of distances
        sum_of_distances = 0
        for i in range(1, N):
            sum_of_distances += i * (N - i) * distance_between_points

        return sum_of_distances

    def calculate_distance(self, A, B):
        return math.sqrt(pow((A[0] - B[0]), 2) + pow((A[1] - B[1]), 2))
    
    def sum_of_distances(self, points):
        total_distance = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                total_distance += self.calculate_distance(points[i], points[j])
        return total_distance




