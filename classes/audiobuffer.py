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
        print(0, current_coordinate, last_coordinate, width)
        array = np.linspace(last_coordinate, current_coordinate, chunk_size)
        self.last_value = [current_coordinate, current_value[1], current_value[2], current_value[3]]
        return array


class RampHandler():
    def __init__(self, ramp_length):
        self.ramp_length = ramp_length
        self.ramp_up = np.linspace(0, 1, self.ramp_length)
        self.ramp_down = np.linspace(1, 0, self.ramp_length)
        self.current_y_value = 1
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
    def __init__(self, path, chunk_size, screen_width, screen_height, linear=False):
        self.track_data, self.track_rate = librosa.load(path, sr=44.1e3, dtype=np.float64, mono=False)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.coordinates = Control()
        self.queue = Queue()
        self.buffer_alive = True
        self.chunk_size = chunk_size
        self.ramp_handler = RampHandler(self.chunk_size * 250)
        self.empty_chunk = np.zeros((2, chunk_size))
        self.sample_length = int(chunk_size//2)
        self.hop_length = int(self.sample_length//10)
        self.populate_function = self._populate_linear_chunk if linear else self._populate_random_chunk
        self.read_from = 0
        self.ramp_last_value = self.screen_width//2
        threading.Thread.__init__(self)
        
    def join(self):
        self.buffer_alive = False
        super().join()
        
    def reset_random(self):
        self.read_from = np.random.randint(0, self.track_data.shape[1] - self.chunk_size)
        
    def run(self):
        remaining_frames = np.zeros((2, self.sample_length))
        self.reset_random()
        while self.buffer_alive:
            if self.queue.qsize() < 5:
                remaining_frames = self.populate_function(remaining_frames)
            else:
                time.sleep(0.005)
                
    def _populate_random_chunk(self, remaining_frames):        
        samples_per_chunk = 0
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
                remaining_frames = np.zeros((2, self.sample_length))
            else:
                sample = self.track_data[:, read_from:read_from + self.sample_length]
                if (current_stack_length + self.sample_length <= self.chunk_size):
                    chunk[:, current_stack_length:current_stack_length + self.sample_length] += sample
                else:
                    chunk[:, current_stack_length:] += sample[:, :self.chunk_size - current_stack_length]
                    remaining_frames[:, :self.sample_length - (self.chunk_size - current_stack_length)] += sample[:, self.chunk_size - current_stack_length:]
                    
                current_stack_length += self.sample_length
                read_from += self.sample_length

        chunk = self._apply_stereo_panning(chunk)
        self.queue.put(chunk)
        self.read_from = read_from
        return remaining_frames
    
    def _apply_stereo_panning(self, chunk):
        print(self.ramp_handler.current_fading, self.coordinates.last_value)
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

class AudioBuffer(threading.Thread):
    def __init__(self, screen_width, screen_height, max_n_people):
        self.tail = Queue()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_n_people = max_n_people
        self.stream_array = []
        self.chunk_size = int(1100)
        self.stream_array.append(Stream('audios/consolidado/base1.wav', self.chunk_size, screen_width, screen_height, linear=True))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidadoHarm A.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidadoHarm A-1.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidadoHarm A-2.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidadoHarm C.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidadoHarm C-1.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/consolidado/base2.wav', self.chunk_size, screen_width, screen_height, linear=True))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidadoHarm C-2.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidadoHarm E.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidadoHarm E-1.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidadoHarm E-2.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidadoHarm G.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidadoHarm G-1.wav', self.chunk_size, screen_width, screen_height))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidadoHarm G-2.wav', self.chunk_size, screen_width, screen_height))
        

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

    def process_queue(self, stream, fading):
        frames = stream.queue.get()
        frames = frames * stream.ramp_handler.get_next_fade(self.chunk_size, fading)
        return frames
    
    def _apply_reverb(self, chunk, wet_level=0.2):
        track_rev = ss.fftconvolve(chunk, self.first_IR, mode="full", axes=1)
        dry_track_with_zeros = np.concatenate([chunk, np.zeros((2, track_rev.shape[1] - chunk.shape[1]))], axis=1)
        tail = self.tail.get()
        track = np.multiply(track_rev, wet_level) + np.multiply(dry_track_with_zeros, 1-wet_level)
        tail_plus_track = tail + track
        actual_tail = tail_plus_track[:, self.chunk_size:]
        actual_chunk = tail_plus_track[:, :self.chunk_size]
        actual_tail = np.concatenate([actual_tail, np.zeros((2, track_rev.shape[1] - actual_tail.shape[1]))], axis=1)
        self.tail.put(actual_tail)
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
                track += self.process_queue(current_stream, fading)
            track = track * self.compute_velocity_from_entropy()
            track = self._apply_reverb(track, 0.4)
            actual_combinated_chunk = self._intercalate_channels2(track)
            ret_data = actual_combinated_chunk.astype(np.float32).tobytes()
            return (ret_data, pyaudio.paContinue)
        return callback
    
    def start(self):
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
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        super().join()
    
    def _intercalate_channels(self, chunk):
        return np.ravel(np.column_stack((chunk[0, :], chunk[1, :])))
    
    def _intercalate_channels2(self, chunk):
        to_be_populated = self.doubled_empty_chunk.copy()
        to_be_populated[::2] = chunk[0, :]
        to_be_populated[1::2] = chunk[1, :]
        return to_be_populated

    def _sum_distances(self, index):
        current = self.stream_array[index].coordinates.last_value
        total = 0
        for stream in self.stream_array:
            total += self.calculate_distance(current, coords)
        return total
            
    def compute_velocity_from_entropy(self):
        value = 0.3
        # if self.people_counter > 1:
        #     max_value = self.calculate_distance((0, 0), (self.screen_width, self.screen_height)) * (self.people_counter - 1)
        #     value = math.pow(1 - (self._sum_distances(index) / max_value), 3)
        return value

    def calculate_distance(self, A, B):
        return math.sqrt(pow((A[0] - B[0]), 2) + pow((A[1] - B[1]), 2))
