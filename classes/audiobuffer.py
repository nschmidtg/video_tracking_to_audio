import pyaudio
import scipy.signal as ss
import numpy as np
import librosa
import time
from queue import Queue
import pdb
import threading
from .settings import Settings
import math

class Stream:
    def __init__(self, path):
        self.track_data, self.track_rate = librosa.load(path, sr=44.1e3, dtype=np.float64, mono=False)
        self.last_coords_queue = Queue()
        self.last_coords_queue.put((0, 0, 0, 0))
        self.queue = Queue()
        self.stream = None

class AudioBuffer:
    def __init__(self, settings):
        self.tail = Queue()
        self.settings = settings
        self.stream_array = []
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 1-lluvia 1.wav'))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 2-lluvia 2.wav'))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 3-lluvia 3.wav'))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 4-lluvia 4.wav'))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 5-lluvia bajo lona.wav'))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 6-lluvia bosque.wav'))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 7-lluvia dentro de furgon.wav'))
        self.stream_array.append(Stream('audios/consolidado/LluviasConsolidado 8-lluvia dentro del furgon.wav'))
        self.track1_rate = self.stream_array[0].track_rate
        # instantiate PyAudio (1)
        self.p = pyaudio.PyAudio()
        
        self.first_IR, self.IR_rate = librosa.load('audios/IRs/208-R1_LargeRoom.wav', sr=44.1e3, dtype=np.float64, mono=False)
        self.chunk_size = int(44100//2)
        track1_frame = self.stream_array[0].track_data[:,0 : self.chunk_size]
        track1 = ss.fftconvolve(track1_frame, self.first_IR, mode="full", axes=1)
        self.tail.put(np.zeros(track1.shape))
        self.queue1 = Queue()
        self.people_counter = 0
        self.stream = None
        self.fft = np.fft.fft2
        self.ifft = np.fft.ifft2
        self.thread = threading.Thread(target=self.run)
        self.buffer_alive = True
        self.stream1 = None
        self.main_stream = None
        self.sample_length = int(self.chunk_size//4)
        self.samples_phase = int(self.sample_length//10)
        self.hop_length = int(self.sample_length - self.samples_phase)
        self.hanning = np.concatenate([np.linspace(0, 1, self.samples_phase//2), np.ones(self.sample_length - self.samples_phase), np.linspace(1, 0, self.samples_phase - (self.samples_phase//2))])
        self.empty_chunk = np.zeros((2, self.chunk_size))
        self.doubled_empty_chunk = np.zeros((self.chunk_size * 2))

    def process_queue(self, index):
        queue = self.stream_array[index].queue
        frames = queue.get()
        frames = frames * self.compute_velocity_from_entropy(index)
        # frames = self._apply_stereo_panning(frames, self.settings.coords[index], self.stream_array[index].last_coords_queue.get())
        self.stream_array[index].last_coords_queue.put(self.settings.coords[index])
        return frames


    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            #tail = self.tail.get()
            track = self.empty_chunk.copy()
            current_n_people = self.people_counter
            print("current_n_people", current_n_people)
            for i in range(current_n_people):
                track += self.process_queue(i) # * (1/current_n_people)
            
            # track_rev = ss.fftconvolve(track, self.first_IR, mode="full", axes=1)
            # dry_signal = np.concatenate([track, np.zeros((2, track_rev.shape[1] - track.shape[1]))], axis=1)
            # track = np.multiply(track_rev, 0.3) + np.multiply(dry_signal, 0.7)
            
            # tail_plus_track = tail + track
            # actual_tail = tail_plus_track[:, self.chunk_size:]
            # actual_chunk = tail_plus_track[:, :self.chunk_size]
            actual_combinated_chunk = self.doubled_empty_chunk.copy()
            actual_combinated_chunk[0::2] = track[0]
            actual_combinated_chunk[1::2] = track[1]
            ret_data = actual_combinated_chunk.astype(np.float32).tobytes()
            #actual_tail = np.concatenate([actual_tail, np.zeros((2, track_rev.shape[1] - actual_tail.shape[1]))], axis=1)
            #self.tail.put(actual_tail)
            # return (track.astype(np.float32).tobytes(), pyaudio.paContinue)
            return (ret_data, pyaudio.paContinue)
        return callback
    
    def start(self):
        self.thread.start()

    def run(self):
        # open stream using callback (3)
        self.stream = self.p.open(format=pyaudio.paFloat32,
                        channels=2,
                        rate=int(44100),
                        output=True,
                        stream_callback=self.get_callback(),
                        frames_per_buffer=self.chunk_size)
        
        for i in range(self.settings.max_people_counter):
            stream = self.stream_array[i]
            stream.stream = threading.Thread(target=self._call_populate_chunk, kwargs={'queue': stream.queue, 'track': stream.track_data, 'index': i})
            stream.stream.start()
        self.main_stream = threading.Thread(target=self._main_stream)
        self.main_stream.start()        

    def kill_process(self):
        self.buffer_alive = False
        for i in range(self.settings.max_people_counter):
            stream = self.stream_array[i]
            stream.stream.join()
        self.main_stream.join()

        # stop stream (6)
        self.stream.stop_stream()
        self.stream.close()

        # close PyAudio (7)
        self.p.terminate()
    
    def _call_populate_chunk(self, **kwargs):
        queue = kwargs['queue']
        track = kwargs['track']
        index = kwargs['index']
        remaining_frames = np.zeros((2, self.sample_length))
        last_coords = (0, 0)
        while self.buffer_alive:
            if queue.qsize() < 100:
                remaining_frames, last_coords = self._populate_chunk(queue, track, index, remaining_frames, last_coords)
            else:
                time.sleep(0.05)

    def _main_stream(self):
        self.stream.start_stream()

    def _populate_chunk(self, queue, track, index, remaining_frames, last_coords):
        bytes = track
        
        samples_per_chunk = 0
        current_coords = (0, 0, 0, 0) # self.settings.coords[index]
        current_stack_length = 0
        chunk = self.empty_chunk.copy()
        
        while current_stack_length < self.chunk_size:
            if current_stack_length == 0 and np.any(remaining_frames):
                chunk[:, :remaining_frames.shape[1]] += remaining_frames
                remaining_frames = np.zeros((2, self.sample_length))
            else:
                read_from = np.random.randint(0, bytes.shape[1] - self.sample_length)
                sample = bytes[:, read_from:read_from + self.sample_length]
                # sample = self._pitch_shift_2d_sample(sample,  int(0 * (1-self.compute_velocity_from_entropy(index))))
                #sample = np.multiply(sample, hanning)
                if (current_stack_length + self.sample_length <= self.chunk_size):
                    chunk[:, current_stack_length:current_stack_length + self.sample_length] += sample
                else:
                    chunk[:, current_stack_length:] += sample[:, :self.chunk_size - current_stack_length]
                    remaining_frames[:, :self.sample_length - (self.chunk_size - current_stack_length)] += sample[:, self.chunk_size - current_stack_length:]
                    
                current_stack_length += self.hop_length
            samples_per_chunk += 1
                
        # sample = self._filter_chunk(sample, index)
        # chunk = self._apply_stereo_panning(chunk, last_coords, current_coords)
        # chunk = np.multiply(chunk, 1)
        # chunk = np.multiply(chunk, 1/samples_per_chunk)
        queue.put(chunk)
        return remaining_frames, current_coords
    
    def _compute_2d_sfft(self, sample):
        return self.fft(sample)
    
    def _compute_2d_isfft(self, sample):
        return self.ifft(sample)
    
    def _pitch_shift_2d_sample(self, sample, shift):
        sample_fft = self._compute_2d_sfft(sample)
        sample_fft = np.roll(sample_fft, shift, axis=1)
        shifted = self._compute_2d_isfft(sample_fft)
        return shifted.astype(np.float32)
    
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
        # print(value)
        return value

    def calculate_distance(self, A, B):
        return math.sqrt(pow((A[0] - B[0]), 2) + pow((A[1] - B[1]), 2))

        
    def _filter_chunk(self, chunk, index):
        fc = 20000 * self.compute_velocity_from_entropy(index)
        #Design of digital filter requires cut-off frequency to be normalised by sampling_rate/2
        w = fc /(self.track1_rate/2)
        b, a = ss.butter(5, w, 'low', analog = False)
        return ss.filtfilt(b, a, chunk)

    def _intercalate_channels(self, chunk):
        return np.ravel(np.column_stack((chunk[0, :], chunk[1, :])))
