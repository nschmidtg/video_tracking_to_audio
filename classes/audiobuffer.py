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



class AudioBuffer:
    def __init__(self, settings):
        self.tail = Queue()
        self.last_coords_queue1 = Queue()
        self.last_coords_queue2 = Queue()
        self.last_coords_queue3 = Queue()
        self.last_coords_queue4 = Queue()
        self.settings = settings
        self.track1_data, self.track1_rate = librosa.load('audios/larga_duracion/Lluvias/lluvia 1.WAV', sr=96000, dtype=np.float64, mono=False)
        self.track2_data, self.track2_rate = librosa.load('audios/larga_duracion/Lluvias/lluvia 2.WAV', sr=96000, dtype=np.float64, mono=False)
        self.track3_data, self.track3_rate = librosa.load('audios/larga_duracion/Lluvias/lluvia 3.WAV', sr=96000, dtype=np.float64, mono=False)
        self.track4_data, self.track4_rate = librosa.load('audios/larga_duracion/Lluvias/lluvia 4.WAV', sr=96000, dtype=np.float64, mono=False)
        print("self.track1_data.shape", self.track1_data.shape)
        # instantiate PyAudio (1)
        self.p = pyaudio.PyAudio()
        
        self.first_IR, self.IR_rate = librosa.load('audios/IRs/314-Cathedral.wav', sr=44.1e3, dtype=np.float64, mono=False)
        print("self.first_IR.shape[1]", self.first_IR.shape[1])
        self.chunk_size = int(44100*0.5)
        track1_frame = self.track1_data[:, 0:self.chunk_size]
        track1 = ss.fftconvolve(track1_frame, self.first_IR, mode="full", axes=1)
        self.tail.put(np.zeros(track1.shape))
        self.last_coords_queue1.put((0, 0, 0, 0))
        self.last_coords_queue2.put((0, 0, 0, 0))
        self.last_coords_queue3.put((0, 0, 0, 0))
        self.last_coords_queue4.put((0, 0, 0, 0))
        self.queue1 = Queue()
        self.queue2 = Queue()
        self.queue3 = Queue()
        self.queue4 = Queue()
        self.people_counter = 0
        self.stream = None
        self.fft = np.fft.fft2
        self.ifft = np.fft.ifft2
        self.thread = threading.Thread(target=self.run)
        self.queue_array = [self.queue1, self.queue2, self.queue3, self.queue4]
        self.last_coords_queue_array = [self.last_coords_queue1, self.last_coords_queue2, self.last_coords_queue3, self.last_coords_queue4]
        self.buffer_alive = True
        self.stream1 = None
        self.stream2 = None
        self.stream3 = None
        self.stream4 = None
        self.main_stream = None
        self.fading_out = False
        self.fade_out_window = np.linspace(1,0,5 * self.chunk_size)
        self.current_fade_out = 0


    def _fade_out(self, frames_l, frames_r):
        start = self.current_fade_out
        end = self.current_fade_out + len(frames_l)
        frames_l = np.multiply(frames_l, self.fade_out_window[start:end])
        frames_r = np.multiply(frames_r, self.fade_out_window[start:end])
        self.current_fade_out += len(frames_l)
        if self.current_fade_out >= len(self.fade_out_window):
            self.fading_out = False
            self.current_fade_out = 0
        return frames_l, frames_r

    def _apply_stereo_panning(self, chunk, last_coords, current_coords):
        ramp = np.linspace(last_coords[0], current_coords[0], chunk.shape[1])
        chunk_l = np.multiply(chunk[0, :], 1-(1 / self.settings.x_screen_size) * ramp)
        chunk_r = np.multiply(chunk[1, :], (1 / self.settings.x_screen_size) * ramp)
        if self.fading_out:
            chunk_l, chunk_r = self._fade_out(chunk_l, chunk_r)
        return np.array((chunk_l, chunk_r))


    def process_queue(self, index):
        queue = self.queue_array[index]
        frames = queue.get()
        frames = frames * self.compute_velocity_from_entropy(index)
        frames = self._apply_stereo_panning(frames, self.settings.coords[index], self.last_coords_queue_array[index].get())
        self.last_coords_queue_array[index].put(self.settings.coords[index])
        return frames


    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            tail = self.tail.get()
            track = np.zeros((2, self.chunk_size))
            current_n_people = self.people_counter
            for i in range(current_n_people):
                track += self.process_queue(i) * (1/current_n_people) 
            
            track_rev = ss.fftconvolve(track, self.first_IR, mode="full", axes=1)
            track = track_rev * 0.8 + 0.2*np.concatenate([track, np.zeros((2, track_rev.shape[1] - track.shape[1]))], axis=1)
            
            tail_plus_track = (1/2 * tail + 1/2 * track) * 5
            actual_tail = tail_plus_track[:, self.chunk_size:]
            actual_chunk = tail_plus_track[:, :self.chunk_size]
            actual_combinated_chunk = np.zeros((2*self.chunk_size))
            actual_combinated_chunk[0::2] = actual_chunk[0]
            actual_combinated_chunk[1::2] = actual_chunk[1]
            ret_data = actual_combinated_chunk.astype(np.float32).tobytes()
            actual_tail = np.concatenate([actual_tail, np.zeros((2, track_rev.shape[1] - actual_tail.shape[1]))], axis=1)
            self.tail.put(actual_tail)
            # return (track.astype(np.float32).tobytes(), pyaudio.paContinue)
            return (ret_data, pyaudio.paContinue)
        return callback
    
    def start(self):
        self.thread.start()

    def run(self):
        # open stream using callback (3)
        self.stream = self.p.open(format=pyaudio.paFloat32,
                        channels=2,
                        rate=int(self.track1_rate),
                        output=True,
                        stream_callback=self.get_callback(),
                        frames_per_buffer=self.chunk_size)
        
        self.stream1 = threading.Thread(target=self._call_populate_chunk, kwargs={'queue': self.queue1, 'track': self.track1_data, 'index': 0})
        self.stream2 = threading.Thread(target=self._call_populate_chunk, kwargs={'queue': self.queue2, 'track': self.track2_data, 'index': 1})
        self.stream3 = threading.Thread(target=self._call_populate_chunk, kwargs={'queue': self.queue3, 'track': self.track3_data, 'index': 2})
        self.stream4 = threading.Thread(target=self._call_populate_chunk, kwargs={'queue': self.queue4, 'track': self.track4_data, 'index': 3})
        self.stream1.start()
        self.stream2.start()
        self.stream3.start()
        self.stream4.start()
        self.main_stream = threading.Thread(target=self._main_stream)
        self.main_stream.start()        

    def kill_process(self):
        self.buffer_alive = False
        self.stream1.join()
        self.stream2.join()
        self.stream3.join()
        self.stream4.join()
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

        sample_length = int(self.chunk_size//2)
        remaining_frames = np.zeros((2, sample_length))
        last_coords = (0, 0)
        samples_phase =  int(sample_length//3*2)
        hop_length = int(sample_length - samples_phase)
        while self.buffer_alive:
            if queue.qsize() < 2:
                print("queue_size: ", queue.qsize())
                remaining_frames, last_coords = self._populate_chunk(queue, track, index, remaining_frames, last_coords, sample_length, samples_phase, hop_length)
            time.sleep(0.1)


    def _main_stream(self):
        self.stream.start_stream()

    def _populate_chunk(self, queue, track, index, remaining_frames, last_coords, sample_length, samples_phase, hop_length):
        bytes = track
        hanning = np.concatenate([np.linspace(0, 1, samples_phase//2), np.ones(sample_length - samples_phase), np.linspace(1, 0, samples_phase - (samples_phase//2))])
        samples_per_chunk = 0
        current_coords = self.settings.coords[0]
        current_stack_length = 0
        chunk = np.zeros((2, self.chunk_size))
        
        while current_stack_length < self.chunk_size:
            if current_stack_length == 0 and np.any(remaining_frames):
                chunk[:, :remaining_frames.shape[1]] += remaining_frames
                remaining_frames = np.zeros((2, sample_length))
            else:
                read_from = np.random.randint(0, bytes.shape[1] - sample_length)
                sample = bytes[:, read_from:read_from + sample_length]
                sample = self._pitch_shift_2d_sample(sample,  int(0 * (1-self.compute_velocity_from_entropy(index))))
                sample = np.multiply(sample, hanning)
                if (current_stack_length + sample_length <= self.chunk_size):
                    chunk[:, current_stack_length:current_stack_length + sample_length] += sample
                else:
                    chunk[:, current_stack_length:] += sample[:, :self.chunk_size - current_stack_length]
                    remaining_frames[:, :sample_length - (self.chunk_size - current_stack_length)] += sample[:, self.chunk_size - current_stack_length:]
                current_stack_length += hop_length
            samples_per_chunk += 1
                
        sample = self._filter_chunk(sample, index)
        #chunk = self._apply_stereo_panning(chunk, last_coords, current_coords)
        # chunk = np.multiply(chunk, 1)
        chunk = np.multiply(chunk, 1/samples_per_chunk)
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
            value = math.pow(1 - (self._sum_distances(index) / max_value), 2)
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
