import math
import numpy as np
import pyaudio
import itertools
from operator import itemgetter
from scipy import interpolate
from scipy.signal import butter, lfilter
from scipy.signal import sawtooth, square
from matplotlib import pyplot as plt


SAMPLE_RATE = 44100
DIVS = 16


class Note:

    NOTES = ['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']

    def __init__(self, note, octave=4):
        self.octave = octave
        if isinstance(note, int):
            self.index = note
            self.note = Note.NOTES[note]
        elif isinstance(note, str):
            self.note = note.strip().lower()
            self.index = Note.NOTES.index(self.note)

    def transpose(self, halfsteps):
        octave_delta, note = divmod(self.index + halfsteps, 12)
        return Note(note, self.octave + octave_delta)

    def frequency(self):
        base_frequency = 16.35159783128741 * 2.0 ** (float(self.index) / 12.0)
        return base_frequency * (2.0 ** self.octave)

    def __float__(self):
        return self.frequency()


# TODO: chord player
class Scale:

    intervals = {'major': [2, 2, 1, 2, 2, 2, 1], 
                 'minor': [2, 1, 2, 2, 1, 2, 2],
                 }

    def __init__(self, root, sType='major'):
        self.root = Note(root.index, 0)
        self.intervals = self.intervals.get(sType)


    def get(self, index):
        intervals = self.intervals
        if index < 0:
            index = abs(index)
            intervals = reversed(self.intervals)
        intervals = itertools.cycle(self.intervals)
        note = self.root
        for i in range(index):
            note = note.transpose(next(intervals))
        return note

    def index(self, note):
        intervals = itertools.cycle(self.intervals)
        index = 0
        x = self.root
        while x.octave != note.octave or x.note != note.note:
            x = x.transpose(next(intervals))
            index += 1
        return index

    def transpose(self, note, interval):
        return self.get(self.index(note) + interval)
    

class Osc:
    def silence(freq, samples:int):
        return np.zeros(int(samples))
    
    def sine(frequency, samples:int):
        factor = float(frequency) * (math.pi * 2) / SAMPLE_RATE
        return np.sin(np.arange(samples) * factor)

    def saw(frequency, samples:int):
        factor = float(frequency) * (math.pi * 2) / SAMPLE_RATE
        return sawtooth(np.arange(samples) * factor, width=0)
    
    def square(frequency, samples:int):
        factor = float(frequency) * (math.pi * 2) / SAMPLE_RATE
        return square(np.arange(samples) * factor, duty=0.5)
    
    def tri(frequency, samples:int):
        factor = float(frequency) * (math.pi * 2) / SAMPLE_RATE
        return sawtooth(np.arange(samples) * factor, width=0.5)
    

class Tempo:
    def __init__(self, bpm:int) -> None:
        self.bpm = bpm
        self.hz = bpm / 60

        # time of div in seconds
        self.divTime = self.hz / DIVS

    def get_samples(self, divCount:int) -> int:
        return SAMPLE_RATE * self.divTime * divCount
    
    def get_samples_time(self, time:float) -> int:
        
        # # debug
        # print("time:", time)
        # print("div time:", self.divTime)
        # print("bpm:", self.bpm)
        # print("hz:", self.hz)
        # print("test:", (time / self.hz) * DIVS)

        return self.get_samples((time / self.hz) * DIVS)
        

class ADSR:

    def __init__(self, adsr:list) -> None:
        assert len(adsr) == 4
        self.adsr = adsr

    def set_for_len(self, len:float):
        A, D, S, R = self.adsr

        # normalized [0,1]
        self.A = ADSR.clamp(A / len)    # attack duration
        self.D = ADSR.clamp(D / len)    # decay duration
        self.S = ADSR.clamp(S)          # sustain level

        self.R = ADSR.clamp(ADSR.clamp(R / len))  # release duration

    def clamp(val:float, _min:float=0, _max:float=1):
        return min(max(val, _min), _max)

    def gen_envelope(self, data, length):
        self.set_for_len(length)
        decayEnd = self.A + self.D
        delta = 1e-6
        releaseBegin = ADSR.clamp(1 - self.R, decayEnd, 1) - delta

        #           _           /             \-                -\              _
        points = {0.0:0.0, self.A:1.0, decayEnd:self.S, releaseBegin:self.S, 1.0:0.0}

        # # debug
        # plt.plot(list(points.keys()), list(points.values()))
        # plt.show()
        # print(points)
        
        return ADSR.shape(data, points)

    def shape(data, points:dict, kind='slinear'):
        items = points.items()
        sorted(items,key=itemgetter(0))
        keys = list(map(itemgetter(0), items))
        vals = list(map(itemgetter(1), items))

        interp = interpolate.interp1d(keys, vals, kind=kind)
        factor = 1.0 / len(data)
        shape = interp(np.arange(len(data)) * factor)

        return data * shape


class Filter:
    def __init__(self, cutoff:float, bType:str='low') -> None:
        assert 0 < cutoff < 1
        self.b, self.a = butter(8, cutoff, btype=bType)
    
    def apply(self, data):
        return lfilter(self.b, self.a, data)


class Synth:
    def __init__(self, wave, harm:int, adsr:list, tempo:Tempo, cutoff:int, volume:float) -> None:
        self.wave = wave
        self.harm = max(0, harm)
        self.volume = max(0, volume)
        self.adsr = ADSR(adsr)
        self.filt = Filter(cutoff)
        self.tempo = tempo

    # TODO: change to unison voices instead of harmonics
    def harmonics(self, freq, time):
        harms = []
        for i in range(self.harm + 1):
            freqMod = 2**(i+1)
            volMod = 0.5**(i+1)
            # print("freqmod:", freqMod)
            # print("volmod:", volMod)
            
            harms.append(self.wave(freq * freqMod, self.tempo.get_samples_time(time)) * self.volume * volMod)
        return sum(harms)

        # single
        # return self.wave(freq, self.tempo.get_samples_time(time)) * self.volume
    
    def harmonics_samp(self, freq, samp):
        harms = []
        for i in range(self.harm + 1):
            freqMod = 2**(i+1)
            volMod = 0.5**(i+1)
            # print("freqmod:", freqMod)
            # print("volmod:", volMod)
            
            harms.append(self.wave(freq * freqMod, samp) * self.volume * volMod)
        return sum(harms)

    def gen_signal(self, freq, time):
        chunk = self.harmonics(freq, time)
        chunk = self.filt.apply(chunk)
        return self.adsr.gen_envelope(chunk, time)
    
    def gen_signal_samp(self, freq, time, samp):
        chunk = self.harmonics_samp(freq, samp)
        chunk = self.filt.apply(chunk)
        return self.adsr.gen_envelope(chunk, time)


#===============================================================


t = Tempo(120)
s = Synth(Osc.sine, 
          harm=1,
          adsr=[0.01, 0.0, 1.0, 0.01],
          tempo=t,
          cutoff=0.7,
          volume=0.2
          )

b = Synth(Osc.saw,
          harm=0,
          adsr=[0.01, 0.1, 0.1, 0.01],
          tempo=t,
          cutoff=0.9,
          volume=0.4          
          )

l = Synth(Osc.square,
          harm=1,
          adsr=[0.7, 0.0, 1.0, 0.01],
          tempo=t,
          cutoff=0.9,
          volume=0.2
          )

r = Synth(Osc.silence,
          harm=0,
          adsr=[0.0, 0.0, 0.0, 0.0],
          tempo=t,
          cutoff=0.9,
          volume=0.2
          )

CM = Scale(Note('C', octave=4), 'major')
Cm = Scale(Note('C#', octave=4), 'minor')

pads = []
bass = []
lead = []

# Pads
notelen = 4
pads.append(s.gen_signal(Note('C', octave=3).frequency(), notelen) + 
            s.gen_signal(Note('E', octave=3).frequency(), notelen) + 
            s.gen_signal(Note('G', octave=3).frequency(), notelen) + 
            s.gen_signal(Note('C', octave=4).frequency(), notelen))

# Bass
for _ in range(8):
    bass.append(b.gen_signal(Note('C', octave=1).frequency(), notelen/16))
    bass.append(b.gen_signal(Note('C', octave=2).frequency(), notelen/16))

# Lead
lead.append(l.gen_signal(Note('C', octave=3).frequency(), 0.5))
lead.append(l.gen_signal(Note('E', octave=3).frequency(), 0.5))
lead.append(l.gen_signal(Note('G', octave=3).frequency(), 0.5))
lead.append(l.gen_signal(Note('C', octave=3).frequency(), 2.5) + 
            l.gen_signal(Note('C', octave=4).frequency(), 2.5))

lead = np.concatenate(lead)
lead = lead.reshape(1, len(lead))


# Combine signals
bass = np.array(bass).flatten()
bass = bass.reshape(1, len(bass))

print("pads", np.shape(pads))
print("bass", np.shape(bass))
print("lead", np.shape(lead))

shortest = min(np.array(pads).size, np.array(bass).size, np.array(lead).size)
print("shortest", shortest)

pads = np.asarray(pads).flatten()[:shortest].reshape(1, shortest)
bass = np.asarray(bass).flatten()[:shortest].reshape(1, shortest)
lead = np.asarray(lead).flatten()[:shortest].reshape(1, shortest)

print('-'*20)
print("pads", np.shape(pads))
print("bass", np.shape(bass))
print("lead", np.shape(lead))

# Add together
chunks = pads + bass + lead


# Play stuff
chunk = np.concatenate(chunks) * 0.25
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, output=1)
stream.write(chunk.astype(np.float32).tobytes())
stream.close()
p.terminate()




