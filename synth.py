import math
import numpy as np
import pandas as pd
import pyaudio
import itertools
from operator import itemgetter
from scipy import interpolate
from scipy.signal import butter, lfilter
from scipy.signal import sawtooth, square
# from matplotlib import pyplot as plt
from sys import maxsize as INTEGER_MAX


SAMPLE_RATE = 44100
DIVS = 16
MASTER_VOL = 4.0


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


# TODO: chord player based on a chord???
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


# TODO: fix tempo to actually give correct time divisions
class Tempo:
    def __init__(self, bpm:int) -> None:
        self.bpm = bpm
        self.hz = bpm / 60

        # time of div in seconds
        self.divTime = self.hz / DIVS

    # Return the number samples for the given divcount
    def get_samples(self, divcount:int) -> int:
        return SAMPLE_RATE * self.divTime * divcount
    
    # Return the time elapsed for the given divcount
    def get_time(self, samples:int) -> float:
        return samples / SAMPLE_RATE
        

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
        self.b, self.a = butter(16, cutoff, btype=bType)
    
    def apply(self, data):
        return lfilter(self.b, self.a, data)


class Synth:

    # vtype = harmonic | unison
    def __init__(self, name:str, wave, voices:int, vtype:str, adsr:list, tempo:Tempo, cutoff:int, volume:float) -> None:
        self.name = name
        self.wave = wave
        self.voices = max(0, voices)
        self.vtype = vtype
        self.volume = max(0, volume) * MASTER_VOL
        self.adsr = ADSR(adsr)
        self.filt = Filter(ADSR.clamp(cutoff, 0.1, 0.9))
        self.tempo = tempo

    def unison(self, freq, samp):
        if self.voices == 0:
            return self.wave(freq, samp) * self.volume * 0.5
        
        unison = []
        
        spreadMod = 1
        polarity = 1

        for i in range(self.voices):
            freqMod = freq * 0.003 * spreadMod
            volMod = 0.2
            # print(f'freq: {freq}\tfreq mod: {freqMod}\tvoice: {freq + (freqMod * polarity)}')
            unison.append(self.wave(freq + (freqMod * polarity), samp) * self.volume * volMod)

            if i % 2 == 0:
                spreadMod += 1
            polarity *= -1

        return sum(unison)
        
    def harmonics(self, freq, samp):
        if self.voices == 0:
            return self.wave(freq, samp) * self.volume * 0.5
        
        harms = []
        for i in range(self.voices + 1):
            freqMod = 2**(i+1)
            volMod = 0.5**(i+1)
            harms.append(self.wave(freq * freqMod, samp) * self.volume * volMod)

        return sum(harms)
    
    def gen_signal(self, freq, samp):
        if self.vtype == 'harmonic':
            chunk = self.harmonics(freq, samp)
        elif self.vtype == 'unison':
            chunk = self.unison(freq, samp)

        chunk = self.filt.apply(chunk)
        return self.adsr.gen_envelope(chunk, self.tempo.get_time(samp))


class NoteGenerator:
    def __init__(self, filepath:str, sheetname:str, t:Tempo) -> None:
        self.data = pd.read_excel(filepath, sheet_name=sheetname)
        self.silence = Synth(name='silence',
                             wave=Osc.silence,
                             voices=0,
                             vtype='harmonic',
                             adsr=[0.0, 0.0, 0.0, 0.0],
                             tempo=t,
                             cutoff=0.9,
                             volume=0.2
                             )
        self.t = t

    # return note, oct
    def parse(self, note:str):
        if len(note) == 2:
            return note[0], int(note[1])
        elif len(note) == 3:
            return note[0:2], int(note[2])

        else:
            return None, None
    

    def gen_notes(self, col:pd.DataFrame, synth:Synth) -> list:
        
        # col = self.data.loc[:,colName]        # OLD
        out = []
        divcount = 1
        current:Note = None

        i = 0
        while i < len(col):
            # Append silence (empty cell)
            if str(col[i]) == 'nan':
                i += 1
                out.append(self.silence.gen_signal(0, self.t.get_samples(1)))
                continue
                
            # Get note / octave
            note, oct = self.parse(col[i])
            # print(f'note:{note}\toct:{oct}')

            if note != None:
                current = Note(note, oct)

            # Get sustain length
            i += 1
            while i < len(col) and col[i] == '|':
                divcount += 1
                i += 1
                
            out.append(synth.gen_signal(current.frequency(), self.t.get_samples(divcount)))

            # Reset divcount
            divcount = 1

        # Organise output
        out = np.concatenate(out).flatten()
        sampLen = len(out)
        out = out.reshape(1, sampLen)

        return out, sampLen
    

    def get(self, synths:list) -> list:
        chunks = []
        shortest = INTEGER_MAX

        # Generate notes
        for synth in synths:
            assert isinstance(synth, Synth)
            
            # Get all instrument instances by name
            cols = [col for col in self.data if col.startswith(synth.name)]
            for c in cols:
                notes, sampLen = self.gen_notes(self.data[c], synth)
                shortest = min(shortest, sampLen)

                # chunks.append(notes)
                chunks.append(notes * (1 / len(cols)))  # add with normalized volume

        # Equalize lengths and sum signals
        temp = []
        for c in chunks:
            temp.append(np.asarray(c).flatten()[:shortest].reshape(1, shortest))
        
        # debug
        print(np.shape(temp))

        return sum(temp)


def main():
    tempo = Tempo(80)
    notegen = NoteGenerator(filepath="notes.xlsx", 
                            sheetname='Demo', 
                            t=tempo)

    # Lead synth
    pad = Synth(name='pad',
                wave=Osc.saw,
                voices=3,
                vtype='unison',
                adsr=[1.3, 0.3, 1.0, 0.3],
                tempo=tempo,
                cutoff=0.8,
                volume=0.3
                )

    # Bass synth
    bass = Synth(name='bass',
                wave=Osc.saw,
                voices=2,
                vtype='unison',
                adsr=[0.01, 0.5, 0.1, 0.05],
                tempo=tempo,
                cutoff=0.4,
                volume=0.17
                )

    # Lead synth
    lead = Synth(name='lead',
                wave=Osc.tri,
                voices=2,
                vtype='unison',
                adsr=[0.01, 0.0, 1.0, 0.05],
                tempo=tempo,
                cutoff=0.9,
                volume=0.3
                )

    # Arp synth
    arp = Synth(name='arp',
                wave=Osc.square,
                voices=1,
                vtype='unison',
                adsr=[0.02, 0.03, 0.4, 0.1],
                tempo=tempo,
                cutoff=0.6,
                volume=0.12
                )

    chunks = notegen.get([pad, bass, lead, arp])

    # Play stuff
    chunk = np.concatenate(chunks) * 0.25
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, output=1)
    stream.write(chunk.astype(np.float32).tobytes())
    stream.close()
    p.terminate()



if __name__ == "__main__":
    main()
