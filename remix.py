from __future__ import division


import audioread
from scipy import signal
import math
from scipy.signal import butter,lfilter
import librosa, pydub
import numpy as np
import pyrubberband
from tempfile import TemporaryFile
import pickle, json, os
#pydub.AudioSegment.converter = 'C:/Users/lemon/Anaconda3/ffmpeg/ffmpeg-4.1.3-win64-static/bin/ffmpeg.exe'

class mash:

    def __init__(self, songs_, verbose, cached):
        self.sr = 44100  # new Sampling Rate for the audio files
        self.hlength = 256
        
        self.range = int(self.sr * 0.05)
        self.songs = songs_
        self.AudioIn = []
        self.AudioOut = []
        self.pathIn = []
        self.pathOut = []
        self.filter = {'in': [], 'out': []}
        self.beats = {'in': [], 'out': []}
        self.tempo = {'in': 0, 'out': 0}
        self.fade = {'in': [],'out': []}
        self.crossFade = {'in': [],'out': []}
        self.audioSegment = {'in': [],'out': []}
        self.final = ""
        self.point = 0
        self._setup()
        self._load(verbose=verbose, cached=cached)
        self._extract(verbose=verbose)
        self._segment(verbose=verbose)
        self._speedUp(verbose=verbose)
        self._outputTemp()
        out = self._mix(verbose=verbose)



        if verbose:
            print("Exporting...")
        out.export(out_f="AudioOut/{}.wav".format(self.final), format="wav")
        print("[SUCCESS] Export as {}.wav".format(self.final))

    def getPoints(self):
        return self.point

    def _setFilter(self,data):
        order = 20
        cutoff = 3000
        nyq = self.sr * 0.5
        normal_cutoff = cutoff / nyq
        b, a = butter(N=order, Wn=normal_cutoff, btype='lowpass', analog=False,)
        y = lfilter(b, a, data)
        return y

    def _setup(self):
        if not os.path.exists('cache'):
            os.makedirs('cache')

    def _load(self, verbose, cached):
        for song in songs:
            # 设置输出名
            if song['mixin']:
                self.final += song['name'] + "&"
            else:
                self.final += song['name']

            # 存在pkl时优先读取pkl             No Update
            if (os.path.exists("cache/{}.pkl".format(song['name'])) and cached):
                if verbose:
                    print("Loading", song['name'], "from cache")
                # with open("cache/%s.pkl"%song['name'], 'rb') as f:
                with open("cache/{}.pkl".format(song['name']), 'rb') as f:
                    if song['mixin']:
                        print("AudioIn=", song['name'])
                        self.AudioIn = pickle.load(f)
                        self.pathIn = song['path']
                    else:
                        print("AudioOut=", song['name'])
                        self.AudioOut = pickle.load(f)
                        self.pathOut = song['path']
                continue

            # 不存在pkl时从文件中读取并存取pkl
            if verbose:
                print("Loading", song['name'])
            y, sr = librosa.load(song['path'], sr=self.sr)
            #librosa.output.write_wav("input.wav", y, self.sr)
            # y = pydub.audioread.ffdec.FFmpegAudioFile(song["path"])
            if song['mixin']:
                self.AudioIn = y
                self.filter['in'] = self._setFilter(y)
                self.pathIn = song['path']
            else:
                self.AudioOut = y
                self.filter['out'] = self._setFilter(y)
                self.pathOut= song['path']

            print("[SUCCESS] Loaded", song['name'])

            if cached:
                try:
                    with open("cache/{}.pkl".format(song['name']),'wb') as f:
                        pickle.dump(y, f)
                        if verbose:
                            print("[SUCCESS] Cached", song['name'])
                except Exception as e:
                    if verbose:
                        print("[FAILED] Caching", song['name'])
                        print(e)




    def _extract(self, verbose):

        # self.AudioOut = self.AudioOut[0] # NOTE: considering 1mixin & 1mixout
        # self.pathOut = self.pathOut[0]

        #获取Beats区间与BPM
        self.tempo['in'], self.beats['in'] = librosa.beat.beat_track(y=self.AudioIn, sr=self.sr, hop_length=self.hlength)
        self.tempo['out'], self.beats['out'] = librosa.beat.beat_track(y=self.AudioOut, sr=self.sr, hop_length=self.hlength)
        self.tempo['in'] = round(self.tempo['in'],2)
        self.tempo['out'] = round(self.tempo['out'],2)

        if verbose:
            print("TempoIn=", self.tempo['in'])
            print("TempoOut=", self.tempo['out'])

        self._OTAC(verbose=verbose)
        self._crossFadeRegion(verbose=verbose)


    def _OTAC(self, verbose): # Optimal Tempo Adjustment Coefficient Computation
        C = [-2, -1, 0, 1, 2]

        if self.tempo['in'] == self.tempo['out']:
            self.tempo['tgt'] = self.tempo['in']
            return

        # 确定CFP时的Tempo
        Tin_ = [(2**c)*self.tempo['in'] for c in C]     # Fade-In的Tempo可选区间

        TinIndex_ = np.argmin(np.absolute(Tin_ - self.tempo['out']))    # 取可选区间中与Tempo-out相差最少的下标
        Copt = C[TinIndex_]     # 相差最少的C子集

        Bopt = (2**Copt)*self.tempo['in']   #算出具体的Tempo

        Tlow = min(Bopt, self.tempo['out'])
        Thigh = max(Bopt, self.tempo['out'])
        self.tempo['in'] = Bopt
        # LDC中的权重系数(Level of DisComfort
        a, b = 0.765, 1

        #CFP时均衡Tempo计算
        Ttgt = (a-b)*Tlow + np.sqrt( ((a-b)**2)*(Tlow**2) + 4*a*b*Thigh*Tlow )
        Ttgt = Ttgt/(2*a)

        # OTAC值计算
        if verbose:
            print("FoptIn=", round(Ttgt/Bopt,2))
            print("FoptOut=", round(Ttgt/self.tempo['out'],2))
            print("Ttgt=", round(Ttgt,2))

        self.tempo['tgt'] = round(Ttgt,2)

    def _crossFadeRegion(self, verbose): # Computes the cross fade region for the mixed song
        Na_min = min(self.beats['in'].shape[0] - 1, self.beats['out'].shape[0] - 1)
        Na = self.beats['in'].shape[0]-1
        #计算最佳CFP区间
        scores = []
        for i in range(int(Na_min / 12),int(Na_min / 4)):
            scores.append(self._score(i,Na))

        noBeats = np.argmax(scores) + int(Na_min / 6)
        self.point = round(np.max(scores), 1)

        inE = librosa.frames_to_time(self.beats['in'][len(self.beats['in'])-1])
        fadeInS = librosa.frames_to_time(self.beats['in'][-int(noBeats)])
        fadeInE = librosa.frames_to_time(self.beats['in'][-int(noBeats / 2)])
        self.fade['in'] = [fadeInS * 1000, fadeInE * 1000, inE * 1000]
        self.crossFade['in'] = [self.beats['in'][-int(noBeats)] * self.hlength, self.beats['in'][-int(noBeats / 2)] * self.hlength, self.beats['in'][len(self.beats['in'])-1] * self.hlength]

        outS = librosa.frames_to_time(self.beats['out'][0])
        fadeOutS = librosa.frames_to_time(self.beats['out'][int(noBeats / 2)])
        fadeOutE = librosa.frames_to_time(self.beats['out'][int(noBeats)])
        self.fade['out'] = [outS * 1000, fadeOutS * 1000, fadeOutE * 1000]
        self.crossFade['out'] = [self.beats['out'][0] * self.hlength, self.beats['out'][int(noBeats / 2)] * self.hlength, self.beats['out'][int(noBeats)] * self.hlength]

        if verbose:
            print("Best Power Corelation Scores=", round(np.max(scores),1))
            print("Number of beats in cross fade region=", noBeats)
            print("fadeInStart=", round(fadeInS,2))
            print("fadeOutEnd=", round(fadeOutE,2))

        # self.crossFade = [fadeInStart*1000, fadeOut*1000] # In milliseconds

    def _score(self, T, Na):
        cr = 0
        for i in range(0, T):
            beatin = np.sum(np.abs(self.filter['in'][self.beats['in'][Na-i] * self.hlength - self.range:self.beats['in'][Na-i] * self.hlength + self.range]))
            beatout = np.sum(np.abs(self.filter['out'][self.beats['out'][i] * self.hlength - self.range:self.beats['out'][i] * self.hlength + self.range]))

            cr += int(beatin * beatout)
        return cr/T

    def _segment(self, verbose):
        if verbose:
            print("Started Segmentation")

        if verbose:
            print("[SUCCESS] Segmented audio files")


        self.segments = {
            'in':[self.AudioIn[:self.crossFade['in'][0]], self.AudioIn[self.crossFade['in'][0]:self.crossFade['in'][1]], self.AudioIn[self.crossFade['in'][1]:]],
            'out': [self.AudioOut[self.crossFade['out'][0]:self.crossFade['out'][1]], self.AudioOut[self.crossFade['out'][1]:self.crossFade['out'][2]], self.AudioOut[self.crossFade['out'][2]:]]
        }



    def _speedUp(self, verbose):
        fadeIn = self.segments['in'][2]
        fadeOut = self.segments['out'][0]
        speed1 = self.tempo['tgt']/self.tempo['in']
        speed2 = self.tempo['tgt']/self.tempo['out']

        if verbose:
            print("Playback Speed of in end segment=",round(speed1,2),'X')
            print("Playback Speed of out start segment=",round(speed2,2),'X')

        self.segments['in'][2] = librosa.effects.time_stretch(fadeIn, speed1)
        self.segments['out'][0] = librosa.effects.time_stretch(fadeOut, speed2)
        del fadeIn, fadeOut

    def _outputTemp(self):
        for i, y in enumerate(self.segments['in']):
            librosa.output.write_wav("In{}.wav".format(i), y, self.sr)
        for i, y in enumerate(self.segments['out']):
            librosa.output.write_wav("Out{}.wav".format(i), y, self.sr)

        for i, seg in enumerate(self.segments['in']):
            self.audioSegment['in'].append(pydub.AudioSegment.from_file("In{}.wav".format(i), format="wav"))
            os.remove("In{}.wav".format(i))
        for i, seg in enumerate(self.segments['out']):
            self.audioSegment['out'].append(pydub.AudioSegment.from_file("Out{}.wav".format(i), format="wav"))
            os.remove("Out{}.wav".format(i))
    def _mix(self, verbose):




        out = pydub.AudioSegment.empty()
        out += self.audioSegment['in'][0]
        out += self.audioSegment['in'][1]



        diff = abs(len(self.audioSegment['out'][0]) - len(self.audioSegment['in'][2]))
        if(len(self.audioSegment['out'][0]) > len(self.audioSegment['in'][2])):
            silence = pydub.AudioSegment.silent(duration=diff)
            xf = self.audioSegment['in'][2].fade_out(duration=len(self.audioSegment['in'][2])) + silence
            xf *= self.audioSegment['out'][0].fade_in(duration=len(self.audioSegment['out'][0]))
            xf = xf + abs((self.audioSegment['in'][1].dBFS + self.audioSegment['out'][1].dBFS) / 2 - xf.dBFS)
            out += xf

            out += self.audioSegment['out'][1]
            out += self.audioSegment['out'][2]
        else:

            fade = self.audioSegment['in'][2].fade_out(duration=len(self.audioSegment['in'][0]))
            xf = self.audioSegment['out'][0].fade_in(duration=len(self.audioSegment['out'][0]))
            xf *= fade[:len(self.audioSegment['out'][0])]
            xf = xf + abs((self.audioSegment['in'][1].dBFS + self.audioSegment['out'][1].dBFS) / 2 - xf.dBFS)
            out += xf



            of = self.audioSegment['out'][1]+self.audioSegment['out'][2]
            silence = pydub.AudioSegment.silent(duration=len(of) - diff)
            of *= self.audioSegment['in'][2][len(self.audioSegment['out'][0]):] + silence

            out += of




        # xf.export("fade.wav",format="wav")
        # fade_in = self.audioSegment['in'][2].fade_out(duration=len(self.audioSegment['in'][2])) + silence
        # fade_in.export("fadeIn.wav", format="wav")
        # fade_out = self.audioSegment['out'][0].fade_in(duration=len(self.audioSegment['out'][0]))
        # fade_out.export("fadeOut.wav", format="wav")
        # of.export("fade_end.wav",format="wav")




        if verbose:
            print("[SUCCESS] Mixed 4 audio segment to 1")
        return out

    def getFinal(self):
        return self.final

if __name__ == '__main__':
    # with open('keys.json', 'r') as k:
    #     keys = json.loads(k.read())
    #
    # songs = []
    # for i, key in enumerate(keys):
    #
    #     if i in [0, 1]:
    #         songs.append({'name': key['name'], 'path': key['path'], 'mixin': i == 0})
    #         if i == 1:
    #             obj = mash(songs, verbose=False, cached=False)
    #             final = obj.getFinal()
    #     else:
    #
    #
    #         songs.clear()
    #         songs.append({'name': final, 'path': 'AudioOut\\{}.wav'.format(final), 'mixin': True})
    #         songs.append({'name': key['name'], 'path': key['path'], 'mixin': False})
    #         obj = mash(songs, verbose=False, cached=False)
    #         final = obj.getFinal()
    #
    #
    #
    #
    # k.close()
    with open('Armin.json', 'r') as k:
        keys = json.loads(k.read())

    songs = []
    with open('points.json', 'r') as p:
        points = json.loads(p.read())
        completed = len(points) // 80

        p.close()
    for i, mixin in enumerate(keys):
        if i <= completed:
            print("============================================Skip {}/{} Epoch============================================".format(i + 1, len(keys)))
            continue
        with open('points.json', 'r') as p:
            points = json.loads(p.read())
            p.close()
        for j, mixout in enumerate(keys):
            if i == j:
                continue
            songs.clear()
            songs.append({'name': mixin['name'], 'path': mixin['path'], 'mixin': True})
            songs.append({'name': mixout['name'], 'path': mixout['path'], 'mixin': False})
            obj = mash(songs, verbose=False, cached=False)
            final = obj.getFinal()
            point = obj.getPoints()
            points.append({'name': final, 'path': 'AudioOut\\{}.wav'.format(final), 'point': point})
            print("============================================Completed {}/{} Remix============================================".format((i * 80) + (j+1), len(keys) * len(keys)))

        with open('points.json', 'w') as p:
            json.dump(points, p, ensure_ascii=False)
            print("============================================Completed {}/{} Epoch============================================".format(i + 1, len(keys)))
            p.close()



    k.close()

    # with open('songs.json','r') as s:
    #     songs = json.loads(s.read())
    #     obj = mash(songs, verbose=False,cached=False)
    #     s.close()
    # print("End All")
