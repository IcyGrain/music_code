import json,os
import pydub,librosa
def trying():
    # with open('songs.json','r') as f:
    #     songs = json.loads(f.read())
    #     print(songs)
    #     for i in range(1):
    #         song_dict = {'name':1,'path':2,'mixin':3}
    #         songs.append(song_dict)
    #     f.close()
    #
    # with open('songs.json','w') as f:
    #     json.dump(songs,f,ensure_ascii=False)
    #     f.close()
    #
    # with open('songs.json','r') as f:
    #     out = json.loads(f.read())
    #     print(out)

    # input = 'AudioIn'
    # input_files = []
    # if os.path.isdir(input):
    #     input_files = [os.path.join(input, f) for f in os.listdir(input) if f.endswith('.wav')]
    # elif os.path.isfile(input) and input.endswith('.wav'):
    #     input_files = [input]
    #
    # print(input_files)
    #
    # with open('songs.json','r') as f:
    #     songs = json.loads(f.read())
    #     print(songs)
    #     for i in input_files:
    #
    #         song_dict = {'name':i.split('\\')[1].split('.wav')[0],'path':i,'mixin':'False'}
    #         songs.append(song_dict)
    #     f.close()
    # print(songs)
    # list = [('A', 31924.0), ('E', 24529.0), ('D', 22806.0), ('G', 18173.0), ('B', 17173.0), ('F#/Gb', 14342.0), ('G#/Ab', 13853.0), ('C#/Db', 13567.0), ('C', 11806.0), ('D#/Eb', 8477.0), ('A#/Bb', 6188.0), ('F', 6054.0)]
    # sum = 0
    # for i in list:
    #     sum += i[1]
    # print(sum)
    #
    # dict = {}
    #
    # for i in range(3):
    #     dict[list[i][0]] = round(list[i][1]/sum,5)
    # print(dict)

    # with open('keys.json','r') as k:
    #     keys = json.loads(k.read())
    #
    # print(keys)
    # songs = []
    # final = "all i got&moments"
    # for i,key in enumerate(keys):
    #
    #     if i in [0,1]:
    #         songs.append({'name':key['name'],'path':key['path'],'mixin':i==0})
    #
    #     else:
    #         songs.clear()
    #         songs.append({'name':final,'path':'AudioOut\\{}.wav'.format(final),'mixin':True})
    #         songs.append({'name':key['name'],'path':key['path'],'mixin':False})
    #     print(songs)


    # s = pydub.AudioSegment.from_file("AudioIn\\heist.wav",format="wav")
    # speed = 2
    # sc = s.speedup(playback_speed = speed)
    # s.export(out_f="s.wav",format="wav")
    # sc.export(out_f="sc.wav", format="wav")

    # beat_track的hop_length修正以提高强拍准确率
    with open('points.json','r') as p:
        armin = json.loads(p.read())
        print(len(armin)//80)
trying()