import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
import os
from scipy.signal import fftconvolve, resample

RIR_folder = '/Users/apple/Library/CloudStorage/Dropbox/My_ESAT/Experiments/GroepT-Jan2025/RIRs'
audiofile = '/Users/apple/Library/CloudStorage/Dropbox/My_ESAT/Projects/MY_PROJECTS/GroupT_installation/MAX_standalone/tracks/mixed_4tracks.wav'

# multi convolution
def multi_convolve(audio1, audio2, Nr):
    audio_conv = audio1
    for i in range(Nr):
        audio_conv = fftconvolve(audio_conv, audio2,mode = 'full')
        audio_conv  = audio_conv / np.max(np.abs(audio_conv)) # Normalize the audio_conv
        print(f'AudioConvolved audio length: {len(audio_conv)/fs_audio} seconds')

    return audio_conv

###################

# PARAMETERS
#list all positions in the RIR folder

pos_list = os.listdir(RIR_folder)

# sort the list of positions
pos_list.sort()

convNr = np.arange(1, 10, 1)
truncation_seconds = 9
seconds_of_audio = 5

rir_name = "/Users/apple/Library/CloudStorage/Dropbox/My_ESAT/Experiments/GroepT-Jan2025/RIRs/pos_0_0/RIR1/RIR_1_mic_A_20250126_133543.wav"
rir, fs_rir = sf.read(rir_name)

# read both the RIR and the audio file
audio, fs_audio = sf.read(audiofile)
audio = audio[:fs_audio*seconds_of_audio,0] # take only one channel 
# Resample the RIR to the audio file's sampling frequency
rir = resample(rir, len(rir) * fs_audio // fs_rir)
rir = rir[:fs_audio*seconds_of_audio] # take only 4 seconds of the RIR 

convolved_rir = multi_convolve(audio,rir, 10)

#play the convolved RIR
print('Playing the multiconvolved file')
sd.play(convolved_rir, fs_audio)
sd.wait()

#plot the convolved RIR
plt.figure(figsize=(10, 4))
plt.plot(convolved_rir)
plt.title('Multiconvolved RIR')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.xlim(0, len(convolved_rir))
plt.grid()
plt.show()  




# for position in pos_list:
    
#     if position is not '.DS_Store':
#         print(f'Position: {position}')

#         for multiconvNr in convNr:
#             print(f'MulticonvNr: {multiconvNr}')

#             rir_subfolder = RIR_folder + '/' + position + '/RIR1/'
#             rir_files = os.listdir(rir_subfolder)

#             # If the file is a .wav file, and contains 'E', load it
#             selected_rir = [f for f in rir_files if f.endswith('.wav') and 'E' in f]
#             rir, fs_rir = sf.read(rir_subfolder + '/' + selected_rir[0])

#             # Augment RIR by 20 dB
#             rir = rir * 10**(20/20)


#             # plt.show()

#             # Load and trim the audio file 
#             audio, fs_audio = sf.read(audiofile)
#             audio = audio[:fs_audio*seconds_of_audio,0] # take only one channel

#             # Resample the RIR to the audio file's sampling frequency
#             rir = resample(rir, len(rir) * fs_audio // fs_rir)
#             rir = rir[:fs_audio*4] # take only 4 seconds of the RIR

#             # display lenght of audio and rir
#             print(f'Audio length: {len(audio)/fs_audio} seconds')
#             print(f'RIR length: {len(rir)/fs_audio} seconds')



#             # MULTICONVOLUTION OF RIR and truncation
#             convolved_rir = multi_convolve(rir,rir, multiconvNr)
#             truncation = fs_audio * truncation_seconds # truncation 
#             reverb = convolved_rir[:truncation]

#             # # Play the reverb
#             # print('Playing the multiconvolved file')
#             # sd.play(reverb, fs_audio)
#             # sd.wait()

#             # save the reverb
#             sf.write(f'audio_outputs/reverb_pos_{position}_convNr_{multiconvNr}.wav', reverb, fs_audio)


