import argparse
import glob
import os
import random
import soundfile as sf
from tqdm import tqdm
import pandas as pd
import numpy as np
from moviepy.editor import AudioFileClip
import ffmpeg
import re
from scipy.io.wavfile import write as write_wav
import librosa as lib

def save_wav(save_path, audio, sr=16000):
    '''Function to write audio'''
    save_path = os.path.abspath(save_path)
    destdir = os.path.dirname(save_path)
    if not os.path.exists(destdir):
        try:
            os.makedirs(destdir)
        except:
            pass
    write_wav(save_path, sr, audio)
    return

# https://github.com/uniBruce/Mead
def convert2wav(path):
    save_path = path.split(".")[0] + ".wav"
    if os.path.exists(save_path):
        return save_path
    stream = ffmpeg.input(path)
    stream = ffmpeg.output(stream, save_path, ar=16000, ac=1)
    ffmpeg.run(stream)
    return save_path

def find_spk(text):
    pattern = r'\bsubject ([0-9]|[1-4][0-9]|50)\b'
    matches = re.findall(pattern, text)
    return matches

from pathlib import Path
def get_all_wavs(root, suffix):
    files = []
    for p in Path(root).iterdir():
        if str(p).endswith(".%s"%suffix):
            files.append(str(p))
        for s in p.rglob("*.%s"%suffix):
            files.append(str(s))
    return list(set(files))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data')
    parser.add_argument('--data-home', type=str)
    args = parser.parse_args()
    emo_dict = {
        "sad": "sad",
        "happy": "happy",
        "disgusted": "disgust",
        "angry": "angry",
        "surprised": "surprised",
        "fear": "fear",
        "neutral": "neutral",
        "contempt": "contempt",
    }
    
    trans_dict = {
        r"angry/\w+/001": "She had your dark suit in greasy wash water all year",
        r"angry/\w+/002": "Don’t ask me to carry an oily rag like that",
        r"angry/\w+/003": "Will you tell me why",
        r"angry/\w+/004": "Who authorized the unlimited expense account",
        r"angry/\w+/005": "Destroy every file related to my audits",
        r"angry/\w+/006": "The cat’s meow always hurts my ears",
        r"angry/\w+/007": "Why else would Danny allow others to go",
        r"angry/\w+/008": "Why do we need bigger and better bombs",
        r"angry/\w+/009": "Nuclear rockets can destroy airfields with ease",
        r"angry/\w+/010": "You’re so preoccupied that you’ve let your faith grow dim",
        r"angry/\w+/011": "Cory and Trish played tag with beach balls for hours",
        r"angry/\w+/012": "He will allow a rare lie",
        r"angry/\w+/013": "Withdraw all phony accusations at once",
        r"angry/\w+/014": "Right now may not be the best time for business mergers",
        r"angry/\w+/015": "Kindergarten children decorate their classrooms for all holidays",
        r"angry/\w+/016": "A few years later the dome fell in",
        r"angry/\w+/017": "But in this one section we welcomed auditors",
        r"angry/\w+/018": "Lot of people will roam the streets in costumes and masks and having a ball",
        r"angry/\w+/019": "In many of his poems death comes by train a strongly evocative visual image",
        r"angry/\w+/020": "Then he would realize they were really things that only he himself could think",
        r"angry/\w+/021": "Todd placed top priority on getting his bike fixed",
        r"angry/\w+/022": "One even gave my little dog a biscuit",
        r"angry/\w+/023": "I’ll have a scoop of that exotic purple and turquoise sherbet",
        r"angry/\w+/024": "His superiors had also preached this saying it was the way for eternal honor",
        r"angry/\w+/025": "The plaintiff in school desegregation cases",
        r"angry/\w+/026": "Land based radar would help with this task",
        r"angry/\w+/027": "It was not whatever tale was told by tails",
        r"angry/\w+/028": "No the man was not drunk he wondered how he got tied up with this stranger",
        r"angry/\w+/029": "No price is too high when true love is at stake",
        r"angry/\w+/030": "The revolution now under way in materials handling makes this much easier",
        
        r"contempt/\w+/001": "She had your dark suit in greasy wash water all year",
        r"contempt/\w+/002": "Don’t ask me to carry an oily rag like that",
        r"contempt/\w+/003": "Will you tell me why",
        r"contempt/\w+/004": "Are your grades higher or lower than Nancy’s",
        r"contempt/\w+/005": "This was easy for us",
        r"contempt/\w+/006": "Only lawyers love millionaires",
        r"contempt/\w+/007": "It’s illegal to postdate a check",
        r"contempt/\w+/008": "He stole a dime from a beggar",
        r"contempt/\w+/009": "His failure to open the store by eight cost him his job",
        r"contempt/\w+/010": "Let us differentiate a few of these ideas",
        r"contempt/\w+/011": "The big dog loved to chew on the old rag doll",
        r"contempt/\w+/012": "Family loyalties and cooperative work have been unbroken for generations",
        r"contempt/\w+/013": "Withdraw only as much money as you need",
        r"contempt/\w+/014": "The way is to rent a chauffeur driven car",
        r"contempt/\w+/015": "No one material is best for all situations",
        r"contempt/\w+/016": "Mosquitoes exist in warm humid climates",
        r"contempt/\w+/017": "We of the liberal led world got all set for peace and rehabilitation",
        r"contempt/\w+/018": "Can your insurance company aid you in reducing administrative costs",
        r"contempt/\w+/019": "She sprang up and went swiftly to the bedroom",
        r"contempt/\w+/020": "Todd placed top priority on getting his bike fixed",
        r"contempt/\w+/021": "One even gave my little dog a biscuit",
        r"contempt/\w+/022": "I’ll have a scoop of that exotic purple and turquoise sherbet",
        r"contempt/\w+/023": "His superiors had also preached this saying it was the way for eternal honor",
        r"contempt/\w+/024": "The plaintiff in school desegregation cases",
        r"contempt/\w+/025": "Land based radar would help with this task",
        r"contempt/\w+/026": "It was not whatever tale was told by tails",
        r"contempt/\w+/027": "No the man was not drunk he wondered how he got tied up with this stranger",
        r"contempt/\w+/028": "No price is too high when true love is at stake",
        r"contempt/\w+/029": "The revolution now under way in materials handling makes this much easier",
        r"contempt/\w+/030": "The revolution now under way in materials handling makes this much easier",
        
        r"disgusted/\w+/001": "Please take this dirty table cloth to the cleaners for me",
        r"disgusted/\w+/002": "The small boy put the worm on the hook",
        r"disgusted/\w+/003": "You’re not living up to your own principles she told my discouraged people",
        r"disgusted/\w+/004": "Don’t do Charlie’s dirty dishes",
        r"disgusted/\w+/005": "Will Robin wear a yellow lily",
        r"disgusted/\w+/006": "Young children should avoid exposure to contagious diseases",
        r"disgusted/\w+/007": "Military personnel are expected to obey government orders",
        r"disgusted/\w+/008": "Basketball can be an entertaining sport",
        r"disgusted/\w+/009": "How good is your endurance",
        r"disgusted/\w+/010": "Barb burned paper and leaves in a big bonfire",
        r"disgusted/\w+/011": "December and January are nice months to spend in Miami",
        r"disgusted/\w+/012": "If people were more generous there would be no need for welfare",
        r"disgusted/\w+/013": "Does society really exist as an entity over and above the agglomeration of men",
        r"disgusted/\w+/014": "Todd placed top priority on getting his bike fixed",
        r"disgusted/\w+/015": "One even gave my little dog a biscuit",
        r"disgusted/\w+/016": "I’ll have a scoop of that exotic purple and turquoise sherbet",
        r"disgusted/\w+/017": "Land based radar would help with this task",
        r"disgusted/\w+/018": "The plaintiff in school desegregation cases",
        r"disgusted/\w+/019": "His superiors had also preached this saying it was the way for eternal honor",
        r"disgusted/\w+/020": "It was not whatever tale was told by tails",
        r"disgusted/\w+/021": "No the man was not drunk he wondered how he got tied up with this stranger",
        r"disgusted/\w+/022": "No price is too high when true love is at stake",
        r"disgusted/\w+/023": "The revolution now under way in materials handling makes this much easier",
        r"disgusted/\w+/024": "She had your dark suit in greasy wash water all year",
        r"disgusted/\w+/025": "Don’t ask me to carry an oily rag like that",
        r"disgusted/\w+/026": "Will you tell me why",
        r"disgusted/\w+/027": "The revolution now under way in materials handling makes this much easier",
        r"disgusted/\w+/028": "She had your dark suit in greasy wash water all year",
        r"disgusted/\w+/029": "Don’t ask me to carry an oily rag like that",
        r"disgusted/\w+/030": "Will you tell me why",
        
        r"fear/\w+/001": "She had your dark suit in greasy wash water all year",
        r"fear/\w+/002": "Don’t ask me to carry an oily rag like that",
        r"fear/\w+/003": "Will you tell me why",
        r"fear/\w+/004": "Call an ambulance for medical assistance",
        r"fear/\w+/005": "Tornado’s often destroy acres of farm land",
        r"fear/\w+/006": "Destroy every file related to my audits",
        r"fear/\w+/007": "Would you allow acts of violence",
        r"fear/\w+/008": "The high security prison was surrounded by barbed wire",
        r"fear/\w+/009": "His shoulder felt as if it were broken",
        r"fear/\w+/010": "The fish began to leap frantically on the surface of the small lake",
        r"fear/\w+/011": "Straw hats are out of fashion this year",
        r"fear/\w+/012": "That diagram makes sense only after much study",
        r"fear/\w+/013": "Special task forces rescue hostages from kidnappers",
        r"fear/\w+/014": "The tooth fairy forgot to come when Roger’s tooth fell out",
        r"fear/\w+/015": "Will Robin wear a yellow lily",
        r"fear/\w+/016": "Their props were two stepladders a chair and a palm fan",
        r"fear/\w+/017": "This is a problem that goes considerably beyond questions of salary and tenure",
        r"fear/\w+/018": "The pulsing glow of a cigarette",
        r"fear/\w+/019": "One looked down on a sea of leaves a breaking wave of flower",
        r"fear/\w+/020": "We will achieve a more vivid sense of what it is by realizing what it is not",
        r"fear/\w+/021": "Todd placed top priority on getting his bike fixed",
        r"fear/\w+/022": "One even gave my little dog a biscuit",
        r"fear/\w+/023": "I’ll have a scoop of that exotic purple and turquoise sherbet",
        r"fear/\w+/024": "His superiors had also preached this saying it was the way for eternal honor",
        r"fear/\w+/025": "The plaintiff in school desegregation cases",
        r"fear/\w+/026": "Land based radar would help with this task",
        r"fear/\w+/027": "It was not whatever tale was told by tails",
        r"fear/\w+/028": "No the man was not drunk he wondered how he got tied up with this stranger",
        r"fear/\w+/029": "No price is too high when true love is at stake",
        r"fear/\w+/030": "The revolution now under way in materials handling makes this much easier",
        
        r"happy/\w+/001": "She had your dark suit in greasy wash water all year",
        r"happy/\w+/002": "Don’t ask me to carry an oily rag like that",
        r"happy/\w+/003": "Will you tell me why",
        r"happy/\w+/004": "Those musicians harmonize marvelously",
        r"happy/\w+/005": "The eastern coast is a place for pure pleasure and excitement",
        r"happy/\w+/006": "Tim takes Sheila to see movies twice a week",
        r"happy/\w+/007": "They used an aggressive policeman to flag thoughtless motorists",
        r"happy/\w+/008": "When you’re less fatigued things just naturally look brighter",
        r"happy/\w+/009": "By that time perhaps something better can be done",
        r"happy/\w+/010": "She found herself able to sing any role and any song which struck her fancy",
        r"happy/\w+/011": "That noise problem grows more annoying each day",
        r"happy/\w+/012": "Project development was proceeding too slowly",
        r"happy/\w+/013": "The oasis was a mirage",
        r"happy/\w+/014": "Are your grades higher or lower than Nancy’s",
        r"happy/\w+/015": "Serve the coleslaw after I add the oil",
        r"happy/\w+/016": "By that one feels that magnetic forces are as general as electrical forces",
        r"happy/\w+/017": "His artistic accomplishments guaranteed him entry into any social gathering",
        r"happy/\w+/018": "He would not carry a brief case",
        r"happy/\w+/019": "Obviously the bridal pair has many adjustments to make to their new situation",
        r"happy/\w+/020": "Both the conditions and the complicity are documented in considerable detail",
        r"happy/\w+/021": "Todd placed top priority on getting his bike fixed",
        r"happy/\w+/022": "One even gave my little dog a biscuit",
        r"happy/\w+/023": "I’ll have a scoop of that exotic purple and turquoise sherbet",
        r"happy/\w+/024": "Land based radar would help with this task",
        r"happy/\w+/025": "The plaintiff in school desegregation cases",
        r"happy/\w+/026": "His superiors had also preached this saying it was the way for eternal honor",
        r"happy/\w+/027": "It was not whatever tale was told by tails",
        r"happy/\w+/028": "No the man was not drunk he wondered how he got tied up with this stranger",
        r"happy/\w+/029": "No price is too high when true love is at stake",
        r"happy/\w+/030": "The revolution now under way in materials handling makes this much easier",
        
        r"neutral/\w+/001": "She had your dark suit in greasy wash water all year",
        r"neutral/\w+/002": "Don’t ask me to carry an oily rag like that",
        r"neutral/\w+/003": "Will you tell me why",
        r"neutral/\w+/004": "Bridges tunnels and ferries are the most common methods of river crossings",
        r"neutral/\w+/005": "The moment of truth is the moment of crisis",
        r"neutral/\w+/006": "The best way to learn is to solve extra problems",
        r"neutral/\w+/007": "Thereupon followed a demonstration that tyranny knows no ideological confines",
        r"neutral/\w+/008": "Calcium makes bones and teeth strong",
        r"neutral/\w+/009": "Catastrophic economic cutbacks neglect the poor",
        r"neutral/\w+/010": "Allow leeway here but rationalize all errors",
        r"neutral/\w+/011": "Greg buys fresh milk each weekday morning",
        r"neutral/\w+/012": "Agricultural products are unevenly distributed",
        r"neutral/\w+/013": "The nearest synagogue may not be within walking distance",
        r"neutral/\w+/014": "As such it was beyond politics and had no need of justification by a message",
        r"neutral/\w+/015": "He always seemed to have money in his pocket",
        r"neutral/\w+/016": "No return address whatsoever",
        r"neutral/\w+/017": "Keep your seats boys I just want to put some finishing touches on this thing",
        r"neutral/\w+/018": "He ripped down the cellophane carefully and laid three dogs on the tin foil",
        r"neutral/\w+/019": "Who authorized the unlimited expense account",
        r"neutral/\w+/020": "Destroy every file related to my audits",
        r"neutral/\w+/021": "Please take this dirty table cloth to the cleaners for me",
        r"neutral/\w+/022": "The small boy put the worm on the hook",
        r"neutral/\w+/023": "Call an ambulance for medical assistance",
        r"neutral/\w+/024": "Tornado’s often destroy acres of farm land",
        r"neutral/\w+/025": "The carpet cleaners shampooed our oriental rug",
        r"neutral/\w+/026": "His shoulder felt as if it were broken",
        r"neutral/\w+/027": "The prospect of cutting back spending is an unpleasant one for any governor",
        r"neutral/\w+/028": "The diagnosis was discouraging however he was not overly worried",
        r"neutral/\w+/029": "Those musicians harmonize marvelously",
        r"neutral/\w+/030": "The eastern coast is a place for pure pleasure and excitement",
        r"neutral/\w+/031": "Todd placed top priority on getting his bike fixed",
        r"neutral/\w+/032": "One even gave my little dog a biscuit",
        r"neutral/\w+/033": "I’ll have a scoop of that exotic purple and turquoise sherbet",
        r"neutral/\w+/034": "Land based radar would help with this task",
        r"neutral/\w+/035": "The plaintiff in school desegregation cases",
        r"neutral/\w+/036": "His superiors had also preached this saying it was the way for eternal honor",
        r"neutral/\w+/037": "It was not whatever tale was told by tails",
        r"neutral/\w+/038": "No the man was not drunk he wondered how he got tied up with this stranger",
        r"neutral/\w+/039": "No price is too high when true love is at stake",
        r"neutral/\w+/040": "The revolution now under way in materials handling makes this much easier",
        
        r"sad/\w+/001": "She had your dark suit in greasy wash water all year",
        r"sad/\w+/002": "Don’t ask me to carry an oily rag like that",
        r"sad/\w+/003": "Will you tell me why",
        r"sad/\w+/004": "The prospect of cutting back spending is an unpleasant one for any governor",
        r"sad/\w+/005": "The diagnosis was discouraging however he was not overly worried",
        r"sad/\w+/006": "We can die too we can die like real people People never live forever",
        r"sad/\w+/007": "He didn’t figure her at all and if he found out a woman it’d be bad",
        r"sad/\w+/008": "There would still be plenty of moments of regret and sadness and guilty relief",
        r"sad/\w+/009": "She drank greedily and murmured thank you as he lowered her head",
        r"sad/\w+/010": "There’s no chance now of all of us getting away",
        r"sad/\w+/011": "Before Thursday’s exam review every formula",
        r"sad/\w+/012": "They enjoy it when I audition",
        r"sad/\w+/013": "John cleans shellfish for a living",
        r"sad/\w+/014": "He stole a dime from a beggar",
        r"sad/\w+/015": "Jeff thought you argued in favor of a centrifuge purchase",
        r"sad/\w+/016": "However the litter remained augmented by several dozen lunchroom suppers",
        r"sad/\w+/017": "American newspaper reviewers like to call his plays nihilistic",
        r"sad/\w+/018": "But the ships are very slow now and we don’t get so many sailors any more",
        r"sad/\w+/019": "It is one of the rare public ventures here on which nearly everyone is agreed",
        r"sad/\w+/020": "No manufacturer has taken the initiative in pointing out the costs involved",
        r"sad/\w+/021": "Todd placed top priority on getting his bike fixed",
        r"sad/\w+/022": "One even gave my little dog a biscuit",
        r"sad/\w+/023": "I’ll have a scoop of that exotic purple and turquoise sherbet",
        r"sad/\w+/024": "Land based radar would help with this task",
        r"sad/\w+/025": "The plaintiff in school desegregation cases",
        r"sad/\w+/026": "His superiors had also preached this saying it was the way for eternal honor",
        r"sad/\w+/027": "It was not whatever tale was told by tails",
        r"sad/\w+/028": "No the man was not drunk he wondered how he got tied up with this stranger",
        r"sad/\w+/029": "No price is too high when true love is at stake",
        r"sad/\w+/030": "The revolution now under way in materials handling makes this much easier",
        
        r"surprised/\w+/001": "She had your dark suit in greasy wash water all year",
        r"surprised/\w+/002": "Don’t ask me to carry an oily rag like that",
        r"surprised/\w+/003": "Will you tell me why",
        r"surprised/\w+/004": "The carpet cleaners shampooed our oriental rug",
        r"surprised/\w+/005": "His shoulder felt as if it were broken",
        r"surprised/\w+/006": "The patient and the surgeon are both recuperating from the lengthy operation",
        r"surprised/\w+/007": "He ate four extra eggs for breakfast",
        r"surprised/\w+/008": "While waiting for Chipper she crisscrossed the square many times",
        r"surprised/\w+/009": "I just saw Jim near the new archeological museum",
        r"surprised/\w+/010": "I took her word for it but is she really going with you",
        r"surprised/\w+/011": "The viewpoint overlooked the ocean",
        r"surprised/\w+/012": "I’d ride the subway but I haven’t enough change",
        r"surprised/\w+/013": "The clumsy customer spilled some expensive perfume",
        r"surprised/\w+/014": "Please dig my potatoes up before frost",
        r"surprised/\w+/015": "Grandmother outgrew her upbringing in petticoats",
        r"surprised/\w+/016": "Salvation reconsidered",
        r"surprised/\w+/017": "Properly used the present book is an excellent instrument of enlightenment",
        r"surprised/\w+/018": "Lighted windows glowed jewel bright through the downpour",
        r"surprised/\w+/019": "But this doesn’t detract from its merit as an interesting if not great film",
        r"surprised/\w+/020": "He further proposed grants of an unspecified sum for experimental Hospitals",
        r"surprised/\w+/021": "Todd placed top priority on getting his bike fixed",
        r"surprised/\w+/022": "One even gave my little dog a biscuit",
        r"surprised/\w+/023": "I’ll have a scoop of that exotic purple and turquoise sherbet",
        r"surprised/\w+/024": "Land based radar would help with this task",
        r"surprised/\w+/025": "The plaintiff in school desegregation cases",
        r"surprised/\w+/026": "His superiors had also preached this saying it was the way for eternal honor",
        r"surprised/\w+/027": "It was not whatever tale was told by tails",
        r"surprised/\w+/028": "No the man was not drunk he wondered how he got tied up with this stranger",
        r"surprised/\w+/029": "No price is too high when true love is at stake",
        r"surprised/\w+/030": "The revolution now under way in materials handling makes this much easier",
    }
    
    # download data from https://www.kaggle.com/tli725/jl-corpus
    data_name = "mead"
    root = os.path.join(args.data_home, data_name)
    
    # name, sample_rate, length, emo, trans
    with open(os.path.join(root, "info.tsv"), "w") as train_f:
        for fname in tqdm(get_all_wavs(root, "m4a")):
            file_path = os.path.realpath(fname)
            try:
                file_path = convert2wav(file_path)
            except:
                continue
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = audio[0]
            if sr != 16000:
                file_path = file_path.replace(f"/{data_name}/", f"/{data_name}_16k/")
                audio = lib.resample(audio, orig_sr=sr, target_sr=16000)
                save_wav(file_path, audio)
                sr = 16000
            for key in trans_dict.keys():
                if re.search(str(key), str(file_path)):
                    trans = trans_dict[key]
                    emo = emo_dict[key.split("/")[0]]
                    break
            spk = file_path.split("MEAD/")[-1].split("/")[0]
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_path, sr, len(audio), spk, emo, "_", trans), file=train_f
            )
