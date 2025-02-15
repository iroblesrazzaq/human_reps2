
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import mat73
from scipy.io import loadmat
import re
import os
import json

@dataclass
class RecordingInfo:
    """Contains metadata about a recording session"""
    start_unix: float
    end_unix: float
    experiment_type: str  # e.g., 'movie', 'preSleep', 'postSleep'

# %%
@dataclass
class Neuron:
    """
    Class for a single Neuron in a single patient
    pid: str pid of patient (redundancy maybe but better safe than sorry)
    area: str | None  recording area
    spikes: list | np.ndarray all spike times (in seconds)
    """
    neuron_id: str
    pid: str
    spike_times: np.ndarray
    area: str | None = None
    metadata: dict | None = None
    
    @property
    def firing_rate(self, window: Tuple[float, float] = None) -> float:
        """
        Function to get firing rate of neuron within a certain time period (default whole recording)

        times are in seconds
        """
        if window:
            spikes = self.spike_times[(self.spike_times >= window[0]) & (self.spike_times <= window[1])]
            duration = window[1] - window[0]
        else:
            spikes = self.spike_times
            duration = self.spike_times[-1] - self.spike_times[0]

        assert duration > 0, "Duration < 0, Error"

        return len(spikes) / duration
        


# %%
class PatientData:
    """
    Contains all relevant information for a single patient

    - movie drift adjusted times with CSV (create new csv to use every time or run code every time -- not expensive so will do second for reproducibility)
    - patient info from all exp epochs (start unix, pre/post, etc etc)
    - recordings!!  
        - dictionary with all neurons and firing times? but also want to be able to filter by brain area


    - methods for analysis?
        - can be functions for general, not specific to patient
        - make analysis class?
            - would have functions for heatmaps, decoders, etc?
        - these will clutter patient class, I mainly just want all data for a single patient concentrated in one place, easy to use and access
    """

    def __init__(self, pid: str):
        # want to load csv, fix correlation issue
        # so call function to load csv, multiple times by coefficient, 
        # get concept onsets from the 
        self.pid = pid
        self.dataloader = Dataloader() # type: ignore

        self.neurons: list[Neuron] = self.dataloader.get_all_patient_neurons(self.pid)

        self.times_dict = self._get_relative_times() # has everything

        self.movie_df = self.times_dict['movie_corrected_concept_times_df'].copy().drop(columns = ['uncorrected_time_sec', 'corrected_time_sec'], axis=1)
        
        self.preSleep_concepts = self.times_dict['rel_preSleep_concept_vocalizations']
        self.postSleep_concepts = self.times_dict['rel_postSleep_concept_vocalizations']



        # timing_info = d.get_timing_info(pid=self.pid)



    def _get_relative_times(self) -> Dict[str, str]:
        """
        Takes unix timing and returns times relative to recording recording start
        Adjusts movie onset timing, recall concept vocalization timing to relative to recording start
        
        """
        # check that all ts_starts are the same
        ts_start = None
        for neuron in self.neurons:
            if not ts_start:
                ts_start = neuron.metadata['ts_start']
            else:
                if neuron.metadata['ts_start'] != ts_start:
                    raise Exception("Neuron ts_starts not aligned, something wrong")
        
        # now have ts_start verified
        unix_times_dict = self.dataloader.get_timing_info(pid=self.pid)
        times_dict = {}
        for key, val in unix_times_dict.items():
            if 'unix' in key:
                if isinstance(val, list):
                    val = val[0]
                times_dict[key] = val # current strat - have all timesin times dict, can make method to remove?
                times_dict[key.replace("unix", "rel")] = val - ts_start
            else:
                times_dict[key] = val


        df = times_dict["movie_corrected_concept_times_df"]
        df["rel_corrected_time_sec"] = df['corrected_time_sec'] + times_dict['movie_start_rel']
        times_dict["movie_corrected_concept_times_df"] = df


        rel_preSleep_concept_vocalizations = {}
        for concept, times_list in times_dict['preSleep_concept_vocalizations'].items():
            rel = times_dict['preSleep_recall_start_rel']
            rel_preSleep_concept_vocalizations[concept] = [t + rel for t in times_list]
            times_dict['rel_preSleep_concept_vocalizations'] = rel_preSleep_concept_vocalizations
        
        rel_postSleep_concept_vocalizations = {}
        for concept, times_list in times_dict['postSleep_concept_vocalizations'].items():
            rel = times_dict['postSleep_recall_start_rel']
            rel_postSleep_concept_vocalizations[concept] = [t + rel for t in times_list]
            times_dict['rel_postSleep_concept_vocalizations'] = rel_postSleep_concept_vocalizations

        return times_dict

class Dataloader:
    """Class to contain functions to load data"""


    def parse_filename(self, filename):
        base = filename.split('-')[-1].replace('.mat', '')

        parsed = filename.replace('.mat', '').split('-')
        if len(parsed) == 2:  # Normal case like GA2-RAH7
            base = parsed[-1]
        elif len(parsed) == 3:  # Case with hyphenated area like GA3-RSUB-PHG1
            base = '-'.join(parsed[1:])  # Join with hyphen to preserve structure
        else:
            return (filename.replace('.mat', ''), None)

        match = re.match(r'(.*?[-]?\w+?)(\d+)$', base)
        if match:
            area_name = match.group(1)  # Group 1 contains everything before the numbers
            channel_num = match.group(2)  # Group 2 contains the numbers
            return base, area_name
        return (base, None) # None for no areaname
    

    def _get_neurons_from_mat(self, file_path, pid):
        """
        Load spike data from .mat file, handling different MATLAB file versions

        Return instances of the Neuron class, adding spike data to each one
        """
        try:
            data = loadmat(file_path)
        except (NotImplementedError, TypeError):
            data = mat73.loadmat(file_path)
        
        # Extract cluster_class data
        cluster_class = data['cluster_class']
        
        # Extract timestampsStart
        ts_start = data["timestampsStart"]
        if ts_start.shape == ():
            ts_start = float(ts_start)
        else:
            ts_start = float(ts_start[0][0])

        filename = file_path.split('/')[-1]
        base, area_name = self.parse_filename(filename)
        neurons = []
        unique_clusters = np.unique(cluster_class[:, 0])
        for cluster_id in unique_clusters:
            mask = cluster_class[:, 0] == cluster_id
            spike_times = cluster_class[mask, 1]

            neurons.append(Neuron(
                neuron_id=f"{base}-{int(cluster_id)}",
                pid = pid,
                spike_times=spike_times,
                area=area_name,
                metadata={'ts_start': ts_start}
                ))

        return neurons
    
    def get_all_patient_neurons(self, pid, base_dir="./Data"):
        neurons = []
        for patient_dir in os.listdir(base_dir): # lists 566_movie paradigm, etc dirs
            patient_dict_name = f"{patient_dir.replace('_MovieParadigm', '')}_files"

            if pid in patient_dir: # we have the correct patient id
                for exp_dir in os.listdir(os.path.join(base_dir, patient_dir)):
                    if len(exp_dir.split('-')) > 2: # then we have our exp-5-6-7 pattern directory with spiking files
                        for file in os.listdir(os.path.join(base_dir, patient_dir, exp_dir, 'CSC_micro_spikes')):
                            file_path = os.path.join(base_dir, patient_dir, exp_dir, 'CSC_micro_spikes', file)
                            neurons += self._get_neurons_from_mat(file_path=file_path, pid=pid)
        return neurons # list of all neurons
    
    def _timing_info(self, pid, base_dir="./Data"):
        res_dict = {}
        for pdir in os.listdir(base_dir):
            
            if pid in pdir: # relevant directory
                #print(f"pdir: {pdir}")
                
                for exp_dir in os.listdir(os.path.join(base_dir, pdir)):
                    if len(exp_dir.split('-')) == 2: # Exp-K directory
                        for file in os.listdir(os.path.join(base_dir, pdir, exp_dir, 'Audio')):
                            if 'FR1' in file:
                                with open(os.path.join(base_dir, pdir, exp_dir, 'Audio', file)) as f:
                                    data = json.load(f)
                                    res_dict['preSleep_concept_vocalizations'] = data
                            elif 'FR2' in file:
                                with open(os.path.join(base_dir, pdir, exp_dir, 'Audio', file)) as f:
                                    data = json.load(f)
                                    res_dict['postSleep_concept_vocalizations'] = data
                            elif "audio_movie_start" in file:
                                with open(os.path.join(base_dir, pdir, exp_dir, 'Audio', file)) as f:
                                    data = json.load(f)
                                    res_dict['movie_timing_info'] = data
                            elif "audio_recall_timing" in file and 'pre' in file:
                                with open(os.path.join(base_dir, pdir, exp_dir, 'Audio', file)) as f:
                                    data = json.load(f)
                                    res_dict['preSleep_recall_timing'] = data
                            elif "audio_recall_timing" in file and 'post' in file:
                                with open(os.path.join(base_dir, pdir, exp_dir, 'Audio', file)) as f:
                                    data = json.load(f)
                                    res_dict['postSleep_recall_timing'] = data
        res_dict['concept_csv_path'] = base_dir + "/40m_act_24_S06E01_30fps_character_frames.csv"
        return res_dict

    def _extract_relevant_timing_info(self, pid, res_dict):
        relevant_timing = {}
        for key, info in res_dict.items():
            if key == "preSleep_concept_vocalizations":
                if info['pID'] == int(pid): # check for correct patient, sanity check
                    preSleep_concept_vocalizations = {}
                    for field, val in info.items():
                        if isinstance(val, list):
                            ms_to_secs = []
                            for time in val:
                                ms_to_secs.append(time/1000)
                            preSleep_concept_vocalizations[field] = ms_to_secs
                    relevant_timing['preSleep_concept_vocalizations'] = preSleep_concept_vocalizations# need to divide by 1000 for ms to s conversion


            if key == "postSleep_concept_vocalizations":
                if info['pID'] == int(pid):
                    postSleep_concept_vocalizations = {}
                    for field, val in info.items():
                        if isinstance(val, list):
                            ms_to_secs = []
                            for time in val:
                                ms_to_secs.append(time/1000)
                            postSleep_concept_vocalizations[field] = ms_to_secs
                    relevant_timing['postSleep_concept_vocalizations'] = postSleep_concept_vocalizations # need to divide by 1000 for ms to s conversion

            if key == "movie_timing_info":
                relevant_timing['movie_drift_factor'] = info["drift_correction_multiplier"]
                relevant_timing['movie_start_unix'] = info['start_unix']

            if key == "preSleep_recall_timing":
                relevant_timing['preSleep_recall_start_unix'] = info['start_unix']
                relevant_timing['preSleep_recall_end_unix'] = info['end_unix']

            if key == "postSleep_recall_timing":
                relevant_timing['postSleep_recall_start_unix'] = info['start_unix']
                relevant_timing['postSleep_recall_end_unix'] = info['end_unix']

            if key == "concept_csv_path": # if concept csv we will read from that path
                print(info)
                concept_csv = pd.read_csv(info)
                drift = relevant_timing['movie_drift_factor']
                concept_csv['corrected_time_sec'] = concept_csv['uncorrected_time_sec'] * drift

                relevant_timing['movie_corrected_concept_times_df'] = concept_csv

        return relevant_timing
    
    def get_timing_info(self, pid) -> Dict[str, str]:
        """
        Public method for class, calls _timing_info and _extract methods
        
        Returns: dict[str: str]
        """
        res = self._timing_info(pid=pid)
        return self._extract_relevant_timing_info(pid=pid, res_dict=res)
