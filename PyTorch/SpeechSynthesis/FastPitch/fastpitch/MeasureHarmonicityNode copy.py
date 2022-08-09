from __future__ import annotations

from typing import Union
import parselmouth
from parselmouth.praat import call

import os
import importlib
from scipy.io import wavfile
# Task Factories know about all types of tasks that can be created and creates the appropriate instance when called
# Abstracts the creation logic, and all of the library importing away from the user
###################################################################################################
# NodeFactory: Abstracts the creation logic for new nodes
# Singleton Factory pattern, so no instantiation needed
###################################################################################################
class NodeFactory:
    """Create new nodes based on a type id"""

    registered_nodes = {}

    ###############################################################################################
    # Register Node (Class Method) : Registers a new node under a specified name/id
    ###############################################################################################
    @classmethod
    def register_node(cls, type_id, type_class):
        """register a new node into the factory

        Args:
            type_id:
            type_class:
        """
        cls.registered_nodes[type_id] = type_class

    ###############################################################################################
    # Create Node (Class Method) : Returns a new instance of the specifed node
    ###############################################################################################
    @classmethod
    def create_node(cls, node_id, type_id):
        """create a new node based on type id

        Args:
            node_id:
            type_id:
        """
        node = cls.registered_nodes[type_id](node_id)
        return node

    ###############################################################################################
    # Import Node (Class Method) : Imports the python class associated with a node and register it
    ###############################################################################################
    @classmethod
    def import_node(cls, type_id, toolkit_id, class_name):
        """import the class for a new node and register it

        Args:
            type_id:
            toolkit_id:
            class_name:
        """
        if type_id not in cls.registered_nodes:
            module_name = "toolkits." + toolkit_id + "." + class_name
            module = importlib.import_module(module_name)
            cls.register_node(type_id, getattr(module, class_name))

        return cls.registered_nodes[type_id]
###################################################################################################
# Node/Task
# + Abstract node for handling the running of a discrete task in a workflow
###################################################################################################
class Node:
    """Abstract Node class. This describes a single operational function in our
    process pipeline
    """

    ################################################################################################
    # Node.__init__: Initializes the node
    # + node_id: The unique id for the node.
    ################################################################################################
    def __init__(self, node_id=None):
        """Initialize Node

        Args:
            node_id:
        """
        self.node_id = node_id  # identifier for the node
        self.ready = (
            {}
        )  # Flags for each argument, all true indicates the node should run
        self.state = (
            {}
        )  # Variables local to the node, set internally by itself. Persits
        self.args = (
            {}
        )  # Variables local to the node, set externally by it's parents. Does not persist
        self.global_vars = {}  # Variables global to the entire pipeline.
        self.done = True  # Flag indicating if this node requires multple passes
        self.event_callbacks = {}
        self.events_fired = {}
        self.default_ready = {}

    ################################################################################################
    # Node.start: Pipeline runs this when the pipeline starts
    ################################################################################################
    def start(self):
        """Default start hook ran by the pipeline before all processing
        begins
        """
        return None

    ################################################################################################
    # Node.start: Pipeline runs this when the node is ready
    ################################################################################################
    def process(self):
        """Default process hook ran by the pipeline when this node is ready"""
        return {}

    ################################################################################################
    # Node.reset: Pipeline runs this to reset this nodes state variables
    ################################################################################################
    def reset(self):
        """Default process hook ran by the pipeline to reset this node's state
        variables
        """
        self.ready = {**self.default_ready}
        return None

    ################################################################################################
    # Node.end: Pipeline runs this when the pipeline ends
    ################################################################################################
    def end(self, results):
        """Default end hook ran by the pipeline once the pipeline has finished

        Args:
            results:
        """
        return results
class VoicelabNode(Node):
    """Extends the basic node with some shared voicelab functionalities
    """
    def pitch_bounds(self, file_path):
        """Finds pitch ceiling and floor

        :param file_path: path to the file
        :type file_path: str

        :returns: tuple of pitch ceiling and pitch floor
        :rtype: tuple[float, float]
        """

        signal, sampling_rate = self.args['voice']
        sound: parselmouth.Sound = parselmouth.Sound(signal, sampling_rate)
        # sound: parselmouth.Sound = parselmouth.Sound(file_path)
        # measure pitch ceiling and floor
        broad_pitch: float = sound.to_pitch_ac(
            None, 50, 15, True, 0.03, 0.45, 0.01, 0.35, 0.14, 500
        )
        broad_mean_f0: float = call(broad_pitch, "Get mean", 0, 0, "hertz")  # get mean pitch

        if broad_mean_f0 > 170:
            pitch_floor = 100
            pitch_ceiling = 500
        elif broad_mean_f0 < 170:
            pitch_floor = 50
            pitch_ceiling = 300
        else:
            pitch_floor = 50
            pitch_ceiling = 500
        return pitch_floor, pitch_ceiling

    def pitch_floor(self, file_path):
        """ Returns the pitch floor
        :param file_path: path to the file
        :type file_path: str

        :returns: pitch floor
        :rtype: float
        """
        #sound: parselmouth.Sound = parselmouth.Sound(file_path)
        signal, sampling_rate = self.args['voice']
        sound: parselmouth.Sound = parselmouth.Sound(signal, sampling_rate)
        return self.pitch_bounds(file_path)[0]

    def pitch_ceiling(self, file_path):
        """ Returns the pitch ceiling
        :param file_path: path to the file
        :type file_path: str

        :returns: pitch floor
        :rtype: float
        """
        #sound: parselmouth.Sound = parselmouth.Sound(file_path)
        signal, sampling_rate = self.args['voice']
        sound: parselmouth.Sound = parselmouth.Sound(signal, sampling_rate)
        return self.pitch_bounds(file_path)[1]

    def max_formant(self, file_path, method="praat_manual"):
        """Find the best maximum formant frequency for formant analysis based on voice pitch.

        :param file_path: path to the file
        :type file_path: str
        :param method: method to use for finding the maximum formant frequency, default is praat_manual
        :type method: str

        :returns: maximum formant frequency
        :rtype: float
        """
        try:
            #sound: parselmouth.Sound = parselmouth.Sound(file_path)
            signal, sampling_rate = self.args['voice']
            sound: parselmouth.Sound = parselmouth.Sound(signal, sampling_rate)
            if method == "praat_manual":
                pitch: parselmouth.Pitch = sound.to_pitch(None, 50, 600)  # check pitch to set formant settings
                mean_f0: float = call(pitch, "Get mean", 0, 0, "Hertz")
                max_formant: float
                if 170 <= mean_f0 <= 300:
                    max_formant = 5500
                elif mean_f0 < 170:
                    max_formant = 5000
                else:
                    max_formant = 5500
                return max_formant
            else:
                max_formant = 5500
        except:
            max_formant = 5500
        return max_formant

    def hz_to_bark(self, hertz):
        """Convert Herts to Bark

        :parameter hertz: Frequency in Hz
        :type hertz: Union[float, int]

        :returns bark: The Frequency in Bark
        :rtype bark: float
        """
        bark = 7.0 * np.log(hertz / 650 + np.sqrt(1 + (hertz / 650) ** 2))
        return bark



class MeasureHarmonicityNode(VoicelabNode):
    """Measure the harmonics-to-noise ratio of a sound. This is effetively the Signal-to-Noise Ratio (SNR) of a periodic sound.

    Arguments:
    ---------
        self.args: dict
            Dictionary of arguments for the node.
            self.args['Algorithm'] : str, default=To Harmonicity (cc)'
                Which pitch algorithm to use. Default is Cross Correlation, alternate is Auto Correlation.
            self.args['Timestep'] : float, default 0.01
                The timestep (hop length/time between windows) to use for the analysis.
            self.args["Silence Threshold"]: float, default=0.1,
                The threshold below which a frame is considered silent.
            self.args["Periods per Window"]: float, default=4.5,
                The number of periods per window.
    """
    def __init__(self, *args, **kwargs):

        """
        Args:
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)

        self.args = {
            "Algorithm": (
                "To Harmonicity (cc)",
                ["To Harmonicity (cc)", "To Harmonicity (ac)"],
            ),
            "Timestep": 0.01,
            "Silence Threshold": 0.1,
            "Periods per Window": 4.5,  # This is a new change from 1.0
        }

    def process(self, file_path):
        """This function measures Harmonics to Noise Ratio

        :return: A dictionary of the results or an error message.
        :rtype: dict[str, Union[str, float]]
        """

        try:
            # file_path: str = self.args["file_path"]
            sound: parselmouth.Sound = parselmouth.Sound(file_path)
            algorithm: str = self.args["Algorithm"][0]
            timestep: float = self.args["Timestep"]
            silence_threshold: float = self.args["Silence Threshold"]
            periods_per_window: float = self.args["Periods per Window"]
            pitch_floor: float = 75.0
            harmonicity: float = call(sound, algorithm, timestep, pitch_floor, silence_threshold, periods_per_window,)
            hnr: float = call(harmonicity, "Get mean", 0, 0)
            # return {"Harmonics to Noise Ratio": hnr}
            return hnr
        except Exception as e:
            return {"Harmonics to Noise Ratio": str(e)}

processer = MeasureHarmonicityNode()
path = "C:/Users/wx_Ca/OneDrive - University of Edinburgh/Desktop/Dissertation/LJSpeech-1.1/wavs"
wavs = os.listdir(path)
f = open("C:/Users/wx_Ca/OneDrive - University of Edinburgh/Desktop/Dissertation/LJSpeech-1.1/harmonicity.txt", "a")
for wave_file in wavs:
    samplerate, data = wavfile.read(path + "/" + wave_file)
    print(f'processing {wave_file}')
    f.write(wave_file + " " + str(processer.process(data)) + "\n")
    # print(str(processer.process(data)))
f = open("C:/Users/wx_Ca/OneDrive - University of Edinburgh/Desktop/Dissertation/LJSpeech-1.1/harmonicity.txt", "r")
print(f.read())