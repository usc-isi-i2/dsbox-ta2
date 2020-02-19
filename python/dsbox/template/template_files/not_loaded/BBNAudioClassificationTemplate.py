from dsbox.template.template import DSBoxTemplate 
from d3m.metadata.problem import TaskKeyword 
from dsbox.template.template_steps import TemplateSteps 
from dsbox.schema import SpecializedProblem 
import typing 
import numpy as np  # type: ignore 
class BBNAudioClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "BBN_Audio_Classification_Template",
            "taskType": {TaskKeyword.CLASSIFICATION.name},
            "taskSubtype": {TaskKeyword.MULTICLASS.name},
            "inputType": "audio",
            "output": "model_step",
            "steps": [
                {
                    "name": "readtarget_step",
                    "primitives":["d3m.primitives.bbn.time_series.TargetsReader"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "readaudio_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.bbn.time_series.AudioReader",
                        "hyperparameters":
                            {
                                "read_as_mono": [(True)],
                                "resampling_rate": [(16000.0)],
                            }
                    }],
                    "inputs": ["template_input"]
                },
                {
                    "name": "channel_step",
                    "primitives": ["d3m.primitives.bbn.time_series.ChannelAverager"],
                    "inputs": ["readaudio_step"]
                },
                {
                    "name": "signaldither_step",
                    "primitives": [{"primitive": "d3m.primitives.bbn.time_series.SignalDither",
                                    "hyperparameters": {
                                        "level": [(0.0001)],
                                        "reseed": [(True)]
                                    }
                                    }],
                    "inputs": ["channel_step"]
                },
                {
                    "name": "signalframer_step",
                    "primitives": [{"primitive": "d3m.primitives.bbn.time_series.SignalFramer",
                                    "hyperparameters": {
                                        "flatten_output": [(False)],
                                        "frame_length_s": [(0.025)],
                                        "frame_shift_s": [(0.01)]
                                    }
                                    }],
                    "inputs": ["signaldither_step"]
                },
                {
                    "name": "MFCC_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.bbn.time_series.SignalMFCC",
                        "hyperparameters": {
                            "cep_lifter": [(22.0)],
                            "frame_mean_norm": [(False)],
                            "nfft": [(None)],
                            "num_ceps": [(20)],
                            "num_chans": [(20)],
                            "preemcoef": [(None)],
                            "use_power": [(False)]
                        }
                    }],
                    "inputs": ["signalframer_step"]
                },
                {
                    "name": "vectorextractor_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.bbn.time_series.IVectorExtractor",
                        "hyperparameters": {
                            "gmm_covariance_type": [("diag")],
                            "ivec_dim": [(100)],
                            "max_gmm_iter": [(20)],
                            "num_gauss": [(32)],
                            "num_ivec_iter": [(7)]
                        }
                    }],
                    "inputs": ["MFCC_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.bbn.sklearn_wrap.BBNMLPClassifier",
                        "hyperparameters": {
                            "activation": [("relu")],
                            "add_index_columns": [(True)],
                            "alpha": [(0.0001)],
                            "beta_1": [(0.9)],
                            "beta_2": [(0.999)],
                            "early_stopping": [(True)],
                            "epsilon": [(1e-8)],
                            "exclude_columns": [([])],
                            # "hidden_layer_sizes":[([30,30])],
                            "learning_rate": [("constant")],
                            "learning_rate_init": [(0.01)],
                            "max_iter": [(200)],
                            "return_result": [("replace")],
                            "shuffle": [(True)],
                            "solver": [("adam")],
                            "tol": [(0.0001)],
                            "use_columns": [([])],
                            "use_semantic_types": [(False)],
                            "warm_start": [(False)]
                        }
                    }],
                    "inputs": ["vectorextractor_step", "readtarget_step"]

                }
            ]
        }


