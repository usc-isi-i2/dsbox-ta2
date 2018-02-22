
import numpy as np
import typing

from d3m_metadata import container, hyperparams, metadata
from d3m_metadata.metadata import PrimitiveMetadata
from primitive_interfaces.transformer import TransformerPrimitiveBase
from primitive_interfaces.base import CallResult

from d3m.primitives.bbn.time_series import (
    ChannelAverager, SignalDither, SignalFramer, SignalMFCC, 
    UniformSegmentation,  SegmentCurveFitter)

class Hyperparams(hyperparams.Hyperparams):
    sampling_rate = hyperparams.UniformInt(lower=16000, upper=66000, default=44100)


Inputs = container.List[container.ndarray]
Outputs = container.ndarray

class BBNAudioPrimitiveWrapper(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    '''BBN Audio Primitive Wrapper'''

    metadata = PrimitiveMetadata({
        "id": "bbn_audio_pipeline",
        "version": "v0.1.0",
        "name": "BBN Audio Featurization Wrapper",
        "description": "Featurization of Audio Data",
        "python_path": "d3m.primitives.dsbox.BBNAudioPrimitiveWrapper",
        "primitive_family": "DATA_PREPROCESSING",
        "algorithm_types": [ "AUDIO_STREAM_MANIPULATION" ],
        "source": {
            "name": 'ISI',
            "uris": [ 'git+https://github.com/usc-isi-i2/dsbox-ta2' ]
            },
        ### Automatically generated
        # "primitive_code"
        # "original_python_path"
        # "schema"
        # "structural_type"
        ### Optional
        "keywords": [ "feature_extraction",  "timeseries"],
        # "installation": [ config.INSTALLATION ],
        #"location_uris": [],
        #"precondition": [],
        #"effects": [],
        #"hyperparms_to_tune": []
        })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, 
                 docker_containers: typing.Dict[str, str] = None) -> None:
        self.hyperparams = hyperparams
        self.random_seed = random_seed
        self.docker_containers = docker_containers
        self._resampling_rate = 16000
        self._feature_pipeline = self._setup_bbn_primitives()

    def set_training_data(self) -> None:  # type: ignore
        """
        A noop.
        """

        return

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        """
        A noop.
        """

        return

    def get_params(self) -> None:
        """
        A noop.
        """

        return None

    def set_params(self, *, params: None) -> None:
        """
        A noop.
        """

        return

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        res = self._extract_feats(inputs,
                                  fext_pipeline = self._fexture_pipeline,
                                  resampling_rate = self._resampling_rate)
        return CallResult(res, True, 1)

    def _setup_bbn_primitives(self):
        channel_mixer = ChannelAverager(
            hyperparams = ChannelAverager.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].defaults())
        dither = SignalDither(
            hyperparams = SignalDither.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].defaults())
        framer_hyperparams = SignalFramer.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        framer_custom_hyperparams = dict()
        framer_custom_hyperparams['sampling_rate'] = self._resampling_rate

        #if args.fext_wlen is not None:
        #  framer_custom_hyperparams['frame_length_s'] = args.fext_wlen
        #if args.fext_rel_wshift is not None:
        #  framer_custom_hyperparams['frame_shift_s'] = args.fext_rel_wshift*args.fext_wlen

        framer = SignalFramer(
            hyperparams = framer_hyperparams(
                framer_hyperparams.defaults(), **framer_custom_hyperparams))
        mfcc_hyperparams = SignalMFCC.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        mfcc_custom_hyperparams = dict()
        mfcc_custom_hyperparams['sampling_rate'] = self._resampling_rate

        #if args.fext_mfcc_ceps is not None:
        #  mfcc_custom_hyperparams['num_ceps'] = args.fext_mfcc_ceps

        mfcc = SignalMFCC(
            hyperparams = mfcc_hyperparams(
                mfcc_hyperparams.defaults(), **mfcc_custom_hyperparams))
        segm = UniformSegmentation(
            hyperparams = UniformSegmentation.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].defaults())

        segm_fitter_hyperparams = SegmentCurveFitter.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        segm_fitter_custom_hyperparams = dict()

        #if args.fext_poly_deg is not None:
        #  segm_fitter_custom_hyperparams['deg'] = args.fext_poly_deg

        segm_fitter = SegmentCurveFitter(
            hyperparams = segm_fitter_hyperparams(
                segm_fitter_hyperparams.defaults(), **segm_fitter_custom_hyperparams
            )
                    )
        self._fexture_pipeline = [ channel_mixer, dither, framer, mfcc, segm, segm_fitter ]

    def _extract_feats(self, inputs, fext_pipeline = None,
                      resampling_rate = None):
        features = container.List()
        i = 0
        for idx, row in inputs.iterrows():
            if row[0] == '':
                features.append(np.array([]))
                continue
            #filename = os.path.join(audio_dir, row[0])
            #print(filename)
            audio_clip = inputs['filename'].values[0][0]
            sampling_rate = resampling_rate
            if 'start' in inputs.columns and 'end' in inputs.columns:
                start = int(sampling_rate * float(inputs.loc[idx]['start']))
                end = int(sampling_rate * float(inputs.loc[idx]['end']))
                audio_clip = audio_clip[start:end]

            audio = container.List[container.ndarray]([audio_clip], {
                        'schema': metadata.CONTAINER_SCHEMA_VERSION,
                        'structural_type': container.List[container.ndarray],
                        'dimension': {
                            'length': 1
                        }
                        })
            audio.metadata = audio.metadata.update((metadata.ALL_ELEMENTS,),
                                { 'structural_type': container.ndarray,
                                  'semantic_type': 'http://schema.org/AudioObject' })
            # sampling_rate is not supported by D3M metadata v2018.1.5
            #audio.metadata = audio.metadata.update((0,),
            #                    { 'sampling_rate': sampling_rate })

            last_output = audio
            #print(audio_clip)
            for fext_step in fext_pipeline:
                #print('xxxxxxxxxxxxxxx')
                #print(fext_step)
                product = fext_step.produce(inputs = last_output)
                last_output = product.value

            features.append(last_output[0])

            i+=1
        return features
