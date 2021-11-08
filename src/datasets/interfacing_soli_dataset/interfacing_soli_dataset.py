"""interfacing_soli_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import h5py

_DESCRIPTION = """
Radar gesture recognition dataset from the paper  
> Interacting with Soli: Exploring Fine-Grained DynamicGesture Recognition in the Radio-Frequency Spectrum

No preprocessing applied.
"""

_CITATION = """
@inproceedings{wang2016interacting,
  title={Interacting with soli: Exploring fine-grained dynamic gesture recognition in the radio-frequency spectrum},
  author={Wang, Saiwen and Song, Jie and Lien, Jaime and Poupyrev, Ivan and Hilliges, Otmar},
  booktitle={Proceedings of the 29th Annual Symposium on User Interface Software and Technology},
  pages={851--860},
  year={2016},
  organization={ACM}
}
"""

GESTURES = ["PinchIndex", "PalmTilt", "FingerSlider", "PinchPinky", "SlowSwipe", "FastSwipe", "Push", "Pull", "FingerRub", "Circle", "PalmHold"]
USERS = [2, 3, 5, 6, 8, 9, 10, 11, 12 ,13]

class InterfacingSoliDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for interfacing_soli_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'rdm': tfds.features.Tensor(shape=(None, 32, 32, 4), dtype=tf.float32),
            'label': tfds.features.ClassLabel(names=GESTURES),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('rdm', 'label'),  # Set to `None` to disable
        homepage='https://github.com/simonwsw/deep-soli',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract('https://polybox.ethz.ch/index.php/s/wG93iTUdvRU8EaT/download')

    person_splits = {}
    for count, i_participant in enumerate(USERS):
      p_data = self._generate_examples(path / 'dsp', i_participant)
      person_splits['p' + str(count)] = p_data

    return person_splits

  def _generate_examples(self, path, i_participant):
    """Yields examples."""
    for i_gesture in range(11):
      for i_instance in range(25):
        filename = '{}_{:d}_{:d}'.format(i_gesture, i_participant, i_instance)
        with h5py.File(path / (filename + '.h5'), 'r') as f:
          data_ch0 = f['ch{}'.format(0)][()]
          length = data_ch0.shape[0]

          data_ch0 = data_ch0.reshape((length,32,32))
          data_ch1 = f['ch{}'.format(1)][()].reshape((length,32,32))
          data_ch2 = f['ch{}'.format(2)][()].reshape((length,32,32))
          data_ch3 = f['ch{}'.format(3)][()].reshape((length,32,32))

          dataStacked = np.stack((data_ch0, data_ch1, data_ch2, data_ch3), axis=-1)

          yield filename, {
              'rdm': dataStacked,
              'label': GESTURES[i_gesture],
          }
