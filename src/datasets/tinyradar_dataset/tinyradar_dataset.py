"""tinyradar_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift

_DESCRIPTION = """
Radar gesture recognition dataset from the paper
> TinyRadarNN: Combining Spatial and Temporal Convolutional Neural Networks for Embedded Gesture Recognition with Short Range Radars

Preprocessing: FFT on windows of length 16 with sweep 8.
"""

_CITATION = """
@misc{scherer2020tinyradarnn,
  title={TinyRadarNN: Combining Spatial and Temporal Convolutional Neural Networks for Embedded Gesture Recognition with Short Range Radars},
  author={Moritz Scherer and Michele Magno and Jonas Erb and Philipp Mayer and Manuel Eggimann and Luca Benini},
  year={2020},
  eprint={2006.16281},
  archivePrefix={arXiv},
  primaryClass={eess.SP}
}
"""

GESTURES = ["Circle", "FastSwipeRL", "FingerRub", "FingerSlider","NoHand", "PalmHold", "PalmTilt", "PinchIndex", "PinchPinky","Pull", "Push", "SlowSwipeRL"]
minSweeps = 32
N_SESSIONS = 5
N_RECORDINGS = 7
POOL_SIZE = 4

def lines2list(pathToFile):
    with open(pathToFile, "r") as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

class TinyradarDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for tinyradar_dataset dataset."""

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
            'rdm': tfds.features.Tensor(shape=(None, 16, 492//POOL_SIZE, 2), dtype=tf.float32),
            'label': tfds.features.ClassLabel(names=GESTURES),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('rdm', 'label'),  # Set to `None` to disable
        homepage='https://tinyradar.ethz.ch/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    data_path = dl_manager.download_and_extract('https://tinyradar.ethz.ch/wp-content/uploads/2020/09/11G.zip')
    data_path = data_path / '11G' / 'data'
    person_splits = {}

    for offset in [0, 4, 8, 12]:
      person_splits[str(offset) +'_p0'] = self._generate_examples(data_path / 'p0_1', offset)
      for i in range(1,26):
        p_name = 'p' + str(i)
        p_data = self._generate_examples(data_path / p_name, offset)
        person_splits[str(offset) + '_' + p_name] = p_data

    return person_splits

  def _generate_examples(self, p_path, fft_offset):
    """Yields examples."""
    for gesture in GESTURES:
      for sess in range(N_SESSIONS):
        for recording in range(N_RECORDINGS):
          instance_name = gesture + '/sess_' + str(sess) + '/' + str(recording)
          base_path = str(p_path / instance_name)
          info = lines2list(base_path + "_info.txt")

          numberOfSweeps = int(info[1])
          sweepFrequency = int(info[2])
          sensorRangePoints0 = int(info[14])
          sensorRangePoints1 = int(info[15])
          if (minSweeps > numberOfSweeps):
            continue
          if (sensorRangePoints0 != sensorRangePoints1):
            print('Unequal range points:', sensorRangePoints0, sensorRangePoints1, base_path)

          dataBinarySensor0 = np.fromfile(base_path + "_s0.dat", dtype=np.complex64)
          dataBinarySensor1 = np.fromfile(base_path + "_s1.dat", dtype=np.complex64)

          dataBinarySensor0 = dataBinarySensor0.reshape((numberOfSweeps, sensorRangePoints0))
          dataBinarySensor1 = dataBinarySensor1.reshape((numberOfSweeps, sensorRangePoints1))

          dataBinaryStacked = np.stack((dataBinarySensor0, dataBinarySensor1), axis=-1)

          dataShape = dataBinaryStacked.shape
          dataReshape = np.reshape(dataBinaryStacked, (dataShape[0], int(dataShape[1]/POOL_SIZE), POOL_SIZE, dataShape[2]))
          dataReshape = np.squeeze(np.mean(dataReshape, axis=2))

          #dataBinaryStacked = [sweeps, range, sensor]
          #rdm = [windows, vel, range, sensor]
          window_legnth = 16
          window_step = 16
          #number_of_windows = int(numberOfSweeps / window_step) -1
          number_of_windows = int((numberOfSweeps-fft_offset) / window_legnth)
          if number_of_windows > 30:
            number_of_windows = 30

          fft_data = [dataReshape[fft_offset+i_window*window_step:fft_offset+i_window*window_step+window_legnth, :, :] for i_window in range(number_of_windows)]
          rdm = abs(fftshift(fft(fft_data, axis=1), axes=1))

          yield instance_name, {
            'rdm': rdm,
            'label': gesture
          }

