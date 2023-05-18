# TODO:
# Include Copyright
# Describes whether the original copyright must be retained.
# Include License
# Including the full text of license in modified software.
# State Changes
# Stating significant changes made to software.
# Include Notice
# If the library has a "NOTICE" file with attribution notes, you must include that NOTICE when you distribute. You may append to this NOTICE file.

from typing import List, Optional, Union

import numpy as np  #TODO: remove
from numpy.fft import fft  #TODO: remove
import torch

from transformers import WhisperFeatureExtractor
from transformers import TensorType, logging
from transformers import BatchFeature

logger = logging.get_logger(__name__)

# TODO: comment better
# TODO: Remove commented code


class WhisperAttackerFeatureExtractor(WhisperFeatureExtractor):
    r"""
    Constructs a Whisper feature extractor.
    This feature extractor inherits from [`WhisperFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.
    This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the `Short Time
    Fourier Transform` which should match pytorch's `torch.stft` equivalent.
    Args:
        feature_size (`int`, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, defaults to 160):
            Length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, defaults to 30):
            The maximum number of chuncks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, defaults to 400):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
    """

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        **kwargs):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.mel_filters = self.get_mel_filters(sampling_rate,
                                                n_fft,
                                                n_mels=feature_size)
        #self.mel_filters = torchaudio.functional.melscale_fbanks(n_freqs=n_fft, f_min=0.0, f_max=28.674781849729335, n_mels=feature_size, sample_rate=sampling_rate, norm="slaney", mel_scale='htk')

    def get_mel_filters(self, sr, n_fft, n_mels=128, dtype=torch.float32):
        # Initialize the weights
        n_mels = int(n_mels)
        weights = torch.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

        # Center freqs of each FFT bin
        fftfreqs = torch.fft.rfftfreq(n=n_fft, d=1.0 / sr)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965

        mels = torch.linspace(min_mel, max_mel, n_mels + 2)

        mels = torch.asarray(mels)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region

        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * torch.exp(logstep *
                                              (mels[log_t] - min_log_mel))

        mel_f = freqs

        #fdiff = torch.diff(mel_f)
        #ramps = np.subtract.outer(mel_f, fftfreqs)

        #for i in range(n_mels):
        #    # lower and upper slopes for all bins
        #    lower = -ramps[i] / fdiff[i]
        #    upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        #    weights[i] = np.maximum(0, np.minimum(lower, upper))

        # create filterbank - based on pytorchaudio's _create_triangular_filterbank
        # calculate the difference between each filter mid point and each stft freq point in hertz
        f_diff = mel_f[1:] - mel_f[:-1]  # (n_filter + 1)
        slopes = mel_f.unsqueeze(0) - fftfreqs.unsqueeze(
            1)  # (n_freqs, n_filter + 2)
        # create overlapping triangles
        zero = torch.zeros(1)
        down_slopes = (-1.0 *
                       slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
        up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
        weights = torch.max(zero, torch.min(down_slopes,
                                            up_slopes))  #fb ->weights

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        #weights *= enorm[:, np.newaxis]
        weights *= enorm.unsqueeze(0)

        return weights

    def fram_wave(self, waveform, center=True):  #TODO: remove
        """
        Transform a raw waveform into a list of smaller waveforms. The window length defines how much of the signal is
        contain in each frame (smalle waveform), while the hope length defines the step between the beginning of each
        new frame.
        Centering is done by reflecting the waveform which is first centered around `frame_idx * hop_length`.
        """
        frames = []
        for i in range(0, waveform.shape[0] + 1, self.hop_length):
            half_window = (self.n_fft - 1) // 2 + 1
            if center:
                start = i - half_window if i > half_window else 0
                end = i + half_window if i < waveform.shape[
                    0] - half_window else waveform.shape[0]

                frame = waveform[start:end]

                if start == 0:
                    padd_width = (-i + half_window, 0)
                    frame = np.pad(frame, pad_width=padd_width, mode="reflect")

                elif end == waveform.shape[0]:
                    padd_width = (0, (i - waveform.shape[0] + half_window))
                    frame = np.pad(frame, pad_width=padd_width, mode="reflect")

            else:
                frame = waveform[i:i + self.n_fft]
                frame_width = frame.shape[0]
                if frame_width < waveform.shape[0]:
                    frame = np.lib.pad(frame,
                                       pad_width=(0, self.n_fft - frame_width),
                                       mode="constant",
                                       constant_values=0)

            frames.append(frame)
        return np.stack(frames, 0)

    def stft(self, frames, window):  #TODO: remove
        """
        Calculates the complex Short-Time Fourier Transform (STFT) of the given framed signal. Should give the same
        results as `torch.stft`.
        """
        frame_size = frames.shape[1]
        fft_size = self.n_fft

        if fft_size is None:
            fft_size = frame_size

        if fft_size < frame_size:
            raise ValueError("FFT size must greater or equal the frame size")
        # number of FFT bins to store
        num_fft_bins = (fft_size >> 1) + 1

        data = np.empty((len(frames), num_fft_bins), dtype=np.complex64)
        fft_signal = np.zeros(fft_size)

        for f, frame in enumerate(frames):
            if window is not None:
                np.multiply(frame, window, out=fft_signal[:frame_size])
            else:
                fft_signal[:frame_size] = frame
            data[f] = fft(fft_signal, axis=0)[:num_fft_bins]
        return data.T

    def _extract_fbank_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-Mel spectrogram of the provided audio, gives similar results whisper's original torch
        implementation with 1e-5 tolerance.
        """
        #window = np.hanning(self.n_fft + 1)[:-1]
        #window = torch.hann_window(self.n_fft + 1, periodic=False)[:-1]
        window = torch.hann_window(self.n_fft, periodic=True)

        #frames = self.fram_wave(waveform)
        #stft = self.stft(frames, window=window)
        #stft = torch.stft(frames, n_fft=self.n_fft) #hop_length=None
        stft = torch.stft(waveform,
                          n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          window=window,
                          center=True,
                          pad_mode='reflect',
                          return_complex=True)
        #print("stft dimension")
        #print(stft.shape)
        magnitudes = torch.abs(stft[:, :-1])**2

        filters = self.mel_filters
        #print(filters.shape)
        #print(magnitudes.shape)
        mel_spec = filters.T @ magnitudes
        #print(mel_spec)

        log_spec = torch.log10(torch.clip(mel_spec, min=1e-10, max=None))
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    def __call__(
            self,
            raw_speech: torch.
        Tensor,  #Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
            truncation: bool = True,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_attention_mask: Optional[bool] = None,
            padding: Optional[str] = "max_length",
            max_length: Optional[int] = None,
            sampling_rate: Optional[int] = None,
            **kwargs) -> torch.Tensor:
        """
        Main method to featurize and prepare for the model one or several sequence(s).
        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            truncation (`bool`, *optional*, default to `True`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*, defaults to None):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.
                [What are attention masks?](../glossary#attention-mask)
                <Tip>
                For Whisper models, `attention_mask` should always be passed for batched inference, to avoid subtle
                bugs.
                </Tip>
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
            padding_value (`float`, defaults to 0.0):
                The value that is used to fill the padding values / vectors.
        """

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        #is_batched = bool(
        #    isinstance(raw_speech, (list, tuple))
        #    and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
        #)

        #if is_batched:
        #    raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
        #elif not is_batched and not isinstance(raw_speech, np.ndarray):
        #    raw_speech = np.asarray(raw_speech, dtype=np.float32)
        #elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
        #    raw_speech = raw_speech.astype(np.float32)

        # always return batch
        #if not is_batched:
        #    raw_speech = [np.asarray([raw_speech]).T]

        #batched_speech = BatchFeature({"input_features": raw_speech})

        # convert into correct format for padding

        #padded_inputs = self.pad(
        #    batched_speech,
        #    padding=padding,
        #    max_length=max_length if max_length else self.n_samples,
        #    truncation=truncation,
        #    pad_to_multiple_of=pad_to_multiple_of,
        #)
        # make sure list is in array format
        #input_features = padded_inputs.get("input_features").transpose(2, 0, 1)

        #input_features = [self._np_extract_fbank_features(waveform) for waveform in input_features[0]]

        #if isinstance(input_features[0], List):
        #   padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]
        #else:
        #   padded_inputs["input_features"] = input_features

        #if return_tensors is not None:
        #padded_inputs = padded_inputs.convert_to_tensors(return_tensors)
        #print("speech format")
        #print(raw_speech.shape)
        input_features = self._extract_fbank_features(raw_speech)

        return BatchFeature({"input_features": input_features})