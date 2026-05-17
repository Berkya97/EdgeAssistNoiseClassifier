"""
Mel Spectrogram Doğrulama Scripti
=================================
Android MelSpectrogramExtractor (düzeltilmiş) ile librosa çıktılarını karşılaştırır.
Sentetik bir sinüs sinyali ve (varsa) gerçek bir ses dosyası ile test eder.

Kullanım:
    python verify_mel.py
    python verify_mel.py --wav dog_bark.wav
"""
import argparse
import numpy as np
import librosa

TARGET_SR = 16000
DURATION_SEC = 1.0
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
FMIN = 0
FMAX = TARGET_SR // 2


def librosa_mel(y: np.ndarray) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=TARGET_SR, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, center=False, fmin=FMIN, fmax=FMAX,
        norm="slaney",
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db


def android_mel_python_replica(y: np.ndarray) -> np.ndarray:
    """Android MelSpectrogramExtractor mantığını Python'da birebir uygular."""
    target_samples = int(TARGET_SR * DURATION_SEC)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    elif len(y) > target_samples:
        start = (len(y) - target_samples) // 2
        y = y[start:start + target_samples]

    n_frames = 1 + (len(y) - N_FFT) // HOP_LENGTH
    hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT))  # periodic
    spectrum_size = N_FFT // 2 + 1

    # librosa-uyumlu mel filterbank (Slaney mel skalası + ramps/fdiff)
    fft_freqs = np.arange(spectrum_size) * TARGET_SR / N_FFT

    def _hz2mel(hz):
        f_sp = 200.0 / 3.0
        min_log_hz, min_log_mel = 1000.0, 1000.0 / f_sp
        logstep = np.log(6.4) / 27.0
        return np.where(np.asarray(hz) < min_log_hz,
                        np.asarray(hz) / f_sp,
                        min_log_mel + np.log(np.maximum(np.asarray(hz), 1e-10) / min_log_hz) / logstep)

    def _mel2hz(mel):
        f_sp = 200.0 / 3.0
        min_log_hz, min_log_mel = 1000.0, 1000.0 / f_sp
        logstep = np.log(6.4) / 27.0
        return np.where(np.asarray(mel) < min_log_mel,
                        np.asarray(mel) * f_sp,
                        min_log_hz * np.exp(logstep * (np.asarray(mel) - min_log_mel)))

    low_mel = float(_hz2mel(FMIN))
    high_mel = float(_hz2mel(FMAX))
    mel_points = np.linspace(low_mel, high_mel, N_MELS + 2)
    mel_f = _mel2hz(mel_points)

    fdiff = np.diff(mel_f)
    ramps = mel_f[:, np.newaxis] - fft_freqs[np.newaxis, :]

    filterbank = np.zeros((N_MELS, spectrum_size))
    for m in range(N_MELS):
        lower = -ramps[m] / fdiff[m]
        upper = ramps[m + 2] / fdiff[m + 1]
        filterbank[m] = np.maximum(0, np.minimum(lower, upper))
        bw = mel_f[m + 2] - mel_f[m]
        if bw > 1e-10:
            filterbank[m] *= 2.0 / bw

    # power spectra + mel (librosa: |FFT|², n_fft bölmesi yok)
    mel_energies = np.zeros((N_MELS, n_frames))
    for i in range(n_frames):
        start = i * HOP_LENGTH
        frame = y[start:start + N_FFT] * hann
        fft_out = np.fft.rfft(frame, n=N_FFT)
        power = np.abs(fft_out) ** 2
        mel_energies[:, i] = filterbank @ power

    # power_to_db (ref=max, amin=1e-10, top_db=80)
    amin = 1e-10
    ref = max(amin, mel_energies.max())
    mel_db = 10.0 * np.log10(np.maximum(mel_energies, amin) / ref)
    mel_db = np.maximum(mel_db, mel_db.max() - 80.0)
    return mel_db


def compare(name: str, y: np.ndarray):
    lib = librosa_mel(y)
    android = android_mel_python_replica(y)

    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    print(f"  librosa  shape: {lib.shape}  min={lib.min():.4f}  max={lib.max():.4f}")
    print(f"  android  shape: {android.shape}  min={android.min():.4f}  max={android.max():.4f}")

    if lib.shape != android.shape:
        min_mels = min(lib.shape[0], android.shape[0])
        min_time = min(lib.shape[1], android.shape[1])
        print(f"  !! Shape uyumsuz — ({min_mels},{min_time}) ile kırparak karşılaştırıyorum")
        lib = lib[:min_mels, :min_time]
        android = android[:min_mels, :min_time]

    diff = np.abs(lib - android)
    print(f"  Fark (abs): mean={diff.mean():.6f}  max={diff.max():.6f}  std={diff.std():.6f}")

    if diff.max() < 0.01:
        print("  SONUC: MÜKEMMEL uyum (<0.01 dB max fark)")
    elif diff.max() < 0.1:
        print("  SONUC: Çok iyi uyum (<0.1 dB max fark)")
    elif diff.max() < 1.0:
        print("  SONUC: Kabul edilebilir (<1.0 dB max fark)")
    else:
        print("  SONUC: UYUMSUZLUK VAR (>1.0 dB max fark)")

    # İlk 5 mel band, ilk 5 frame detaylı
    print("\n  İlk 3 mel band x ilk 5 frame karşılaştırması:")
    for m in range(min(3, lib.shape[0])):
        lib_vals = "  ".join(f"{lib[m, t]:8.3f}" for t in range(min(5, lib.shape[1])))
        and_vals = "  ".join(f"{android[m, t]:8.3f}" for t in range(min(5, android.shape[1])))
        print(f"    mel[{m}] librosa:  {lib_vals}")
        print(f"    mel[{m}] android:  {and_vals}")


def debug_filterbank():
    """librosa filterbank vs android replica filterbank karşılaştırması."""
    print(f"\n{'='*60}")
    print("Filterbank doğrulama")
    print(f"{'='*60}")

    # librosa filterbank
    lib_fb = librosa.filters.mel(
        sr=TARGET_SR, n_fft=N_FFT, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, norm="slaney"
    )

    # Android replica filterbank (Slaney mel skalası)
    spectrum_size = N_FFT // 2 + 1
    fft_freqs = np.arange(spectrum_size) * TARGET_SR / N_FFT

    def _h2m(hz):
        f_sp = 200.0/3.0; mlh = 1000.0; mlm = mlh/f_sp; ls = np.log(6.4)/27.0
        return np.where(np.asarray(hz)<mlh, np.asarray(hz)/f_sp, mlm+np.log(np.maximum(np.asarray(hz),1e-10)/mlh)/ls)
    def _m2h(mel):
        f_sp = 200.0/3.0; mlh = 1000.0; mlm = mlh/f_sp; ls = np.log(6.4)/27.0
        return np.where(np.asarray(mel)<mlm, np.asarray(mel)*f_sp, mlh*np.exp(ls*(np.asarray(mel)-mlm)))

    low_mel = float(_h2m(FMIN))
    high_mel = float(_h2m(FMAX))
    mel_points = np.linspace(low_mel, high_mel, N_MELS + 2)
    mel_f = _m2h(mel_points)
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, np.newaxis] - fft_freqs[np.newaxis, :]
    and_fb = np.zeros((N_MELS, spectrum_size))
    for m in range(N_MELS):
        lower = -ramps[m] / fdiff[m]
        upper = ramps[m + 2] / fdiff[m + 1]
        and_fb[m] = np.maximum(0, np.minimum(lower, upper))
        bw = mel_f[m + 2] - mel_f[m]
        if bw > 1e-10:
            and_fb[m] *= 2.0 / bw

    diff = np.abs(lib_fb - and_fb)
    print(f"  librosa  shape: {lib_fb.shape}  sum[0]={lib_fb[0].sum():.6f}")
    print(f"  android  shape: {and_fb.shape}  sum[0]={and_fb[0].sum():.6f}")
    print(f"  Fark: mean={diff.mean():.10f}  max={diff.max():.10f}")

    if diff.max() < 1e-6:
        print("  SONUC: Filterbank MÜKEMMEL uyum")
    else:
        print(f"  SONUC: Filterbank farkı var! max={diff.max():.10f}")
        worst_m = np.unravel_index(diff.argmax(), diff.shape)
        print(f"  En büyük fark: mel[{worst_m[0]}][{worst_m[1]}] lib={lib_fb[worst_m]:.10f} and={and_fb[worst_m]:.10f}")


def debug_stft():
    """librosa STFT vs android replica STFT karşılaştırması."""
    print(f"\n{'='*60}")
    print("STFT / Power Spectrum doğrulama")
    print(f"{'='*60}")

    t = np.arange(int(TARGET_SR * DURATION_SEC)) / TARGET_SR
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # librosa STFT
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, center=False, window='hann')
    lib_power = np.abs(S) ** 2  # shape: (513, 30)

    # Android replica (periodic Hann)
    hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT))
    n_frames = 1 + (len(y) - N_FFT) // HOP_LENGTH
    and_power = np.zeros((N_FFT // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * HOP_LENGTH
        frame = y[start:start + N_FFT] * hann
        fft_out = np.fft.rfft(frame, n=N_FFT)
        and_power[:, i] = np.abs(fft_out) ** 2

    diff = np.abs(lib_power - and_power)
    print(f"  librosa  shape: {lib_power.shape}  max={lib_power.max():.6f}")
    print(f"  android  shape: {and_power.shape}  max={and_power.max():.6f}")
    print(f"  Fark: mean={diff.mean():.10f}  max={diff.max():.10f}")

    if diff.max() < 1e-4:
        print("  SONUC: Power spectrum MÜKEMMEL uyum")
    else:
        print(f"  SONUC: Power spectrum farkı var!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, default=None, help="Opsiyonel gerçek ses dosyası")
    args = parser.parse_args()

    debug_filterbank()
    debug_stft()

    # Test 1: 440 Hz sinüs
    t = np.arange(int(TARGET_SR * DURATION_SEC)) / TARGET_SR
    sine_440 = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    compare("440 Hz sinüs", sine_440)

    # Test 2: Beyaz gürültü
    rng = np.random.default_rng(42)
    noise = (0.1 * rng.standard_normal(int(TARGET_SR * DURATION_SEC))).astype(np.float32)
    compare("Beyaz gürültü", noise)

    # Test 3: Gerçek ses dosyası
    if args.wav:
        y, _ = librosa.load(args.wav, sr=TARGET_SR, mono=True)
        target = int(TARGET_SR * DURATION_SEC)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        elif len(y) > target:
            start = (len(y) - target) // 2
            y = y[start:start + target]
        compare(f"Dosya: {args.wav}", y)

    # Ek: train.py'deki mel shape
    print(f"\n{'='*60}")
    print("Time frames doğrulama:")
    n_frames_expected = 1 + (TARGET_SR - N_FFT) // HOP_LENGTH
    lib_shape = librosa_mel(sine_440).shape
    print(f"  Formül:  1 + (16000 - 1024) / 512 = {n_frames_expected}")
    print(f"  librosa: {lib_shape[1]}")
    print(f"  android: {n_frames_expected}")
    if lib_shape[1] == n_frames_expected:
        print("  SONUC: Time frames UYUMLU")
    else:
        print(f"  SONUC: UYUMSUZ! librosa={lib_shape[1]} vs beklenen={n_frames_expected}")


if __name__ == "__main__":
    main()
