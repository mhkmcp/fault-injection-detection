# Image classifier with a built-in fault monitor: portable bundle

This folder holds a trained CIFAR-10 image classifier and a monitor that watches the model's
internal activations. If the stored weights get corrupted, for example by bit flips in memory,
the monitor raises a flag. You don't need the training data, the training code or PyTorch.

## Pick the file for your device

| Device | File | Monitor? |
|---|---|---|
| Any device with a browser (laptop, phone, tablet) | `ui/` running `model.onnx` | yes |
| Linux, Windows, macOS, Raspberry Pi, Jetson (Python or C/C++) | `model.onnx` | yes |
| Android, iOS, ONNX Runtime minimal builds | `formats/model.ort` | yes |
| Same, classifier only, about 2× faster and 3× smaller | `formats/classifier_int8.onnx` / `.ort` | no |
| Android / Linux with LiteRT (TFLite) | `formats/classifier_fp32.tflite`, `classifier_fp16.tflite` | no |
| Microcontrollers (ESP32-S3, STM32H7, Cortex-M7) via TFLite Micro | `formats/classifier_int8.tflite`, or `classifier_int8_tflite.h` compiled into the firmware | no |

`manifest.json` → `formats` lists each file with its accuracy on 2,000 held-out test pictures,
its size, its SHA-256 and its runtime. The model with the monitor needs more compute. The
manifest records how much time the monitor adds on the export machine (`overhead`). Re-measure it
on your own device.

## Browser UI

```bash
python run.py --ui                    # opens http://localhost:8000/ui/
python run.py --ui --host 0.0.0.0     # reach it from a phone on the same network
```

You only need Python's standard library to serve it. The inference runs inside the browser with
ONNX Runtime Web, which is included in `ui/`, so it works offline. Any static web server rooted
at this folder works too. Opening `ui/index.html` by double-clicking it does not: browsers block
loading the model from `file://`.

In the UI you can:
- pick a test picture or upload or photograph your own, and see the answer and the monitor's
  verdict;
- choose how many false alarms are acceptable: 5 %, 1 % or 0.1 % of pictures on a healthy model;
- see the picture **layer by layer**: every observed layer has its own detector (one SVDD per
  layer) and threshold. Each layer's score is shown against its threshold, the first layer over
  it is marked, and ⚡ marks the layer fed by the weights you flipped;
- **damage the memory**, with the same controls as the research demo:
  - **Injection in weight:** the number format the weights are stored in — FP32, FP16, FP8
    (E4M3), INT2, INT4 or INT8. Choosing one converts every weight to that format (integers:
    symmetric, one scale per output channel) and shows the accuracy change; the flips then hit
    that format's bits. The monitor was calibrated on FP32, so a low-precision format can raise
    false alarms on its own; the scores compare each flip against the same format before flips.
  - **Where:** early, middle or late layers.
  - **Which bit:** for floats sign, exponent top (drastic), exponent low, mantissa high,
    mantissa low (subtle); for integers sign, top bit, high, middle and lowest bits.
  - **How many flips:** 1, 4 or 16.

  "Surprise me" picks a random drastic combination, and "Advanced: flip one exact bit" targets a
  single weight. Flips add up until "Reset model". You then see whether the answer changes and
  whether the monitor notices. The file on disk is never touched.
- **see scores on 200 test pictures** (20 per class), recomputed whenever the model changes:
  - **Classifier:** accuracy, plus precision, recall, F1 and F2, macro-averaged and per class.
    Each is shown next to the original model's value.
  - **Fault monitor:** the confusion matrix (TP/FP/FN/TN), recall (detection rate), precision,
    F1, F2, false-alarm rate and accuracy. Recall is shown separately for pictures whose answer
    the fault changed and pictures whose answer it left unchanged.
  - **Which layer notices the fault?** The network drawn as a chain of layers. The layer fed by
    the flipped weights has an orange frame and the flip count; each layer shows how many
    corrupted pictures it flags, its false alarms on the model before flips, and how often it is
    the first layer to alarm. The layer that alarms first most often is outlined in blue. So you
    can see where the fault was injected and where the damage becomes visible. Each layer may
    false-alarm at the allowed share on its own, so "any layer" false-alarms more often than the
    main monitor; the table under it gives both.

  F2 counts recall four times as much as precision. That suits a fault monitor, where a missed
  corruption is worse than a false alarm. The same pictures are run through the original model
  (should not be flagged) and the corrupted model (should be flagged). Precision, F1 and F2
  therefore assume a 50/50 mix; recall and the false-alarm rate don't. `manifest.json` → `metrics` has the classifier's scores on all
  2,000 test pictures for reference.
- **compare detector designs** (section 5, when the bundle was exported after
  `fidnn granularity`): one layer, every layer separately then combined, a group of layers, or all
  layers in one SVDD. Choose the kind of fault (answer wrong or broken / answer unchanged) and the
  score (detection rate at the false-alarm setting from section 2, or AUROC). The page lists the
  key facts with 95 % intervals, draws every design as a bar with its cost per picture, and
  optionally every possible group of layers. These are recorded results from the fault sweep,
  not computed in the browser.

## Command line

Install with `pip install -r requirements.txt` (onnxruntime, numpy, Pillow), then:

```bash
python run.py --selftest                          # checks model.onnx is unaltered and matches
python run.py cat.png ship.jpg                    # class, confidence, monitor verdict,
                                                  # first layer over its own threshold
python run.py --alpha 0.001 photos/*.png          # stricter: fewer false alarms
python run.py --window 8 3 frames/*.png           # ALARM when 3 of the last 8 are flagged
python run.py --model formats/classifier_int8.onnx cat.png   # any .onnx/.ort file here
```

## From your own code

**ONNX Runtime (Python, C, C++, C#, Java, JavaScript).** Input `images` is uint8 of shape
(N, 32, 32, 3), RGB, 0–255. Normalisation is built into the model.

```python
import onnxruntime as ort
s = ort.InferenceSession("model.onnx")
logits, probs, pred, score, alarms, tap_scores, tap_alarms = s.run(
    None, {"images": images_uint8_nhwc})
# alarms[:, k] corresponds to manifest["monitor"]["alphas"][k]
# tap_scores[:, t] / tap_alarms[:, t, k]: layer t of manifest["monitor"]["per_layer"]["taps"]
# (bundles exported before per-layer monitoring return only the first five)
```

**TFLite / LiteRT.** Input is (1, 32, 32, 3) RGB, pixel values 0–255:
- float32 files take raw 0–255 floats;
- the INT8 file takes int8, quantised with the input tensor's scale and zero point:
  `q = round(pixel / scale + zero_point)`. For this model that is simply `pixel − 128`.
  `manifest.json` records both.

**Microcontrollers.**
1. Add `classifier_int8_tflite.h` to the firmware.
2. Create a `tflite::MicroInterpreter` over `classifier_int8_tflite`.
3. Register exactly these operators: Conv2D, Add, Pad, StridedSlice, Sub, Mul, Mean and
   FullyConnected.
4. Give it a tensor arena of about 100 KB, and increase it until `AllocateTensors()` succeeds.

## Limits

- **Microcontrollers get the classifier only.** The monitor is too heavy for them; it needs about
  1 MB and sorts every activation of the first layer.
- **Not an adversarial defence.** The monitor detects weight corruption that changes the model's
  behaviour. It is not a defence against adversarial images.
- **CIFAR-10-like images only.** The false-alarm rate holds only for images like CIFAR-10.
  Unusual photos may be flagged for reasons unrelated to faults.
- **One training run.** The model comes from a single training seed. It is a demonstration, not a
  research result.
