# 🧭 Beyond the Build — design reflections

Everything in the [README](../README.md) and the [Build Report](../BUILD-REPORT.md) is built,
tested, and measured. This page is deliberately different: **none of it is implemented.** It is
the thinking that doesn't fit in a test suite — where a system like this should go next, and how
I'd decide. Wherever an idea connects to something this build actually measured, I say which,
because that's the point: these aren't ideas I read somewhere, they're the ideas the
measurements ask for.

---

## 1. ASR is not the product — the conversation is

A drive-thru customer never sees a transcript. They experience a *conversation* — its speed,
its accuracy, and how gracefully it recovers when something goes wrong. The transcript is an
intermediate artifact. That reframing changes what you optimize.

### 1.1 Latency is a UX metric wearing an engineering costume

Human turn-taking runs on gaps of a few hundred milliseconds; past about a second of silence,
people assume the other party didn't hear and start repeating themselves — which produces
overlapped speech, which produces worse ASR, which produces more repetition. Latency failures
*compound into accuracy failures*.

This build's budget (≤ 3 s end-to-end for a 5 s utterance, RTF ≤ 0.6 on CPU int8) was an
engineering envelope. In production I'd invert the derivation: start from the conversational
budget — how long a gap keeps the exchange feeling natural across the *whole multi-turn order* —
and let that dictate model size, quantization, and how much preprocessing the pipeline can
afford. A model that wins 0.5 WER but costs 800 ms per turn is very plausibly a net loss for
the user. That trade should be measured, not assumed (see §1.3).

### 1.2 Design the output for the reader downstream, not for a human

The transcript's real consumer is an NLU layer or an ordering agent, not a person. That has
consequences this build already gestures at and production should take further:

- **Structured output over flat strings.** The pipeline already emits a decision trace
  (VAD verdict, noise class, applied policy, guard actions, keydetector corrections). The
  natural next step is emitting **entity candidates with provenance**: raw span, corrected
  span, rule or lexicon match that produced it, confidence. An agent that knows *"MegaCheddar
  Blast" came from a fuzzy lexicon match at 0.82* can confirm intelligently instead of
  guessing blindly.
- **The keydetector is a proto-NLU bridge.** Deterministically normalizing toward menu
  entities *is* entity linking, done with rules so it ships in minutes and reverses instantly.
  Growing it means growing the contract with the layer above, not just the rule count.
- **Don't destroy disfluencies — expose them.** "Eh… no, mejor sin queso" contains a repair
  signal an agent needs. Aggressive cleanup optimizes for a human reader that doesn't exist.
- **Context should flow backwards too.** The prompt builder already injects menu terms into
  Whisper's decoding. The next version of that idea is *order-state-conditioned* biasing: what
  has already been ordered changes what's acoustically likely next ("¿algo más?" → sizes,
  drinks, "no, eso es todo").

### 1.3 Measure friction, not just transcription

WER is the field's metric, not the user's. This build already took one step away from it — KER
(did the *menu-critical* words survive?) drives 40% of the NDI, because "quiero una ~~coca~~"
and "~~quiero una~~ coca" have the same WER and wildly different consequences.

Production should finish that journey. The KPIs I'd gate promotions on:

| KPI | What it captures |
|-----|------------------|
| **Repair rate** | how often the customer has to repeat or correct — the purest friction signal |
| **Confirmation-loop length** | turns burned on "did you say…?" before the order advances |
| **Order-edit rate at the window** | errors that survived the whole conversation — the expensive ones |
| **Abandonment / handoff-to-human rate** | the experience failed completely |

Two disciplines around these:

- **Slice everything.** Per-language gates are already a hard rule in this repo; the next
  mandatory slice is **accent and variety**. "Spanish" alone spans Rioplatense, Mexican,
  Caribbean; a model can improve on average while regressing for one community, and the
  aggregate will hide it. An accent-stratified eval set is a fairness gate — same mechanics as
  every other gate here, just a different partition of the data.
- **Mixed methods, literally.** The flywheel's human review queue is not just a labeling
  station — it's a qualitative window into real failure. The quantitative telemetry says
  *where* conversations break; listening to fifty of them says *why*. A weekly habit of
  reading the review queue like field notes would steer the roadmap better than any dashboard.

---

## 2. The noise you never let in — the acoustic front-end

This build fights noise **after** the microphone, in software — and one of its clearest
measured lessons is humbling: several denoisers that *sound* cleaner made transcription
*worse*, and the generated mitigation policy auto-rejected them. Post-capture cleanup has a
ceiling. The bigger, cheaper wins live **upstream, before the waveform ever reaches the
model** — in the microphone, the geometry, and the first ten milliseconds of DSP.

The [NDI](../README.md#-does-it-actually-work) names the enemies — babble and vehicle noise
dominate in both languages. These are the four research lines I'd pursue against them, each
with the latency budget it must respect:

| Line | Attacks | Added latency | Core tech |
|------|---------|---------------|-----------|
| 1. Differential two-mic capture | stationary + ambient | < 10 ms | hypercardioid + reference mic, NLMS adaptive filter |
| 2. MEMS array + beamforming | off-axis babble | ≈ 0 (on-DSP) | 4–8 mic board, XMOS/TI DSP, ODAS |
| 3. Per-site acoustic fingerprinting | fryers, extractors, HVAC | 10–30 ms | WebRTC APM spectral subtraction, RNNoise |
| 4. Psychoacoustic preprocessing | broadband masking | < 5 ms | multiband downward expander + linear-phase EQ (JUCE / SoX / FFmpeg) |

### Line 1 — Differential capture: subtract the world

A hypercardioid mic aims at the car window and captures *voice + near noise*; a second,
omnidirectional mic faces away and captures *the world*. A fast adaptive filter (NLMS-class)
subtracts the second from the first in real time. No ML, no model weights, single-digit
milliseconds. The research question is **geometry**: place the reference mic too close and it
captures voice too — the filter then cancels the very signal you want (the classic
self-cancellation failure). Critical distance, and physical *acoustic shadowing* between the
capsules, are where the engineering actually lives.

### Line 2 — Beamforming: aim the microphone like a lens

An array of 4–8 MEMS capsules, millimeters apart, measures the microsecond differences in
arrival time and synthesizes a virtual *beam* pointed at the speaker's head — smart-speaker
technology, industrialized for an outdoor totem (weatherproofing and vibration isolation are
the unglamorous hard parts). Run on a dedicated DSP, it adds effectively zero latency.

The honest caveat, and the reason this line doesn't retire the others: beamforming kills
**off-axis** babble — the kitchen bleed, the sidewalk conversation. The kid in the back seat
yelling *"¡y papas fritas!"* is nearly **on-axis**, inside the beam. In-cabin babble — the
NDI's most damaging noise class — survives the best front-end, which is exactly why the
model- and rules-side axes of this build stay necessary no matter how good the hardware gets.

### Line 3 — Fingerprint the restaurant, subtract it forever

Fryers, extractor hoods, and HVAC are *stationary colored noise* — stable spectral signatures
that barely change for hours. So: profile the site's spectrum during the moments nobody is
speaking, and continuously subtract that fingerprint from the incoming signal (WebRTC's
audio-processing stack and RNNoise — a sub-1 MB RNN/DSP hybrid under 20 ms — are the mature
open implementations).

This line composes beautifully with what's already built. The **VAD gate that guards Whisper
also delimits the noise-only windows** you'd profile from — the sensor is already in place.
And the NDI says which noise *classes* hurt globally, while the fingerprint says what *this
store* actually emits: together they yield a **per-site mitigation policy** — the same
measured-policy mechanism this build generates per noise type, taken one level deeper.

### Line 4 — Process for a machine listener, not a human one

Broadcast radio solved intelligibility-under-noise decades ago: multiband downward expansion
(a frequency-aware gate that ducks bands where speech isn't) plus EQ focused on the
300 Hz–3.4 kHz intelligibility band. The twist is that the listener here is Whisper, not a
person — and *what "sounds good" to Whisper is an empirical question*, not an audio-engineering
convention. Implementations are cheap (JUCE natively, or SoX / FFmpeg `compand` + `biquad`
chains at the edge, under 5 ms); the research is entirely in the tuning-against-WER loop.

### The rule that survives the hardware

Every one of these lines is subject to the same law this build enforced in software: **a
front-end stage is enabled only if it measurably improves WER/KER — never because it sounds
cleaner.** And the harness to enforce it already exists: the Noise Lab's SNR-accurate mixer
and sensitivity matrix were built to compare denoisers, but they compare *microphone
configurations and DSP chains* just as well. Point the eval matrix at hardware candidates and
the NDI tells you, per noise class and per language, whether the physics actually helped.

*Trust ΔWER over intuition* doesn't stop at the microphone.

---

## 3. Open questions I'd bring to a team

1. **Where is the real bottleneck today** — capture/cleaning, ASR accuracy, or NLU context
   management? The NDI answers that question per noise class; the same measurement discipline
   should exist per pipeline *stage*, so investment follows evidence instead of fashion.
2. **Which production KPI actually correlates with a frustrated customer** — and is any model
   promotion currently gated on it, or only on WER?
3. **What does an accent-stratified eval set look like** for a bilingual US + LATAM
   deployment — and who decides what "good enough" means per slice?

---

*Back to the [README](../README.md) · the measured record is in the
[Build Report](../BUILD-REPORT.md).*
