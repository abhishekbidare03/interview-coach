// Microphone capture worklet: resample to 16 kHz, convert to PCM16, ship upstream.
//
// This runs on the audio render thread, so it must stay cheap and allocation-light.
// Resampling here rather than on the main thread keeps the WebSocket payload at
// 32 KB/s instead of ~96 KB/s at the typical 48 kHz device rate, and Silero VAD
// requires exactly 16 kHz anyway.

const TARGET_RATE = 16000;
// 512 samples @ 16 kHz is one Silero VAD frame. Sending whole frames means the
// server never has to buffer a partial one in the common case.
const FRAME = 512;

class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE; // `sampleRate` is a worklet global
    this.pos = 0;
    this.out = new Int16Array(FRAME);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0]?.[0];
    if (!ch) return true;

    // Linear interpolation. Whisper is trained on far noisier signal than the
    // artifacts this introduces, so a windowed-sinc resampler would be effort
    // spent where it cannot be heard.
    while (this.pos < ch.length) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      const a = ch[i];
      const b = i + 1 < ch.length ? ch[i + 1] : a;
      const s = Math.max(-1, Math.min(1, a + (b - a) * frac));
      this.out[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;

      if (this.n === FRAME) {
        // Transfer the buffer to avoid a copy, then allocate a fresh one.
        const buf = this.out.buffer;
        this.port.postMessage(buf, [buf]);
        this.out = new Int16Array(FRAME);
        this.n = 0;
      }
      this.pos += this.ratio;
    }
    this.pos -= ch.length;
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
