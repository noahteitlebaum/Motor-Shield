#include <Arduino.h>

// --------------------------------------------------
// Must match Python config
// --------------------------------------------------
const uint32_t BAUD_RATE = 921600;
const uint32_t SAMPLE_RATE_HZ = 1000;
const uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;

const size_t PHASE_COUNT = 3;
const size_t BLOCK_SAMPLES = 256;

// Python expects uint16 ADC-like values
uint16_t sampleBuffer[BLOCK_SAMPLES][PHASE_COUNT];

// --------------------------------------------------
// Signal settings
// --------------------------------------------------
const float ADC_MIDPOINT = 2048.0f;     // 12-bit midpoint
const float CURRENT_AMPLITUDE = 500.0f; // trapezoid half-amplitude in ADC counts
const float NOISE_AMPLITUDE = 60.0f;    // random noise amplitude in ADC counts

// Electrical frequency of the simulated motor current
const float ELECTRICAL_FREQ_HZ = 8.0f;

// --------------------------------------------------
// Trapezoidal waveform helper
// Input: phase in [0,1)
// Output: waveform in [-1,1]
// --------------------------------------------------
float trapezoidWave(float phase) {
  // keep phase in [0,1)
  while (phase < 0.0f) phase += 1.0f;
  while (phase >= 1.0f) phase -= 1.0f;

  // 6-step style trapezoid:
  // 0.00 - 1/6   : ramp  0 -> +1
  // 1/6  - 3/6   : hold +1
  // 3/6  - 4/6   : ramp +1 -> -1
  // 4/6  - 5/6   : hold -1
  // 5/6  - 1.00  : ramp -1 -> 0

  if (phase < 1.0f / 6.0f) {
    return 6.0f * phase;
  }
  else if (phase < 3.0f / 6.0f) {
    return 1.0f;
  }
  else if (phase < 4.0f / 6.0f) {
    float local = (phase - 3.0f / 6.0f) * 6.0f; // 0 to 1
    return 1.0f - 2.0f * local;                 // +1 to -1
  }
  else if (phase < 5.0f / 6.0f) {
    return -1.0f;
  }
  else {
    float local = (phase - 5.0f / 6.0f) * 6.0f; // 0 to 1
    return -1.0f + local;                       // -1 to 0
  }
}

// --------------------------------------------------
// Add bounded random noise
// --------------------------------------------------
float noisySample(float idealValue) {
  long noise = random((long)-NOISE_AMPLITUDE, (long)NOISE_AMPLITUDE + 1);
  return idealValue + (float)noise;
}

// --------------------------------------------------
// Clamp to 12-bit ADC range and convert to uint16
// --------------------------------------------------
uint16_t clampToAdc(float x) {
  if (x < 0.0f) x = 0.0f;
  if (x > 4095.0f) x = 4095.0f;
  return (uint16_t)(x + 0.5f);
}

void setup() {
  Serial.begin(BAUD_RATE);
  delay(1000);

  // seed RNG for noise
  randomSeed(esp_random());
}

void loop() {
  static uint32_t nextSampleTimeUs = micros();
  static uint32_t globalSampleIndex = 0;

  for (size_t i = 0; i < BLOCK_SAMPLES; i++) {
    while ((int32_t)(micros() - nextSampleTimeUs) < 0) {
      // wait for next sample instant
    }
    nextSampleTimeUs += SAMPLE_PERIOD_US;

    float t = (float)globalSampleIndex / (float)SAMPLE_RATE_HZ;
    float basePhase = ELECTRICAL_FREQ_HZ * t;

    // Convert to [0,1) electrical phase
    float phaseA = basePhase - floorf(basePhase);
    float phaseB = basePhase - 1.0f / 3.0f;
    float phaseC = basePhase - 2.0f / 3.0f;

    phaseB = phaseB - floorf(phaseB);
    phaseC = phaseC - floorf(phaseC);

    // Ideal 3-phase trapezoidal currents
    float iaIdeal = ADC_MIDPOINT + CURRENT_AMPLITUDE * trapezoidWave(phaseA);
    float ibIdeal = ADC_MIDPOINT + CURRENT_AMPLITUDE * trapezoidWave(phaseB);
    float icIdeal = ADC_MIDPOINT + CURRENT_AMPLITUDE * trapezoidWave(phaseC);

    // Add measurement noise
    float iaNoisy = noisySample(iaIdeal);
    float ibNoisy = noisySample(ibIdeal);
    float icNoisy = noisySample(icIdeal);

    // Store as uint16 ADC-like values
    sampleBuffer[i][0] = clampToAdc(iaNoisy);
    sampleBuffer[i][1] = clampToAdc(ibNoisy);
    sampleBuffer[i][2] = clampToAdc(icNoisy);

    globalSampleIndex++;
  }

  // Send one full block as binary
  Serial.write(
    reinterpret_cast<uint8_t*>(sampleBuffer),
    sizeof(sampleBuffer)
  );
}