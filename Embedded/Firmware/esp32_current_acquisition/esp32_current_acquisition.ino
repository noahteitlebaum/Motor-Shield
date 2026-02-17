#include <Arduino.h>
#include <SPI.h>

// ===================== USER CONFIG =====================
static const uint32_t SERIAL_BAUD = 921600;

// ESP32 VSPI default pins (you can change if needed)
static const int PIN_SCK  = 18;
static const int PIN_MISO = 19;
static const int PIN_MOSI = 23;

// ADS1256 module pins (set to how you wired it)
static const int PIN_CS   = 5;   // CS
static const int PIN_DRDY = 4;   // DRDY (data ready)
static const int PIN_RST  = 2;   // RST (optional but recommended)
static const int PIN_PWDN = -1;  // PWDN (optional)

// Sampling target
static const uint32_t FS_PER_CH = 5000;              // 5 kHz per phase
static const uint32_t TICK_HZ   = FS_PER_CH * 3;     // 15 kHz tick (U,V,W interleaved)

// ACS712-20A sensitivity (typical): 100 mV/A = 0.1 V/A
static const float ACS_SENS_V_PER_A = 0.100f;

// If ACS712 powered at 5V, centered ~2.5V. We'll auto-calibrate per-channel offset.
static const bool AUTO_ZERO_CAL = true;
static const int  CAL_SAMPLES_PER_CH = 1000;         // per channel (motor OFF during cal)

// ADS1256 reference voltage depends on your module.
// Many modules use 2.5V reference (chip has internal ref buffer options; board may tie to 2.5V ref).
// If your board uses 2.5V reference, leave as 2.5. If it uses 5V (rare), set to 5.0.
static const float ADC_VREF = 2.500f;

// =======================================================

// ADS1256 commands
static const uint8_t CMD_WAKEUP  = 0x00;
static const uint8_t CMD_RDATA   = 0x01;
static const uint8_t CMD_RDATAC  = 0x03;
static const uint8_t CMD_SDATAC  = 0x0F;
static const uint8_t CMD_RREG    = 0x10;
static const uint8_t CMD_WREG    = 0x50;
static const uint8_t CMD_SELFCAL = 0xF0;
static const uint8_t CMD_SYNC    = 0xFC;
static const uint8_t CMD_RESET   = 0xFE;

// ADS1256 registers
static const uint8_t REG_STATUS = 0x00;
static const uint8_t REG_MUX    = 0x01;
static const uint8_t REG_ADCON  = 0x02;
static const uint8_t REG_DRATE  = 0x03;
static const uint8_t REG_IO     = 0x04;

// MUX: positive channel in high nibble, negative channel in low nibble
// We'll use AINCOM (common) as negative input.
// ADS1256 MUX codes: AIN0..AIN7 = 0..7, AINCOM = 8
static const uint8_t MUX_AINCOM = 0x08;

// Data rate codes (from ADS1256 datasheet). 30000 SPS = 0xF0
static const uint8_t DRATE_30000SPS = 0xF0;

// Gain bits in ADCON (PGA[2:0] in lower bits): 1=000, 2=001, 4=010, etc.
// We'll use gain=1 for max input range.
static const uint8_t PGA_GAIN_1 = 0x00;

// SPI settings (ADS1256 supports up to 7.68MHz; use a safe value)
static SPIClass *vspi = nullptr;
static SPISettings spiSettings(2000000, MSBFIRST, SPI_MODE1);

// Timer
hw_timer_t *timer0 = nullptr;
volatile bool tickFlag = false;

// Channel sequencing
static const uint8_t CH_LIST[3] = {0, 1, 2}; // AIN0, AIN1, AIN2
volatile uint8_t chIndex = 0;

// Zero offsets (in volts)
float v0[3] = {2.5f, 2.5f, 2.5f};

// ---------- Low-level helpers ----------
static inline void csLow()  { digitalWrite(PIN_CS, LOW); }
static inline void csHigh() { digitalWrite(PIN_CS, HIGH); }

static inline void spiBegin() {
  vspi->beginTransaction(spiSettings);
  csLow();
}
static inline void spiEnd() {
  csHigh();
  vspi->endTransaction();
}

static inline uint8_t spiXfer(uint8_t x) {
  return vspi->transfer(x);
}

static void adsSendCommand(uint8_t cmd) {
  spiBegin();
  spiXfer(cmd);
  spiEnd();
  // Small command settle time
  delayMicroseconds(5);
}

static void adsWriteReg(uint8_t reg, uint8_t value) {
  spiBegin();
  spiXfer(CMD_WREG | (reg & 0x0F));
  spiXfer(0x00);     // write 1 register
  spiXfer(value);
  spiEnd();
  delayMicroseconds(5);
}

static uint8_t adsReadReg(uint8_t reg) {
  spiBegin();
  spiXfer(CMD_RREG | (reg & 0x0F));
  spiXfer(0x00);     // read 1 register
  delayMicroseconds(5);
  uint8_t v = spiXfer(0xFF);
  spiEnd();
  return v;
}

static inline bool drdyLow() {
  return digitalRead(PIN_DRDY) == LOW;
}

// Read 24-bit conversion result (blocking on DRDY)
static int32_t adsReadData24() {
  // Wait until data ready
  while (!drdyLow()) { /* spin */ }

  spiBegin();
  spiXfer(CMD_RDATA);
  // datasheet: wait t6 (~50 * 1/clk) => a few us is fine
  delayMicroseconds(10);

  uint8_t b0 = spiXfer(0xFF);
  uint8_t b1 = spiXfer(0xFF);
  uint8_t b2 = spiXfer(0xFF);
  spiEnd();

  int32_t raw = ((int32_t)b0 << 16) | ((int32_t)b1 << 8) | b2;

  // Sign extend 24-bit to 32-bit
  if (raw & 0x800000) raw |= 0xFF000000;

  return raw;
}

// Set MUX to single-ended channel vs AINCOM and sync
static void adsSetChannel(uint8_t ain_pos) {
  uint8_t mux = ((ain_pos & 0x0F) << 4) | (MUX_AINCOM & 0x0F);
  adsWriteReg(REG_MUX, mux);

  // SYNC + WAKEUP is recommended after MUX changes
  adsSendCommand(CMD_SYNC);
  adsSendCommand(CMD_WAKEUP);
}

// Convert raw code to volts (bipolar)
// Full-scale code is +/- 0x7FFFFF for +/-Vref/gain (approx).
static float codeToVolts(int32_t code) {
  // ADS1256 is bipolar. LSB = Vref/(gain*2^23)
  const float denom = (float)(1UL << 23); // 2^23
  return ( (float)code * (ADC_VREF / denom) ) / 1.0f; // gain=1
}

// Timer ISR: set a flag every tick
void IRAM_ATTR onTimer() {
  tickFlag = true;
}

// Zero calibration: read each channel many times and average its voltage offset
static void zeroCalibrate() {
  long double sumV[3] = {0,0,0};

  // Ensure stopped continuous mode, do self-cal
  adsSendCommand(CMD_SDATAC);
  adsSendCommand(CMD_SELFCAL);
  delay(5);

  for (int ch = 0; ch < 3; ch++) {
    adsSetChannel(CH_LIST[ch]);
    // Throw away a few samples after MUX change
    for (int k = 0; k < 5; k++) (void)adsReadData24();

    for (int i = 0; i < CAL_SAMPLES_PER_CH; i++) {
      int32_t code = adsReadData24();
      sumV[ch] += (long double)codeToVolts(code);
    }
    v0[ch] = (float)(sumV[ch] / (long double)CAL_SAMPLES_PER_CH);
  }

  Serial.print("# zero offsets (V): ");
  Serial.print(v0[0], 6); Serial.print(",");
  Serial.print(v0[1], 6); Serial.print(",");
  Serial.println(v0[2], 6);
}

static void adsInit() {
  pinMode(PIN_CS, OUTPUT);
  csHigh();

  pinMode(PIN_DRDY, INPUT_PULLUP);

  if (PIN_RST >= 0) {
    pinMode(PIN_RST, OUTPUT);
    digitalWrite(PIN_RST, HIGH);
    delay(2);
    digitalWrite(PIN_RST, LOW);
    delay(5);
    digitalWrite(PIN_RST, HIGH);
    delay(5);
  } else {
    adsSendCommand(CMD_RESET);
    delay(5);
  }

  // Stop continuous mode if any
  adsSendCommand(CMD_SDATAC);

  // STATUS: enable buffer? Most modules work fine with buffer ON or OFF.
  // We'll set: MSB first, auto-cal disable (we call SELFCAL), buffer ON (bit 1).
  // STATUS bits: ORDER(3)=0 (MSB first), ACAL(2)=0, BUFEN(1)=1
  adsWriteReg(REG_STATUS, 0x02);

  // ADCON: Clock out off, sensor detect off, PGA=1
  adsWriteReg(REG_ADCON, PGA_GAIN_1);

  // DRATE: 30,000 SPS
  adsWriteReg(REG_DRATE, DRATE_30000SPS);

  // Self calibrate once
  adsSendCommand(CMD_SELFCAL);
  delay(5);

  // Start with channel 0
  adsSetChannel(CH_LIST[0]);

  // Optional: read back regs for sanity
  uint8_t status = adsReadReg(REG_STATUS);
  uint8_t adcon  = adsReadReg(REG_ADCON);
  uint8_t drate  = adsReadReg(REG_DRATE);

  Serial.print("# ADS1256 STATUS=0x"); Serial.println(status, HEX);
  Serial.print("# ADS1256 ADCON=0x");  Serial.println(adcon, HEX);
  Serial.print("# ADS1256 DRATE=0x");  Serial.println(drate, HEX);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(200);

  // SPI init
  vspi = new SPIClass(VSPI);
  vspi->begin(PIN_SCK, PIN_MISO, PIN_MOSI, PIN_CS);

  adsInit();

  if (AUTO_ZERO_CAL) {
    Serial.println("# Calibrating zero offsets... ensure motor OFF");
    zeroCalibrate();
  }

  // CSV header
  Serial.println("t_us,ch,raw,volts,amps");

  // Timer setup: 15 kHz tick
  timer0 = timerBegin(0, 80, true); // 80 MHz / 80 = 1 MHz (1 tick = 1 us)
  timerAttachInterrupt(timer0, &onTimer, true);
  timerAlarmWrite(timer0, 1000000UL / TICK_HZ, true); // period in us
  timerAlarmEnable(timer0);
}

void loop() {
  if (!tickFlag) return;
  tickFlag = false;

  // Select which channel this tick corresponds to
  uint8_t idx = chIndex;
  chIndex = (chIndex + 1) % 3;

  uint8_t ain = CH_LIST[idx];

  // Change MUX to desired channel
  adsSetChannel(ain);

  // Read conversion (blocking on DRDY)
  int32_t raw = adsReadData24();
  float v = codeToVolts(raw);
  float i = (v - v0[idx]) / ACS_SENS_V_PER_A;

  uint32_t t = (uint32_t)micros();

  // Stream: time, channel index (0/1/2), raw code, volts, amps
  Serial.print(t);
  Serial.print(",");
  Serial.print((int)idx);
  Serial.print(",");
  Serial.print(raw);
  Serial.print(",");
  Serial.print(v, 6);
  Serial.print(",");
  Serial.println(i, 4);
}
