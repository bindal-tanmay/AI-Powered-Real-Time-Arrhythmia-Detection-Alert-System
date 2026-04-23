

// ── Pin definitions ─────────────────────────────────────────
#define ECG_PIN     A0      // AD8232 OUTPUT pin → Arduino A0
#define LO_PLUS     10      // Leads-off detection LO+
#define LO_MINUS    11      // Leads-off detection LO-
#define SAMPLE_RATE 360     // Hz — matches MIT-BIH dataset sample rate

// ── Sampling timer ──────────────────────────────────────────
unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE; // microseconds


// High-pass filter (removes baseline wander, cutoff ~0.5Hz)
//    y[n] = x[n] - x[n-1] + 0.995 * y[n-1]
float hp_x_prev = 0;
float hp_y_prev = 0;

// 2. Low-pass filter (removes high-freq noise, cutoff ~40Hz)
//    Simple 2nd order: y[n] = x[n] + 2*x[n-1] + x[n-2] - 2*y[n-1] - y[n-2]  (scaled /4)
float lp_x1 = 0, lp_x2 = 0;
float lp_y1 = 0, lp_y2 = 0;

// 3. Notch filter at 50Hz (removes powerline interference)
//    IIR notch: coefficients pre-computed for fs=360Hz, f0=50Hz, Q=30
//    H(z) = (1 - 2cos(w0)z^-1 + z^-2) / (1 - 2r*cos(w0)z^-1 + r^2*z^-2)
const float NOTCH_W0  = 2.0 * 3.14159265 * 50.0 / SAMPLE_RATE;  // 0.8727 rad
const float NOTCH_R   = 0.97;
const float NOTCH_B0  = 1.0;
const float NOTCH_B1  = -2.0 * cos(NOTCH_W0);   // computed at init
const float NOTCH_B2  = 1.0;
float notch_A1, notch_A2;
float notch_x1 = 0, notch_x2 = 0;
float notch_y1 = 0, notch_y2 = 0;
float nb1;   // stored notch_B1 value

// 4. Derivative filter (Pan-Tompkins step 3)
//    y[n] = (1/8)(−x[n−2] − 2x[n−1] + 2x[n+1] + x[n+2])
//    Approximated causally:
//    y[n] = (2x[n] + x[n-1] - x[n-3] - 2x[n-4]) / 8
float d_buf[5] = {0,0,0,0,0};

// 5. Squaring + Moving Window Integration (Pan-Tompkins)
#define MWI_SIZE 36          // 0.1s window at 360Hz
float mwi_buf[MWI_SIZE];
int   mwi_idx = 0;
float mwi_sum = 0;

// ── R-peak detection (adaptive threshold) ───────────────────
float threshold    = 512.0;
float peak_val     = 0;
int   rr_interval  = 0;
int   last_r_idx   = 0;
int   sample_count = 0;
bool  r_detected   = false;

// ── Output buffer ────────────────────────────────────────────
// We send 3 values over Serial: raw, filtered, r_peak_flag
// Format: "RAW,FILTERED,RPEAK\n"

// ============================================================
void setup() {
  Serial.begin(115200);
  pinMode(LO_PLUS,  INPUT);
  pinMode(LO_MINUS, INPUT);
  pinMode(13, OUTPUT);    // ← ADD THIS LINE
  // ... rest of setup


  // Pre-compute notch coefficients
  nb1      = -2.0 * cos(NOTCH_W0);
  notch_A1 = -2.0 * NOTCH_R * cos(NOTCH_W0);
  notch_A2 = NOTCH_R * NOTCH_R;

  // Zero MWI buffer
  for (int i = 0; i < MWI_SIZE; i++) mwi_buf[i] = 0;

  // Header for Python to parse
  Serial.println("RAW,FILTERED,RPEAK");
}

// ============================================================
void loop() {
  unsigned long now = micros();
  if (now - lastSampleTime < SAMPLE_INTERVAL_US) return;
  lastSampleTime = now;

  // ── Leads-off check ─────────────────────────────────────
  // if (digitalRead(LO_PLUS) == 1 || digitalRead(LO_MINUS) == 1) {
  //   Serial.println("LEADS_OFF");
  //   return;
  // }

  // ── Read raw ADC (0–1023) ────────────────────────────────
  float raw = (float)analogRead(ECG_PIN);

  // ── FILTER CHAIN ─────────────────────────────────────────

  // Step 1: High-pass filter (baseline wander removal)
  float hp_y = raw - hp_x_prev + 0.995f * hp_y_prev;
  hp_x_prev  = raw;
  hp_y_prev  = hp_y;

  // Step 2: Notch filter at 50Hz
  float notch_x0 = hp_y;
  float notch_y0 = NOTCH_B0 * notch_x0
                 + nb1       * notch_x1
                 + NOTCH_B2  * notch_x2
                 - notch_A1  * notch_y1
                 - notch_A2  * notch_y2;
  notch_x2 = notch_x1; notch_x1 = notch_x0;
  notch_y2 = notch_y1; notch_y1 = notch_y0;

  // Step 3: Low-pass filter (~40Hz cutoff)
  float lp_y = (notch_y0 + 2*lp_x1 + lp_x2 - 2*lp_y1 - lp_y2) / 4.0f;
  // Guard against integrator blow-up
  if (lp_y >  800) lp_y =  800;
  if (lp_y < -800) lp_y = -800;
  lp_x2 = lp_x1; lp_x1 = notch_y0;
  lp_y2 = lp_y1; lp_y1 = lp_y;

  float filtered = lp_y;

  // ── PAN-TOMPKINS R-PEAK DETECTION ───────────────────────

  // Step 4: Derivative
  d_buf[4] = d_buf[3]; d_buf[3] = d_buf[2];
  d_buf[2] = d_buf[1]; d_buf[1] = d_buf[0];
  d_buf[0] = filtered;
  float deriv = (2*d_buf[0] + d_buf[1] - d_buf[3] - 2*d_buf[4]) / 8.0f;

  // Step 5: Square
  float squared = deriv * deriv;

  // Step 6: Moving window integration
  mwi_sum -= mwi_buf[mwi_idx];
  mwi_buf[mwi_idx] = squared;
  mwi_sum += squared;
  mwi_idx = (mwi_idx + 1) % MWI_SIZE;
  float integrated = mwi_sum / MWI_SIZE;

  // Step 7: Adaptive thresholding
  if (integrated > peak_val) peak_val = integrated;
  threshold = 0.5f * peak_val;
  peak_val *= 0.995f;   // decay

  int r_flag = 0;
  if (integrated > threshold && !r_detected && (sample_count - last_r_idx) > 50) {
    r_flag      = 1;
    r_detected  = true;
    rr_interval = sample_count - last_r_idx;
    last_r_idx  = sample_count;
    digitalWrite(13, HIGH);  
  } else if (integrated < threshold * 0.5f) {
    r_detected = false;
     digitalWrite(13, LOW);    // ← ADD THIS LINE (LED OFF)
  }

  // ── Serial output ────────────────────────────────────────
  // Format: RAW,FILTERED,RPEAK  (RPEAK=1 on R-peak, else 0)
  // Serial.print((int)raw);
  // Serial.print(',');
  // Serial.print(filtered, 2);
  // Serial.print(',');
  // Serial.println(r_flag);
  Serial.print(raw);
Serial.print(" ");
Serial.print(filtered);
Serial.print(" ");
Serial.println(r_flag * 500);  // amplify R-peak for visibility

  sample_count++;
}

// Blink test — built in LED on pin 13
// If this works → Arduino is perfectly fine

// void setup() {
//   pinMode(13, OUTPUT);
// }

// void loop() {
//   digitalWrite(13, HIGH);   // LED ON
//   delay(500);               
//   digitalWrite(13, LOW);    // LED OFF
//   delay(500);               
// }
