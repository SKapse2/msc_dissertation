// Auto-generated test reference values for ESP32 integration test
// Source: notebook 05_tflite_conversion.ipynb
// Test window index: 3834
// Window end: 2014-02-09 12:25:00

#ifndef TEST_REFERENCE_H_
#define TEST_REFERENCE_H_

#include <stdint.h>

const int8_t kTestInput[60] = {
    -116, -110, -110, -115, -113, -114, -115, -114, -115, -110, -110, -110,
    -115, -115, -117, -114, -110, -115, -112, -112, -116, -113, -117, -110,
    -115, -112, -111, -114, -116, -116, -112, -113, -116, -117, -112, -111,
    -113, -110, -113, -112, -114, -117, -112, -116, -112, -111, -116, -110,
    -115, -112, -112, -112,  -98,  -72,  -44,  -10,    7,   23,   33,   44,
};

const int8_t kExpectedOutput[60] = {
    -112, -110, -114, -114, -112, -109, -112, -115, -112, -112, -112, -114,
    -114, -113, -116, -117, -111, -111, -111, -114, -113, -111, -113, -116,
    -111, -110, -111, -118, -116, -111, -113, -118, -114, -115, -112, -114,
    -112, -108, -112, -115, -113, -113, -113, -116, -113, -109, -113, -114,
    -114, -113, -115, -115,  -90,  -71,  -48,  -18,   11,   27,   37,   39,
};

const float kExpectedReconstructionMSE = 0.015170f;
const float kExpectedConfidence = 1.536143f;
const float kThresholdInt8 = 0.005981f;

#endif  // TEST_REFERENCE_H_