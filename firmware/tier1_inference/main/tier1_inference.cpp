#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "esp_log.h"
#include "esp_timer.h"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "autoencoder_model.h"
#include "test_reference.h"

static const char *TAG = "tier1";

// Tensor arena: scratch memory for TFLite Micro's intermediate tensors.
// Conservative initial guess; we log actual usage and tune later.
constexpr int kTensorArenaSize = 20 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Tier 1 inference firmware starting");

    // --- Load and validate the model ---
    const tflite::Model *model = tflite::GetModel(g_autoencoder_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Schema version mismatch: model=%d, runtime=%d",
                 (int)model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    ESP_LOGI(TAG, "Model loaded (schema v%d, %u bytes)",
             (int)model->version(), (unsigned)g_autoencoder_model_len);

    // --- Set up op resolver and interpreter ---
    // Using AllOpsResolver for initial bring-up. Once inference works,
    // we'll switch to MicroMutableOpResolver and register only the ops
    // our model actually uses, to reduce binary size.
    // Register only the ops our model uses (smaller binary than AllOpsResolver).
    // Each AddXxx call returns kTfLiteOk on success; we abort if any fail.
  // Register the ops used by the Conv1DTranspose-based autoencoder.
    // Tile is no longer needed; TRANSPOSE_CONV replaces the UpSampling1D pattern.
    static tflite::MicroMutableOpResolver<10> resolver;
    if (resolver.AddConv2D()         != kTfLiteOk) { ESP_LOGE(TAG, "Resolver: Conv2D");         return; }
    if (resolver.AddMaxPool2D()      != kTfLiteOk) { ESP_LOGE(TAG, "Resolver: MaxPool2D");      return; }
    if (resolver.AddTransposeConv()  != kTfLiteOk) { ESP_LOGE(TAG, "Resolver: TransposeConv");  return; }
    if (resolver.AddReshape()        != kTfLiteOk) { ESP_LOGE(TAG, "Resolver: Reshape");        return; }
    if (resolver.AddExpandDims()     != kTfLiteOk) { ESP_LOGE(TAG, "Resolver: ExpandDims");     return; }
    if (resolver.AddAdd()            != kTfLiteOk) { ESP_LOGE(TAG, "Resolver: Add");            return; }
    if (resolver.AddConcatenation()  != kTfLiteOk) { ESP_LOGE(TAG, "Resolver: Concatenation");  return; }
    if (resolver.AddShape()          != kTfLiteOk) { ESP_LOGE(TAG, "Resolver: Shape");          return; }
    if (resolver.AddStridedSlice()   != kTfLiteOk) { ESP_LOGE(TAG, "Resolver: StridedSlice");   return; }

    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed — tensor arena likely too small");
        return;
    }

    ESP_LOGI(TAG, "Arena: %u bytes used of %d allocated",
             (unsigned)interpreter.arena_used_bytes(), kTensorArenaSize);

    // --- Inspect input/output tensors ---
    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    ESP_LOGI(TAG, "Input:  shape=[%d,%d,%d], dtype=%d",
             input->dims->data[0], input->dims->data[1], input->dims->data[2],
             (int)input->type);
    ESP_LOGI(TAG, "Output: shape=[%d,%d,%d], dtype=%d",
             output->dims->data[0], output->dims->data[1], output->dims->data[2],
             (int)output->type);

    // --- Copy test input into the model ---
    memcpy(input->data.int8, kTestInput, 60);

    // --- Run inference, time it ---
    int64_t t0 = esp_timer_get_time();
    TfLiteStatus status = interpreter.Invoke();
    int64_t t1 = esp_timer_get_time();

    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed: %d", (int)status);
        return;
    }
    ESP_LOGI(TAG, "Inference completed in %lld us", (long long)(t1 - t0));

    // --- Compare output to expected ---
    int exact_matches = 0;
    int max_diff = 0;
    int sum_abs_diff = 0;
    for (int i = 0; i < 60; i++) {
        int diff = abs((int)output->data.int8[i] - (int)kExpectedOutput[i]);
        if (diff == 0) exact_matches++;
        if (diff > max_diff) max_diff = diff;
        sum_abs_diff += diff;
    }
    ESP_LOGI(TAG, "Output check: %d/60 exact matches, max diff = %d, total diff = %d",
             exact_matches, max_diff, sum_abs_diff);

    // --- Print first 12 bytes of output and expected for visual confirmation ---
    printf("Got:      ");
    for (int i = 0; i < 12; i++) printf("%4d ", (int)output->data.int8[i]);
    printf("...\n");
    printf("Expected: ");
    for (int i = 0; i < 12; i++) printf("%4d ", (int)kExpectedOutput[i]);
    printf("...\n");

    ESP_LOGI(TAG, "Done");
}