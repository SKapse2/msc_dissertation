idf_component_register(SRCS "tier1_inference.cpp"
                       INCLUDE_DIRS "."
                       REQUIRES esp-tflite-micro esp_timer)