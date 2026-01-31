package com.trader.trading_backend.dto;

import lombok.Builder;
import lombok.Data;

import java.util.Map;

@Data
@Builder
public class InferenceResponseDTO {

    private String modelVersion;
    private Map<String,Double> targetWeights;
    private Double ConfidenceScore;
}
