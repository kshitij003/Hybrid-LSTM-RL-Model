package com.trader.trading_backend.dto;

import lombok.Builder;
import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
@Builder
public class TrainingDataDTO {
    private List<String> stocks;
    private String startDate;
    private String endDate;
    private Map<String, List<StockFeatureDTO>> marketData;
    private TrainingConfig config;
    
    @Data
    @Builder
    public static class TrainingConfig {
        private Integer lstmEpochs;
        private Integer ppoTimesteps;
        private Integer sequenceLength;
        private Double initialBalance;
    }
}
