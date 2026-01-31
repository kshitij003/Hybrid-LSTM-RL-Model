package com.trader.trading_backend.dto;
// src/main/java/com/project/dto/DailySentimentDTO.java

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import java.time.LocalDate;

@Data
@AllArgsConstructor
@Builder
public class DailySentimentDTO {
    private Long stockId;
    private LocalDate date;
    private Double avgScore;
    private Long count;
}
