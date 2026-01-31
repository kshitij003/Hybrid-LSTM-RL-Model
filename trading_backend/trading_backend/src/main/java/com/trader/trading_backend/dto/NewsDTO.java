package com.trader.trading_backend.dto;

import lombok.Builder;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@Builder
public class NewsDTO {
    private String headline;
    private String source;
    private String url;
    private LocalDateTime publishedAt;
}