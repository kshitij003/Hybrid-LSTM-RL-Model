package com.trader.trading_backend.entity.Market_Data_Management;

import com.trader.trading_backend.Enum.SentimentLabel;
import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "market_news")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MarketNews {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String headline;
    private String source;
    private LocalDateTime publishedAt;

    private Double sentimentScore;

    @Enumerated(EnumType.STRING)
    private SentimentLabel sentimentLabel;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "stock_id")
    private Stock stock;
}