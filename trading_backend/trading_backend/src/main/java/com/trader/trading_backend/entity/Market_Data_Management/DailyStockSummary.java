package com.trader.trading_backend.entity.Market_Data_Management;


import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDate;

@Entity
@Table(name = "daily_stock_summaries", indexes = {
        @Index(name = "idx_summary_stock_date", columnList = "stock_id, date")
})
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DailyStockSummary {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private LocalDate date;

    private Double averageSentimentScore;

    private Integer newsCount;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "stock_id", nullable = false)
    private Stock stock;
}
