package com.trader.trading_backend.entity.Market_Data_Management;

import jakarta.persistence.*;
import lombok.*;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Entity
@Table(name = "stock_prices")
@ToString
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Getter
@Setter
public class StockPrice {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ToString.Exclude
    @com.fasterxml.jackson.annotation.JsonIgnore
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "stock_id", nullable = false)
    private Stock stock;

    @Column(nullable = false)
    private LocalDateTime timestamp;

    @Column(nullable = false, precision = 19, scale = 4)
    private BigDecimal openPrice;

    @Column(nullable = false, precision = 19, scale = 4)
    private BigDecimal highPrice;

    @Column(nullable = false, precision = 19, scale = 4)
    private BigDecimal lowPrice;

    @Column(nullable = false, precision = 19, scale = 4)
    private BigDecimal closePrice;

    private Long volume;
}
