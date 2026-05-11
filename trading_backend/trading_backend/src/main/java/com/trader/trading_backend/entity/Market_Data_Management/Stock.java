package com.trader.trading_backend.entity.Market_Data_Management;

import jakarta.persistence.*;
import lombok.*;
import java.util.List;

@Entity
@Table(name = "stocks")
@ToString
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Getter
@Setter
public class Stock {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true, nullable = false)
    private String ticker;

    private String companyName;
    private String sector;
    @Builder.Default
    private Double currentPrice = 100.0; // Default fallback for demo

    @ToString.Exclude
    @OneToMany(mappedBy = "stock", cascade = CascadeType.ALL)
    private List<StockPrice> priceHistory;

    @ToString.Exclude
    @OneToMany(mappedBy = "stock")
    private List<MarketNews> news;
}