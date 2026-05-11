package com.trader.trading_backend.entity.Portfolio_Trading_Engine;
import com.trader.trading_backend.Enum.TransactionType;
import com.trader.trading_backend.entity.Market_Data_Management.Stock;
import jakarta.persistence.*;
import lombok.*;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Entity
@Table(name = "transactions")
@Getter
@Setter
@ToString
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Transaction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Enumerated(EnumType.STRING)
    private TransactionType type;

    private LocalDateTime transactionDate;

    private Integer quantity;

    @Column(precision = 19, scale = 4)
    private BigDecimal priceAtTransaction;

    @Column(precision = 19, scale = 4)
    private BigDecimal totalAmount;

    @ToString.Exclude
    @com.fasterxml.jackson.annotation.JsonIgnore
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "portfolio_id", nullable = false)
    private Portfolio portfolio;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "stock_id", nullable = false)
    private Stock stock;
}