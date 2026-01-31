package com.trader.trading_backend.entity.AI_Model_Governance;
import com.trader.trading_backend.Enum.SignalAction;
import com.trader.trading_backend.entity.Market_Data_Management.Stock;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Portfolio;
import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "model_signals")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ModelSignal {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private LocalDateTime generatedAt;

    private String modelVersion; // e.g. "LSTM-RL-v2"

    // The RL Agent's desired weight for this stock (e.g. 0.15 for 15%)
    private Double targetWeight;

    @Enumerated(EnumType.STRING)
    private SignalAction actionRecommended;

    private Double predictedConfidence;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "stock_id")
    private Stock stock;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "portfolio_id")
    private Portfolio portfolio;
}
