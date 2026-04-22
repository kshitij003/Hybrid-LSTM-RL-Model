package com.trader.trading_backend.service;

import com.trader.trading_backend.Repository.AI_Model_Governance.ModelSignalRepository;
import com.trader.trading_backend.Repository.Portfolio_Trading_Engine.PortfolioRepository;
import com.trader.trading_backend.dto.InferenceResponseDTO;
import com.trader.trading_backend.entity.AI_Model_Governance.ModelSignal;
import com.trader.trading_backend.entity.Market_Data_Management.Stock;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Portfolio;
import com.trader.trading_backend.Repository.Market_Data_Management.StockRepository;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * ModelIntegrationService — calls the Python ML service for portfolio rebalancing signals.
 *
 * Simplified from previous version:
 *   ❌ Removed: market data fetching from DB (StockPrice, DailyStockSummary, MarketNews)
 *   ❌ Removed: sentiment lookup from DB
 *   ✅ Kept   : predict call to ML service (simplified payload — just portfolio state + tickers)
 *   ✅ Kept   : model decision audit logging (ModelSignal)
 *
 * The ML service now handles all data fetching internally via yfinance + NewsAPI/FinBERT.
 * Spring Boot only needs to send: currentCash, currentHoldings, tickers.
 */
@Service
@RequiredArgsConstructor
public class ModelIntegrationService {

    private final StockRepository       stockRepository;
    private final ModelSignalRepository modelSignalRepository;
    private final RestTemplate          restTemplate = new RestTemplate();

    @Value("${ai.service.url:http://localhost:8000}")
    private String mlServiceUrl;

    /**
     * Get AI portfolio rebalancing weights for a given portfolio.
     *
     * Sends a lightweight request to the ML service containing only:
     *   - current cash balance
     *   - current holdings (ticker → INR value)
     *   - list of tickers to consider
     *
     * The ML service fetches market data and sentiment internally.
     *
     * @param portfolio The active portfolio to rebalance
     * @return Map of ticker → target weight (0.0 to 1.0), empty on failure
     */
    @Transactional
    public Map<String, Double> getRebalancingSignals(Portfolio portfolio) {
        System.out.println("🤖 ML Inference: Requesting signals for Portfolio ID: " + portfolio.getId());

        try {
            // Build simplified payload — no market data needed
            Map<String, Object> payload = buildSimplifiedPayload(portfolio);

            ResponseEntity<InferenceResponseDTO> response = restTemplate.postForEntity(
                    mlServiceUrl + "/api/predict",
                    payload,
                    InferenceResponseDTO.class
            );

            InferenceResponseDTO result = response.getBody();
            if (result == null || result.getTargetWeights() == null) {
                throw new RuntimeException("Empty response from ML service");
            }

            System.out.println("✅ ML Response received. Confidence: " + result.getConfidenceScore());

            // Audit log — persist model decision for explainability
            logModelDecision(portfolio, result);

            return result.getTargetWeights();

        } catch (Exception e) {
            System.err.println("❌ ML service call failed: " + e.getMessage());
            // Fallback: hold current position (return empty weights map)
            return Collections.emptyMap();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Build the simplified predict payload.
     * ML service fetches OHLCV + sentiment internally — we only send portfolio state.
     */
    private Map<String, Object> buildSimplifiedPayload(Portfolio portfolio) {

        // Current holdings: ticker → approximate INR value of position
        Map<String, Double> holdingsMap = portfolio.getHoldings().stream()
                .collect(Collectors.toMap(
                        h -> h.getStock().getTicker(),
                        h -> h.getQuantity().doubleValue()
                             * h.getCurrentPrice().doubleValue()
                ));

        // Tickers to run inference on (all stocks in portfolio)
        List<String> tickers = portfolio.getHoldings().stream()
                .map(h -> h.getStock().getTicker())
                .collect(Collectors.toList());

        // If portfolio has no holdings yet, use all registered stocks
        if (tickers.isEmpty()) {
            tickers = stockRepository.findAll().stream()
                    .map(Stock::getTicker)
                    .collect(Collectors.toList());
        }

        return Map.of(
                "currentCash",     portfolio.getCurrentCashBalance(),
                "currentHoldings", holdingsMap,
                "tickers",         tickers
        );
    }

    /**
     * Persist each model weight decision as a ModelSignal for audit / explainability.
     */
    private void logModelDecision(Portfolio portfolio, InferenceResponseDTO result) {
        result.getTargetWeights().forEach((ticker, weight) -> {
            if ("CASH".equals(ticker)) return;

            stockRepository.findByTickerIgnoreCase(ticker).ifPresent(stock -> {
                ModelSignal signal = ModelSignal.builder()
                        .portfolio(portfolio)
                        .stock(stock)
                        .generatedAt(java.time.LocalDateTime.now())
                        .modelVersion(result.getModelVersion())
                        .predictedConfidence(result.getConfidenceScore())
                        .targetWeight(weight)
                        .actionRecommended(
                            weight > 0
                                ? com.trader.trading_backend.Enum.SignalAction.BUY
                                : com.trader.trading_backend.Enum.SignalAction.SELL
                        )
                        .build();
                modelSignalRepository.save(signal);
            });
        });
    }
}
