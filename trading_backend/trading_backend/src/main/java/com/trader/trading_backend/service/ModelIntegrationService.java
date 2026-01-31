package com.trader.trading_backend.service;

import com.trader.trading_backend.Repository.AI_Model_Governance.ModelSignalRepository;
import com.trader.trading_backend.Repository.Market_Data_Management.DailyStockSummaryRepository;
import com.trader.trading_backend.Repository.Market_Data_Management.MarketNewsRepository;
import com.trader.trading_backend.Repository.Market_Data_Management.StockPriceRepository;
import com.trader.trading_backend.Repository.Market_Data_Management.StockRepository;
import com.trader.trading_backend.dto.InferenceRequestDTO;
import com.trader.trading_backend.dto.InferenceResponseDTO;
import com.trader.trading_backend.dto.StockFeatureDTO;
import com.trader.trading_backend.entity.AI_Model_Governance.ModelSignal;
import com.trader.trading_backend.entity.Market_Data_Management.DailyStockSummary;
import com.trader.trading_backend.entity.Market_Data_Management.Stock;
import com.trader.trading_backend.entity.Market_Data_Management.StockPrice;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Portfolio;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDate;
import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class ModelIntegrationService {

    private final StockRepository stockRepository;
    private final MarketNewsRepository marketNewsRepository;
    private final StockPriceRepository stockPriceRepository;
    private final DailyStockSummaryRepository dailyStockSummaryRepository;
    private final ModelSignalRepository modelSignalRepository;

    private final RestTemplate restTemplate = new RestTemplate();

    @Value("${ai.service.url:http://localhost:8000}")
    private String pythonServiceUrl;

    // The LSTM needs exactly 60 days of data to look back
    private static final int SEQUENCE_LENGTH = 60;
    @Transactional
    public Map<String, Double> getRebalancingSignals(Portfolio portfolio) {
        System.out.println("🤖 AI Inference: Preparing payload for Portfolio ID: {}"+ portfolio.getId());

        // 1. Build the Payload
        InferenceRequestDTO payload = buildInferencePayload(portfolio);

        // 2. Call Python API
        try {
            ResponseEntity<InferenceResponseDTO> response = restTemplate.postForEntity(
                    pythonServiceUrl + "/predict",
                    payload,
                    InferenceResponseDTO.class
            );

            InferenceResponseDTO result = response.getBody();
            if (result == null || result.getTargetWeights() == null) {
                throw new RuntimeException("Received empty response from AI Model");
            }

            System.out.println("AI Response received. Confidence: {}"+result.getConfidenceScore());

            // 3. Log the decision for Audit (Explainability)
            logModelDecisions(portfolio, result);

            return result.getTargetWeights();

        } catch (Exception e) {
            System.out.println("❌ AI Service Failed: {}"+e.getMessage());
            // Fallback: Return empty map (Hold current position)
            return Collections.emptyMap();
        }
    }

    /**
     * Payload Builder
     * Merges Price + News + Summary into a single timeline.
     */
    private InferenceRequestDTO buildInferencePayload(Portfolio portfolio) {
        List<Stock> allStocks = stockRepository.findAll();
        Map<String, List<StockFeatureDTO>> marketDataMap = new HashMap<>();

        for (Stock stock : allStocks) {
            // A. Fetch last 60 days of prices
            List<StockPrice> prices = stockPriceRepository.findByStockIdOrderByTimestampDesc(stock.getId(), PageRequest.of(0, SEQUENCE_LENGTH));

            // Reverse to Chronological Order (Day 1 -> Day 60) for LSTM
            Collections.reverse(prices);

            List<StockFeatureDTO> features = new ArrayList<>();

            for (StockPrice price : prices) {
                LocalDate date = price.getTimestamp().toLocalDate();

                // B. Fetch Sentiment for this specific day
                // Logic: Try finding a Summary first (Old data), if null, check raw News (Recent data)
                Double sentiment = getSentimentForDate(stock.getId(), date);

                features.add(StockFeatureDTO.builder()
                        .date(date.toString())
                        .close(price.getClosePrice().doubleValue())
                        .volume(price.getVolume().doubleValue())
                        .sentimentScore(sentiment)
                        .build());
            }
            marketDataMap.put(stock.getTicker(), features);
        }

        // C. Build Portfolio State map
        Map<String, Double> holdingsMap = portfolio.getHoldings().stream()
                .collect(Collectors.toMap(
                        h -> h.getStock().getTicker(),
                        h -> h.getQuantity().doubleValue() * h.getStock().getPriceHistory().get(0).getClosePrice().doubleValue() // Approx value
                ));

        return InferenceRequestDTO.builder()
                .currentCash(portfolio.getCurrentCashBalance())
                .currentHoldings(holdingsMap)
                .marketData(marketDataMap)
                .build();
    }

    // Helper: Smart Sentiment Lookup
    protected Double getSentimentForDate(Long stockId, LocalDate date) {
        // 1. Check Summary Table (Fast)
        Optional<DailyStockSummary> summary = dailyStockSummaryRepository.findByStockIdAndDate(stockId, date);
        if (summary.isPresent()) {
            return summary.get().getAverageSentimentScore();
        }

        return marketNewsRepository.findAverageSentimentByStockIdAndPublishedAt(stockId, date); // Default to Neutral
    }

    // Helper: Audit Logger
    private void logModelDecisions(Portfolio portfolio, InferenceResponseDTO result) {
        result.getTargetWeights().forEach((ticker, weight) -> {
            if (ticker.equals("CASH")) return; // Don't log cash as a stock signal

            Stock stock = stockRepository.findByTickerIgnoreCase(ticker).orElse(null);
            if (stock != null) {
                ModelSignal signal = ModelSignal.builder()
                        .portfolio(portfolio)
                        .stock(stock)
                        .generatedAt(java.time.LocalDateTime.now())
                        .modelVersion(result.getModelVersion())
                        .predictedConfidence(result.getConfidenceScore())
                        .targetWeight(weight)
                        .actionRecommended(weight > 0 ? com.trader.trading_backend.Enum.SignalAction.BUY : com.trader.trading_backend.Enum.SignalAction.SELL) // Simplified logic
                        .build();
                modelSignalRepository.save(signal);
            }
        });
    }
}
