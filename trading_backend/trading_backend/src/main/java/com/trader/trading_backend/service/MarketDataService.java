package com.trader.trading_backend.service;

import com.trader.trading_backend.External.ExternalAPICall;

import com.trader.trading_backend.Repository.Market_Data_Management.StockPriceRepository;
import com.trader.trading_backend.Repository.Market_Data_Management.StockRepository;
import com.trader.trading_backend.dto.StockFeatureDTO;
import com.trader.trading_backend.dto.StockPriceDTO;
import com.trader.trading_backend.entity.Market_Data_Management.Stock;
import com.trader.trading_backend.entity.Market_Data_Management.StockPrice;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.*;

@Service
@RequiredArgsConstructor
public class MarketDataService {

    private final ExternalAPICall externalAPICall;
    private final StockRepository stockRepository;
    private final StockPriceRepository stockPriceRepository;
    private final ModelIntegrationService mis;

    @Transactional
    public void syncstockprices(List<String> tickers){

        for(String ticker:tickers) {
            try {
                Stock stock = stockRepository
                        .findByTickerIgnoreCase(ticker)
                        .orElseThrow(() -> new RuntimeException("Stock not found in DB: " + ticker));

                LocalDateTime date = stockPriceRepository.findLastTimeById(stock.getId());

                LocalDate start_date = (date == null)
                        ? LocalDate.now().minusYears(2)
                        : date.toLocalDate().plusDays(1);

                if (start_date.isAfter(LocalDate.now())) {
                    System.out.println("Start date is after end date");
                    continue;
                }

                System.out.println("Start date: " + start_date);
                System.out.println("Fetching data...");

                List<StockPriceDTO> price = externalAPICall.fetchHistory(ticker, start_date);

                if(price.isEmpty()){
                    System.out.println("No history found for ticker: " + ticker);
                    continue;
                }
                savePrices(stock,price);

            }catch (Exception e){
                throw new RuntimeException(e);
            }
        }
        System.out.println("Sync stock prices completed");
    }

    @Transactional
    public void savePrices(Stock stock, List<StockPriceDTO> price){
        List<StockPrice> entities = price.stream()
                .map(dto -> StockPrice.builder()
                        .stock(stock)
                        .timestamp(dto.getTimestamp()) // Ensure API returns LocalDateTime
                        .openPrice(BigDecimal.valueOf(dto.getOpen()))
                        .highPrice(BigDecimal.valueOf(dto.getHigh()))
                        .lowPrice(BigDecimal.valueOf(dto.getLow()))
                        .closePrice(BigDecimal.valueOf(dto.getClose()))
                        .volume(dto.getVolume())
                        .build())
                .toList();

        stockPriceRepository.saveAll(entities);
        System.out.println("Saving prices completed");
    }

    /**
     * TASK 3: Prepare Data Payload for AI or Frontend Chart
     * Gathers the last N days of data.
     */
    @Transactional(readOnly = true)
    public Map<String, Object> getLatestFeatures(List<String> tickers, int sequenceLength) {
        Map<String, Object> payload = new HashMap<>();

        // Define the limit (e.g., Top 60 rows)
        PageRequest limit = PageRequest.of(0, sequenceLength);

        for (String ticker : tickers) {
            // 1. Find Stock
            Stock stock = stockRepository.findByTickerIgnoreCase(ticker)
                    .orElseThrow(() -> new RuntimeException("Stock not found: " + ticker));

            // 2. Fetch last N prices efficiently (Using Pageable)
            List<StockPrice> prices = stockPriceRepository.findByStockIdOrderByTimestampDesc(stock.getId(), limit);

            // Reverse list so it goes from Oldest -> Newest (Day 1 to Day 60)
            // This is crucial for both Charts and LSTM
            Collections.reverse(prices);

            List<StockFeatureDTO> features = new ArrayList<>();

            for (StockPrice price : prices) {
                LocalDate date = price.getTimestamp().toLocalDate();

                // 3. Smart Sentiment Lookup (Check Summary -> Then Raw News)
                Double sentiment = mis.getSentimentForDate(stock.getId(), date);

                features.add(StockFeatureDTO.builder()
                        .date(date.toString())
                        .close(price.getClosePrice().doubleValue()) // Corrected getter
                        .volume(price.getVolume().doubleValue())
                        .sentimentScore(sentiment)
                        .build());
            }

            // 4. Add to payload map
            payload.put(ticker, features);
        }
        return payload;
    }

    /**
     * Export training data from database for ML service
     * Fetches historical data with sentiment scores
     */
    @Transactional(readOnly = true)
    public com.trader.trading_backend.dto.TrainingDataDTO exportTrainingData(
            List<String> tickers,
            LocalDate startDate,
            LocalDate endDate
    ) {
        System.out.println("📊 Exporting training data: " + tickers);
        System.out.println("   Date range: " + startDate + " to " + endDate);

        Map<String, List<StockFeatureDTO>> marketData = new HashMap<>();

        for (String ticker : tickers) {
            // 1. Find stock
            Stock stock = stockRepository.findByTickerIgnoreCase(ticker)
                    .orElseThrow(() -> new RuntimeException("Stock not found: " + ticker));

            // 2. Fetch all prices in date range
            List<StockPrice> prices = stockPriceRepository
                    .findByStockIdAndTimestampBetween(
                            stock.getId(),
                            startDate.atStartOfDay(),
                            endDate.atTime(23, 59, 59)
                    );

            // 3. Convert to features with sentiment
            List<StockFeatureDTO> features = new ArrayList<>();

            for (StockPrice price : prices) {
                LocalDate date = price.getTimestamp().toLocalDate();

                // Get sentiment for this date
                Double sentiment = mis.getSentimentForDate(stock.getId(), date);

                features.add(StockFeatureDTO.builder()
                        .date(date.toString())
                        .close(price.getClosePrice().doubleValue())
                        .volume(price.getVolume().doubleValue())
                        .sentimentScore(sentiment != null ? sentiment : 0.0)
                        .build());
            }

            System.out.println("   " + ticker + ": " + features.size() + " days");
            marketData.put(ticker, features);
        }

        return com.trader.trading_backend.dto.TrainingDataDTO.builder()
                .stocks(tickers)
                .startDate(startDate.toString())
                .endDate(endDate.toString())
                .marketData(marketData)
                .build();
    }

    // Helper: Smart Sentiment Lookup
//    private Double getSentimentForDate(Long stockId, LocalDate date) {
//        // A. Check Compressed/Old Data Table
//        Optional<DailyStockSummary> summary = summaryRepo.findByStockIdAndDate(stockId, date);
//        if (summary.isPresent()) {
//            return summary.get().getAverageSentimentScore();
//        }
//
//        // B. If not found (meaning it's recent data), check Raw News Table
//
//        // Return 0.0 (Neutral) if no news exists for that day
//        return newsRepo.findAverageSentimentByStockIdAndDate(stockId, date);
//    }
}
