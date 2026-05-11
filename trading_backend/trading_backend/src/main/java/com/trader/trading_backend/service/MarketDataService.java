package com.trader.trading_backend.service;

import com.trader.trading_backend.Repository.Market_Data_Management.StockPriceRepository;
import com.trader.trading_backend.Repository.Market_Data_Management.StockRepository;
import com.trader.trading_backend.entity.Market_Data_Management.Stock;
import com.trader.trading_backend.entity.Market_Data_Management.StockPrice;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import yahoofinance.YahooFinance;
import yahoofinance.quotes.stock.StockQuote;

import java.io.IOException;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class MarketDataService {

    private final StockRepository stockRepo;
    private final StockPriceRepository priceRepo;

    /**
     * Fetch latest prices from Yahoo Finance and save to DB.
     * Ensures we are NOT using stale data for NAV/Inference.
     */
    @Transactional
    public void syncstockprices(List<String> tickers) {
        for (String ticker : tickers) {
            try {
                yahoofinance.Stock yStock = YahooFinance.get(ticker);
                if (yStock == null) continue;

                StockQuote quote = yStock.getQuote();
                
                // Find or Create the Stock entity
                Stock stock = stockRepo.findByTickerIgnoreCase(ticker)
                        .orElseGet(() -> stockRepo.save(Stock.builder()
                                .ticker(ticker.toUpperCase())
                                .companyName(yStock.getName())
                                .build()));

                // Update the quick-access currentPrice field in Stock entity
                BigDecimal latestPrice = quote.getPrice();
                if (latestPrice != null) {
                    stock.setCurrentPrice(latestPrice.doubleValue());
                    stockRepo.save(stock);
                }

                // Save fresh price point in history
                StockPrice price = StockPrice.builder()
                        .stock(stock)
                        .timestamp(LocalDateTime.now())
                        .openPrice(quote.getOpen())
                        .highPrice(quote.getDayHigh())
                        .lowPrice(quote.getDayLow())
                        .closePrice(latestPrice) 
                        .volume(quote.getVolume())
                        .build();

                priceRepo.save(price);
                System.out.println("✅ Synced fresh data for " + ticker + " | Price: $" + latestPrice);

            } catch (IOException e) {
                System.err.println("❌ Failed to sync " + ticker + ": " + e.getMessage());
            }
        }
    }
}
