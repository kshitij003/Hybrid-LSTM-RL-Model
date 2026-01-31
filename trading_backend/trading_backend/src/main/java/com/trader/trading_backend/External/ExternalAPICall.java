package com.trader.trading_backend.External;

import com.trader.trading_backend.dto.NewsDTO;
import com.trader.trading_backend.dto.StockPriceDTO;
import org.slf4j.Logger;
import org.springframework.stereotype.Component;
import yahoofinance.Stock;
import yahoofinance.YahooFinance;
import yahoofinance.histquotes.HistoricalQuote;
import yahoofinance.histquotes.Interval;

import java.time.LocalDate;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

@Component
public class ExternalAPICall {
//    private final RestTemplate restTemplate = new RestTemplate();
//    private final String PYTHON_SERVICE_URL = "http://localhost:8000/analyze-sentiment";

    public List<StockPriceDTO> fetchHistory(String ticker, LocalDate startDate) {
        List<StockPriceDTO> result = new ArrayList<>();
        try {
            // Convert LocalDate to Calendar (Library requirement)
            Calendar from = Calendar.getInstance();
            from.setTime(java.sql.Date.valueOf(startDate));

            Calendar to = Calendar.getInstance(); // NOW

            // Fetch Data
            Stock stock = YahooFinance.get(ticker);
            List<HistoricalQuote> history = stock.getHistory(from, to, Interval.DAILY);

            // Map Library Object -> Your DTO
            for (HistoricalQuote quote : history) {
                if (quote.getClose() == null) continue; // Skip incomplete data

                // Convert Calendar -> LocalDateTime
                LocalDate date = quote.getDate().getTime().toInstant()
                        .atZone(ZoneId.systemDefault()).toLocalDate();

                StockPriceDTO dto = StockPriceDTO.builder()
                        .timestamp(date.atStartOfDay()) // Store as start of day
                        .open(quote.getOpen().doubleValue())
                        .high(quote.getHigh().doubleValue())
                        .low(quote.getLow().doubleValue())
                        .close(quote.getClose().doubleValue())
                        .volume(quote.getVolume())
                        .build();

                result.add(dto);
            }
        } catch (Exception e) {
            System.out.println("Error fetching history for {}: {}"+ticker+" "+e.getMessage());
        }
        return result;
    }
}
