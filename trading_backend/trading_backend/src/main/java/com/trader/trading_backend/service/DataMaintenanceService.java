package com.trader.trading_backend.service;

import com.trader.trading_backend.Repository.Market_Data_Management.DailyStockSummaryRepository;
import com.trader.trading_backend.Repository.Market_Data_Management.MarketNewsRepository;
import com.trader.trading_backend.Repository.Market_Data_Management.StockRepository;
import com.trader.trading_backend.dto.DailySentimentDTO;
import com.trader.trading_backend.entity.Market_Data_Management.DailyStockSummary;
import com.trader.trading_backend.entity.Market_Data_Management.Stock;
import jakarta.transaction.Transactional;
import lombok.AllArgsConstructor;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class DataMaintenanceService {

    private final MarketNewsRepository marketNewsRepository;
    private final StockRepository stockRepository;
    private final DailyStockSummaryRepository dailyStockSummaryRepository;

    private static final int retention_days = 30;

    @Transactional
    public void compressoldnews(){

        LocalDateTime now = LocalDateTime.now().minusDays(retention_days);
        System.out.println("Compressing old news for "+now);

        List<DailySentimentDTO> sentiments = marketNewsRepository.findSentimentAggregates(now);

        if(sentiments.isEmpty()){
            System.out.println("No sentiments found");
            return;
        }

        List<DailyStockSummary> summaries = sentiments.stream()
                .map(dto -> {
                    Stock stock = stockRepository.getReferenceById(dto.getStockId());
                    return DailyStockSummary.builder()
                            .stock(stock)
                            .date(dto.getDate())
                            .averageSentimentScore(dto.getAvgScore())
                            .newsCount(dto.getCount().intValue())
                            .build();
                })
                .collect(Collectors.toList());

        dailyStockSummaryRepository.saveAll(summaries);
        System.out.println("Saved {} summary records."+ summaries.size());

        marketNewsRepository.deleteAllBefore(now);
        System.out.println("🗑️ Deleted raw news entries older than {}."+ now);
    }
}
