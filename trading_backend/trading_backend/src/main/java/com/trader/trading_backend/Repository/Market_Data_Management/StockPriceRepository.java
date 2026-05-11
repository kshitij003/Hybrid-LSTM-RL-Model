package com.trader.trading_backend.Repository.Market_Data_Management;

import com.trader.trading_backend.entity.Market_Data_Management.StockPrice;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface StockPriceRepository extends JpaRepository<StockPrice, Long> {
    
    /**
     * Used by PortfolioService to get the latest market price for NAV calculations.
     */
    List<StockPrice> findByStockIdOrderByTimestampDesc(Long stockId, Pageable pageable);
    
    /**
     * Used for fetching historical data for training/inference.
     */
    List<StockPrice> findByStockTickerOrderByTimestampDesc(String ticker, Pageable pageable);
}
