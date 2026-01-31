package com.trader.trading_backend.Repository.Market_Data_Management;

import com.trader.trading_backend.entity.Market_Data_Management.StockPrice;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface StockPriceRepository extends JpaRepository<StockPrice,Long> {

    @Query("select MAX(sp.timestamp) from StockPrice sp where sp.stock.id=?1")
    LocalDateTime findLastTimeById(@Param("stockId") Long stockId);

    List<StockPrice> findByStockIdOrderByTimestampDesc(Long stockId, Pageable pageable);
    
    // For training data export - fetch all prices in date range
    List<StockPrice> findByStockIdAndTimestampBetween(Long stockId, LocalDateTime startDate, LocalDateTime endDate);
}
