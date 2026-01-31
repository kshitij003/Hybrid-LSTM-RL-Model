package com.trader.trading_backend.Repository.Market_Data_Management;

import com.trader.trading_backend.entity.Market_Data_Management.DailyStockSummary;
import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDate;
import java.util.Optional;

public interface DailyStockSummaryRepository extends JpaRepository<DailyStockSummary, Long> {

    Optional<DailyStockSummary> findByStockIdAndDate(Long stockId, LocalDate date);
}
